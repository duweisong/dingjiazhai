"""
分析仓位信号与后续收益的关系 —— 回答"如何让仓位预测指导盈利"
"""
import sys, os
sys.path.insert(0, 'C:/AI')
sys.path.insert(0, 'C:/AI/nt-position-sizer/src')
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import numpy as np
import pandas as pd
from pathlib import Path
from predictive_engine import (
    load_market_data, load_nt_conviction, compute_market_features,
    align_quarterly, FEATURE_COLS,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('C:/AI')

def analyze():
    # 1. 加载数据
    market = load_market_data()
    features = compute_market_features(market)
    conviction = load_nt_conviction()
    aligned = align_quarterly(features, conviction)

    # 2. 历史滚动预测（模拟实盘可得的预测仓位时间序列）
    print('=' * 65)
    print('  仓位信号 vs 后续收益 量化分析')
    print('=' * 65)

    records = []
    for i in range(3, len(aligned)):
        train_data = aligned.iloc[:i]
        valid = train_data.dropna(subset=['target_conviction'] + FEATURE_COLS)
        if len(valid) < 3:
            continue

        X_train = valid[FEATURE_COLS].values
        y_train = valid['target_conviction'].values
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=0.5).fit(scaler.transform(X_train), y_train)

        feat = aligned.iloc[i][FEATURE_COLS].to_dict()
        X_pred = np.array([[feat.get(c, 0) for c in FEATURE_COLS]])
        pred_conv = float(model.predict(scaler.transform(X_pred))[0])

        # 三因子→仓位
        actual_conv = aligned.iloc[i]['conviction']
        trend = actual_conv - (aligned.iloc[i-1]['conviction'] if i > 0 else actual_conv)

        conv_norm = np.clip(pred_conv / 0.4, -1, 1)
        trend_norm = np.clip(trend / 0.3, -1, 1)
        raw = 0.50 * conv_norm + 0.30 * trend_norm + 0.20 * 0
        pred_pos = float(1.0 / (1.0 + np.exp(-3.0 * raw)))

        records.append({
            'quarter': aligned.index[i],
            'pred_position': np.clip(pred_pos, 0.02, 0.98),
            'pred_conviction': pred_conv,
            'actual_conviction': actual_conv,
        })

    pos_df = pd.DataFrame(records)

    # 3. 对齐季度收益
    hs300_q = market['close'].resample('QE').last().pct_change()

    returns = []
    for _, row in pos_df.iterrows():
        q = row['quarter']
        yr, qn = q[:4], q[-1]
        date_str = yr + {'1': '0331', '2': '0630', '3': '0930', '4': '1231'}[qn]
        ts = pd.Timestamp(date_str)
        # 下一季度收益（仓位信号生效的季度）
        next_ts = ts + pd.DateOffset(months=3)
        if next_ts in hs300_q.index:
            returns.append(hs300_q[next_ts])
        else:
            returns.append(np.nan)

    pos_df['forward_return'] = returns
    pos_df = pos_df.dropna(subset=['forward_return'])

    # 4. 分析：仓位分档后的收益表现
    pos_df['position_bucket'] = pd.cut(pos_df['pred_position'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['0-30%(防御)', '30-50%(保守)', '50-70%(中性)', '70-100%(积极)'])

    print('\n--- 各仓位档位的下季度平均收益 ---')
    bucket_stats = pos_df.groupby('position_bucket', observed=False)['forward_return'].agg(['mean', 'std', 'count'])
    bucket_stats['mean'] = bucket_stats['mean'] * 100
    bucket_stats['std'] = bucket_stats['std'] * 100
    bucket_stats['win_rate'] = pos_df.groupby('position_bucket', observed=False)['forward_return'].apply(
        lambda x: (x > 0).sum() / len(x) * 100)
    print(bucket_stats.to_string())

    # 5. 相关性
    corr = pos_df['pred_position'].corr(pos_df['forward_return'])
    print(f'\n预测仓位 vs 下季度收益 相关系数: {corr:.4f}')

    # 6. 模拟不同策略的累计收益
    print('\n--- 策略对比（样本外滚动） ---')

    # 策略 A: 固定满仓
    nav_full = 1.0
    # 策略 B: 预测仓位缩放
    nav_pred = 1.0
    # 策略 C: 预测仓位 + 阈值过滤（仓位<30%时空仓）
    nav_filtered = 1.0
    # 策略 D: 非对称（高仓位时加杠杆1.2x，低仓位时0.5x）
    nav_asym = 1.0

    for _, row in pos_df.iterrows():
        r = row['forward_return']
        pos = row['pred_position']

        nav_full *= (1 + r)
        nav_pred *= (1 + r * pos)
        nav_filtered *= (1 + r * (pos if pos >= 0.30 else 0))
        nav_asym *= (1 + r * (pos * 1.3 if pos > 0.60 else pos * 0.5))

    yrs = len(pos_df) / 4
    print(f'  {"策略":<25} {"年化":>8} {"vs满仓":>8}')
    print(f'  {"-"*41}')
    for name, nav in [('满仓持有', nav_full), ('预测仓位缩放', nav_pred),
                       ('预测+30%阈值过滤', nav_filtered), ('非对称(高倍低减)', nav_asym)]:
        cagr = nav ** (1/max(yrs,0.1)) - 1
        ex = cagr - (nav_full ** (1/max(yrs,0.1)) - 1)
        print(f'  {name:<25} {cagr:>+7.2%} {ex:>+7.2%}')

    # 7. 关键发现：仓位信号的边际价值在哪
    print('\n--- 关键发现 ---')
    high_pos = pos_df[pos_df['pred_position'] > 0.60]
    low_pos = pos_df[pos_df['pred_position'] < 0.40]

    if len(high_pos) > 0:
        print(f'  高仓位(>60%)季度: {len(high_pos)}个, 平均收益 {high_pos["forward_return"].mean():+.2%}, 胜率 {(high_pos["forward_return"]>0).mean():.0%}')
    if len(low_pos) > 0:
        print(f'  低仓位(<40%)季度: {len(low_pos)}个, 平均收益 {low_pos["forward_return"].mean():+.2%}, 胜率 {(low_pos["forward_return"]>0).mean():.0%}')

    # 8. 优化建议
    print('\n--- 盈利导向优化方向 ---')
    if corr > 0.1:
        print(f'  [有效] 仓位信号与收益正相关(r={corr:.3f}) → 仓位缩放有效')
    elif corr > 0:
        print(f'  [弱效] 仓位信号与收益弱正相关(r={corr:.3f}) → 需增强')
    else:
        print(f'  [反向] 仓位信号与收益负相关(r={corr:.3f}) → 需重构')

    print(f'  建议方向:')
    print(f'    1. 非对称缩放：高置信度时适度加杠杆(1.2-1.5x)，低时大幅减仓')
    print(f'    2. 阈值过滤：仓位<30%时完全空仓（避免在熊市损耗）')
    print(f'    3. 收益直预：将预测目标从"NT信心"改为"下季度收益"')
    print(f'    4. 多信号融合：仓位信号 + 市场技术指标(MA60/RSI) 联合决策')

    return pos_df

if __name__ == '__main__':
    analyze()
