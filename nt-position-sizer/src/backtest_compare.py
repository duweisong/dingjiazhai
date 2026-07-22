"""
预测仓位 vs 实际仓位 vs 满仓 — 历史回测对比

策略:
  P1: 预测仓位（K线拟合，领先1季）→ 按预测仓位配置ETF
  P2: 实际仓位（跟季报，滞后）→ 按实际NT信心配置ETF
  P3: 满仓（始终100%）
  P4: 50-50 基准（始终50%仓位）

模拟资产: HS300 ETF (510300) 季度收益
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from predictive_engine import (
    load_market_data, load_nt_conviction, compute_market_features,
    align_quarterly, train_model, predict_conviction, conviction_to_position,
    FEATURE_COLS,
)

PARENT = Path(__file__).parent.parent.parent


def run_full_backtest():
    print('[Data] Loading...')
    market = load_market_data()
    features = compute_market_features(market)
    conviction = load_nt_conviction()
    aligned = align_quarterly(features, conviction)

    # 季度收益
    hs300_q = market['close'].resample('QE').last().pct_change()

    # 滚动预测：每季度只用该季度之前的数据训练
    results = []
    nav_pred = nav_actual = nav_full = nav_half = 1.0

    for i in range(3, len(aligned)):  # 从第4个季度开始（需要足够训练数据）
        train_data = aligned.iloc[:i]  # 只用当前之前的数据

        # 训练
        valid_train = train_data.dropna(subset=['target_conviction'] + FEATURE_COLS)
        if len(valid_train) < 3:
            continue

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_train = valid_train[FEATURE_COLS].values
        y_train = valid_train['target_conviction'].values
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=0.5).fit(scaler.transform(X_train), y_train)

        # 预测下一季度
        current_feat = aligned.iloc[i][FEATURE_COLS].to_dict()
        X_pred = np.array([[current_feat.get(c, 0) for c in FEATURE_COLS]])
        predicted_conv = float(model.predict(scaler.transform(X_pred))[0])

        # 实际信心
        actual_conv = aligned.iloc[i]['conviction']

        # 仓位
        pred_pos = conviction_to_position(predicted_conv)
        actual_pos = conviction_to_position(actual_conv)

        # 获取下季度实际收益
        q_label = aligned.index[i]
        yr = q_label[:4]
        q_num = q_label[-1]
        next_q = f'{yr}Q{int(q_num)+1}' if int(q_num) < 4 else f'{int(yr)+1}Q1'
        # 映射到日期
        month_map = {'1': '0331', '2': '0630', '3': '0930', '4': '1231'}
        date_str = yr + month_map[q_num]
        ts = pd.Timestamp(date_str)
        next_ts = ts + pd.DateOffset(months=3)

        # 找实际可用的收益
        q_ret = 0.0
        if next_ts in hs300_q.index:
            q_ret = hs300_q[next_ts]
        elif ts in hs300_q.index:
            q_ret = hs300_q[ts]
        else:
            continue

        # 更新净值
        nav_pred *= (1 + q_ret * pred_pos)
        nav_actual *= (1 + q_ret * actual_pos)
        nav_full *= (1 + q_ret)
        nav_half *= (1 + q_ret * 0.50)

        results.append({
            'quarter': q_label,
            'q_return': f'{q_ret:+.2%}',
            'pred_conv': round(predicted_conv, 3),
            'actual_conv': round(actual_conv, 3),
            'pred_pos': f'{pred_pos:.0%}',
            'actual_pos': f'{actual_pos:.0%}',
            'nav_pred': round(nav_pred, 4),
            'nav_actual': round(nav_actual, 4),
            'nav_full': round(nav_full, 4),
        })

    # 计算年化
    years = len(results) / 4
    cagr = lambda nav: nav ** (1 / max(years, 0.1)) - 1

    # 打印
    print()
    print(f'{"Quarter":<10} {"Q_Ret":>8} {"Pred":>8} {"Actual":>8} {"P_Pos":>6} {"A_Pos":>6} {"N_Pred":>8} {"N_Act":>8} {"N_Full":>8}')
    print('-' * 80)
    for r in results:
        print(f'{r["quarter"]:<10} {r["q_return"]:>8} {r["pred_conv"]:>+8.3f} {r["actual_conv"]:>+8.3f} '
              f'{r["pred_pos"]:>6} {r["actual_pos"]:>6} '
              f'{r["nav_pred"]:>8.4f} {r["nav_actual"]:>8.4f} {r["nav_full"]:>8.4f}')

    last = results[-1] if results else None
    print()
    print('=' * 60)
    print(f'  Backtest: {len(results)} quarters ({years:.1f} years)')
    print('=' * 60)
    print(f'  {"Strategy":<20} {"Final NAV":>10} {"CAGR":>10}')
    print(f'  {"-"*40}')
    for name, nav in [('Predicted Position', nav_pred), ('Actual (Lagged)', nav_actual),
                       ('Full Position', nav_full), ('50-50 Baseline', nav_half)]:
        print(f'  {name:<20} {nav:>10.4f} {cagr(nav):>+9.2%}')

    # 预测准确性
    if results:
        errors = [abs(float(r['pred_conv']) - float(r['actual_conv'])) for r in results]
        print(f'\n  预测误差 MAE: {np.mean(errors):.3f}')
        direction_correct = sum(
            (float(r['pred_conv']) > 0) == (float(r['actual_conv']) > 0)
            for r in results
        )
        print(f'  方向正确率: {direction_correct}/{len(results)} ({direction_correct/len(results):.0%})')

    return results


if __name__ == '__main__':
    run_full_backtest()
