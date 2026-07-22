"""
预测性仓位引擎 — 用沪指K线拟合国家队行为，提前预警

核心思路:
  国家队行为是市场条件的反应函数。如果能建模"什么市场条件触发国家队买卖"，
  就可以用当前K线数据预测国家队下一步动作，不等季报滞后 1-2 个月。

模型: Market Features(t) → Predicted NT Conviction(t+1) → Position Size

用法:
  python predictive_engine.py                # 当前预测仓位
  python predictive_engine.py --backtest      # 回测
  python predictive_engine.py --compare       # 对比: 预测仓位 vs 实际仓位 vs 满仓
"""

import sys, os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

PARENT = Path(__file__).parent.parent.parent
CACHE_DIR = PARENT / '.cache' / 'national_team'
sys.path.insert(0, str(Path(__file__).parent))  # for position_engine import

# ============================================================
# 1. 数据准备
# ============================================================

def load_market_data() -> pd.DataFrame:
    """加载 HS300 ETF 日线数据"""
    etf_cache = PARENT / '.cache' / 'etf_sector'
    hs300_path = etf_cache / 'hs300_2022.parquet'
    if hs300_path.exists():
        return pd.read_parquet(hs300_path)

    # Fallback: fetch directly
    import efinance as ef
    df = ef.fund.get_quote_history('510300')
    df = df.rename(columns={'日期': 'date', '累计净值': 'close'})
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
    df = df.sort_values('date').set_index('date')
    df = df.loc['20220101':'20260612']
    return df


def load_nt_conviction() -> pd.DataFrame:
    """加载国家队季度信心指数"""
    from position_engine import load_conviction_history
    return load_conviction_history()


# ============================================================
# 2. 特征工程
# ============================================================

def compute_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    从日线计算市场特征矩阵。

    Returns DataFrame index=date, columns=features
    """
    close = market['close']
    df = pd.DataFrame(index=market.index)

    # 价格动量
    df['ret_5d'] = close.pct_change(5)
    df['ret_20d'] = close.pct_change(20)
    df['ret_60d'] = close.pct_change(60)

    # MA 位置
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    df['dist_ma20'] = (close - ma20) / ma20
    df['dist_ma60'] = (close - ma60) / ma60
    df['dist_ma120'] = (close - ma120) / ma120

    # MA 趋势（MA 本身的方向）
    df['ma20_slope'] = ma20.pct_change(20)
    df['ma60_slope'] = ma60.pct_change(20)

    # 相对强弱
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 波动率
    df['vol_20d'] = close.pct_change().rolling(20).std()
    df['vol_60d'] = close.pct_change().rolling(60).std()

    # 距高点距离
    df['hh_60d'] = close / close.rolling(60).max() - 1
    df['hh_120d'] = close / close.rolling(120).max() - 1

    # 距低点距离
    df['ll_60d'] = close / close.rolling(60).min() - 1

    # 市场广度模拟（价格变化的方向持续性）
    df['up_days_20'] = (close.pct_change() > 0).rolling(20).sum() / 20

    return df


def align_quarterly(features: pd.DataFrame, conviction: pd.DataFrame) -> pd.DataFrame:
    """
    将季度信心指数与季度末的市场特征对齐。

    每个季度取最后交易日的特征，对应当季 NT 信心指数。
    同时创建"预测"标签：用本季特征预测下季 NT 信心。
    """
    rows = []
    for i, row in conviction.iterrows():
        q_end_str = row['date']
        q_end = pd.Timestamp(q_end_str)

        # 找最近交易日
        available = features.index[features.index <= q_end]
        if len(available) == 0:
            continue
        feat_date = available[-1]

        feat = features.loc[feat_date].to_dict()
        feat['quarter'] = row['quarter']
        feat['conviction'] = row['conviction']

        rows.append(feat)

    df = pd.DataFrame(rows)
    df = df.set_index('quarter')

    # 创建预测目标：下季度的 conviction
    df['target_conviction'] = df['conviction'].shift(-1)

    return df


# ============================================================
# 3. 预测模型
# ============================================================

FEATURE_COLS = [
    'ret_5d', 'ret_20d', 'ret_60d',
    'dist_ma20', 'dist_ma60', 'dist_ma120',
    'ma20_slope', 'ma60_slope',
    'rsi_14', 'vol_20d', 'vol_60d',
    'hh_60d', 'hh_120d', 'll_60d', 'up_days_20',
]


def train_model(aligned: pd.DataFrame):
    """用历史数据训练预测模型"""
    train = aligned.dropna(subset=['target_conviction'] + FEATURE_COLS)

    if len(train) < 3:
        return None, None, None

    X = train[FEATURE_COLS].values
    y = train['target_conviction'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge 回归（数据少，正则化防过拟合）
    model = Ridge(alpha=0.5)
    model.fit(X_scaled, y)

    # 计算特征重要性
    importance = pd.Series(
        np.abs(model.coef_),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    return model, scaler, importance


def predict_conviction(model, scaler, latest_features: dict) -> float:
    """用最新市场特征预测下季度 NT 信心指数"""
    X = np.array([[latest_features.get(c, 0) for c in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


# ============================================================
# 4. 仓位计算
# ============================================================

def conviction_to_position(conviction: float, trend: float = 0, acceleration: float = 0) -> float:
    """三因子 → 仓位（同 position_engine 的公式）"""
    conv_norm = np.clip(conviction / 0.4, -1, 1)
    trend_norm = np.clip(trend / 0.3, -1, 1)
    accel_norm = np.clip(acceleration / 0.3, -1, 1)
    raw = 0.50 * conv_norm + 0.30 * trend_norm + 0.20 * accel_norm
    pos = 1.0 / (1.0 + np.exp(-3.0 * raw))
    return np.clip(pos, 0.02, 0.98)


# ============================================================
# 5. 回测
# ============================================================

def backtest_predictive(features, conviction_df, market):
    """回测预测性仓位策略 vs 实际仓位策略 vs 满仓"""
    aligned = align_quarterly(features, conviction_df)
    model, scaler, importance = train_model(aligned)

    if model is None:
        return {'error': '训练数据不足'}

    results = []
    for i in range(2, len(aligned) - 1):
        # 用当前季度特征预测下季度
        train_data = aligned.iloc[:i + 1].dropna(subset=['target_conviction'] + FEATURE_COLS)
        if len(train_data) < 3:
            continue

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data['target_conviction'].values
        scaler_i = StandardScaler().fit(X_train)
        model_i = Ridge(alpha=0.5).fit(scaler_i.transform(X_train), y_train)

        # 预测
        current_features = aligned.iloc[i][FEATURE_COLS].to_dict()
        X_pred = np.array([[current_features.get(c, 0) for c in FEATURE_COLS]])
        predicted_conv = float(model_i.predict(scaler_i.transform(X_pred))[0])

        # 实际
        actual_conv = aligned.iloc[i]['conviction']

        # 下一季度的实际收益
        next_q = aligned.index[i + 1] if i + 1 < len(aligned) else None

        results.append({
            'quarter': aligned.index[i],
            'actual_conviction': actual_conv,
            'predicted_conviction': round(predicted_conv, 4),
            'error': round(predicted_conv - actual_conv, 4),
        })

    # 模拟交易
    hs300_q = market['close'].resample('QE').last().pct_change()
    nav_pred = nav_actual = nav_full = 1.0

    for r in results:
        q = r['quarter']
        date_str = q[:4] + {'1': '0331', '2': '0630', '3': '0930', '4': '1231'}[q[-1]]
        ts = pd.Timestamp(date_str)

        # 下一季度才生效
        next_ts = ts + pd.DateOffset(months=3)
        if next_ts in hs300_q.index:
            q_ret = hs300_q[next_ts]

            pred_pos = conviction_to_position(r['predicted_conviction'])
            actual_pos = conviction_to_position(r['actual_conviction'])

            nav_pred *= (1 + q_ret * pred_pos)
            nav_actual *= (1 + q_ret * actual_pos)
            nav_full *= (1 + q_ret)

    years = len(results) / 4
    return {
        'results': results,
        'importance': importance.to_dict(),
        'pred_cagr': nav_pred ** (1 / max(years, 0.1)) - 1,
        'actual_cagr': nav_actual ** (1 / max(years, 0.1)) - 1,
        'full_cagr': nav_full ** (1 / max(years, 0.1)) - 1,
    }


# ============================================================
# 6. 报告
# ============================================================

def print_report(result: dict, conviction_df: pd.DataFrame, features: pd.DataFrame,
                 importance: dict = None):
    """打印预测仓位报告"""
    # 训练模型
    aligned = align_quarterly(features, conviction_df)
    model, scaler, imp = train_model(aligned)

    if model is None:
        print('[ERROR] 数据不足')
        return

    # 最新特征 → 预测
    latest_features = features.iloc[-1][FEATURE_COLS].to_dict()
    predicted_conv = predict_conviction(model, scaler, latest_features)
    latest_actual = conviction_df.iloc[-1]['conviction']

    pred_pos = conviction_to_position(predicted_conv)
    actual_pos = conviction_to_position(latest_actual)

    print()
    print('=' * 60)
    print('  预测性仓位引擎 — 基于沪指K线拟合国家队行为')
    print('=' * 60)
    print(f'  训练数据: {len(aligned.dropna(subset=["target_conviction"]))} 个季度')
    print()
    print(f'  {"指标":<20} {"实际(滞后)":>12} {"预测(领先)":>12}')
    print(f'  {"-"*44}')
    print(f'  {"NT 信心指数":<20} {latest_actual:>+11.4f}  {predicted_conv:>+11.4f}')
    print(f'  {"建议仓位":<20} {actual_pos:>11.0%}  {pred_pos:>11.0%}')
    print(f'  {"仓位差异":<20} {"":>12}  {(pred_pos-actual_pos):>+10.0%}')

    if imp is not None:
        print()
        print(f'  Top 5 预测因子:')
        for feat, score in list(imp.items())[:5]:
            print(f'    {feat:<15} {score:.4f}')

    # 保存预测仓位供 V4.2 消费
    top3 = {}
    if imp is not None:
        try:
            top3 = {str(k): round(float(v), 4) for k, v in list(imp.items())[:3]}
        except Exception:
            top3 = {}
    output = {
        'date': datetime.now().isoformat(),
        'predicted_conviction': round(predicted_conv, 4),
        'actual_conviction': latest_actual,
        'predicted_position': round(pred_pos, 3),
        'actual_position': round(actual_pos, 3),
        'signal': 'BUY' if pred_pos > 0.60 else ('SELL' if pred_pos < 0.40 else 'HOLD'),
        'top_factors': top3,
    }
    cache_file = PARENT / '.cache' / 'national_team' / 'predicted_position.json'
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n[Output] Predicted position saved: {cache_file}')

    print('=' * 60)

    # 回测
    if result and 'pred_cagr' in result:
        print()
        print(f'  {"策略":<20} {"年化收益":>12}')
        print(f'  {"-"*32}')
        print(f'  {"预测仓位":<20} {result["pred_cagr"]:>+11.2%}')
        print(f'  {"实际仓位":<20} {result["actual_cagr"]:>+11.2%}')
        print(f'  {"满仓持有":<20} {result["full_cagr"]:>+11.2%}')

        if result['results']:
            print()
            print(f'  {"季度":<8} {"实际信心":>10} {"预测信心":>10} {"误差":>8}')
            print(f'  {"-"*38}')
            for r in result['results']:
                print(f'  {r["quarter"]:<8} {r["actual_conviction"]:>+9.4f} '
                      f'{r["predicted_conviction"]:>+9.4f} {r["error"]:>+7.4f}')


def main():
    parser = argparse.ArgumentParser(description='预测性仓位引擎')
    parser.add_argument('--backtest', action='store_true', help='回测模式')
    parser.add_argument('--compare', action='store_true', help='对比预测 vs 实际')
    args = parser.parse_args()

    print('[数据] 加载市场数据...')
    market = load_market_data()
    print(f'  HS300: {len(market)} 天')

    print('[数据] 加载国家队信心指数...')
    conviction = load_nt_conviction()
    print(f'  NT: {len(conviction)} 个季度')

    print('[特征] 提取市场特征...')
    features = compute_market_features(market)
    print(f'  特征: {len(FEATURE_COLS)} 维, {len(features.dropna())} 有效天')

    # 回测
    bt_result = None
    if args.backtest or args.compare:
        print('[回测] 滚动预测回测...')
        bt_result = backtest_predictive(features, conviction, market)

    # 报告
    print_report(bt_result, conviction, features)


if __name__ == '__main__':
    main()
