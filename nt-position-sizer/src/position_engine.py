"""
国家队仓位引擎 — 从 9 季度持仓数据 → 0-100% 连续仓位建议

用法:
  python position_engine.py                # 当前仓位建议
  python position_engine.py --backtest      # 回测仓位策略
  python position_engine.py --live          # 更新数据 + 输出仓位
"""

import sys, os, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 复用父项目数据
PARENT = Path(__file__).parent.parent.parent
CACHE_DIR = PARENT / '.cache' / 'national_team'

NT_KEYWORDS = ['全国社保基金', '基本养老保险基金', '中国证券金融', '中央汇金', '证金公司', '养老金']


def load_conviction_history() -> pd.DataFrame:
    """从缓存加载所有季度的信心指数"""
    rows = []
    files = list(CACHE_DIR.glob('holdings_*.parquet'))
    for f in sorted(files):
        stem = f.stem
        if stem == 'holdings_latest':
            meta_file = CACHE_DIR / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as mf:
                    date_str = json.load(mf)['date']
            else:
                continue
        else:
            date_str = stem.replace('holdings_', '')
        df = pd.read_parquet(f)

        col_map = {}
        for c in df.columns:
            decoded = c.encode('raw_unicode_escape').decode('gbk', errors='replace')
            col_map[decoded] = c

        holder_col = col_map.get('股东类型', df.columns[1])
        name_col = col_map.get('股东名称', df.columns[1])
        change_col = col_map.get('期末持股-持股变动', df.columns[9])
        mkt_val_col = col_map.get('期末持股-流通市值', df.columns[10])

        def is_nt(name):
            if pd.isna(name): return False
            return any(kw in str(name) for kw in NT_KEYWORDS)

        nt = df[df[holder_col].apply(is_nt) | df[name_col].apply(is_nt)]
        total_mv = pd.to_numeric(df[mkt_val_col], errors='coerce').sum()
        nt_mv = pd.to_numeric(nt[mkt_val_col], errors='coerce').sum()
        buys = (nt[change_col].astype(str).str.contains('增')).sum()
        sells = (nt[change_col].astype(str).str.contains('减')).sum()
        conv = (buys - sells) / (buys + sells) if (buys + sells) > 0 else 0

        q_label = date_str[:4] + 'Q' + str((int(date_str[4:6]) - 1) // 3 + 1)
        rows.append({
            'quarter': q_label,
            'date': date_str,
            'nt_stocks': len(nt),
            'nt_mv_b': round(nt_mv / 1e8, 1),
            'buys': int(buys),
            'sells': int(sells),
            'conviction': round(conv, 4),
        })

    df = pd.DataFrame(rows).sort_values('date')
    return df


def calculate_position(df: pd.DataFrame) -> dict:
    """
    计算当前建议仓位。

    Formula:
      raw = 0.50*conv + 0.30*trend + 0.20*accel
      position = sigmoid(3 * raw)
    """
    if len(df) < 3:
        return {'position': 0.50, 'confidence': 'low', 'reason': '数据不足'}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    conv = latest['conviction']
    trend = conv - prev['conviction']
    acceleration = trend - (prev['conviction'] - prev2['conviction'])

    # Normalize to -1..+1
    conv_norm = np.clip(conv / 0.4, -1, 1)       # /0.4 因为历史极值约 ±0.4
    trend_norm = np.clip(trend / 0.3, -1, 1)
    accel_norm = np.clip(acceleration / 0.3, -1, 1)

    raw = 0.50 * conv_norm + 0.30 * trend_norm + 0.20 * accel_norm

    # Sigmoid
    position = 1.0 / (1.0 + np.exp(-3.0 * raw))

    # Smoothing: max ±30% change from last quarter
    if len(df) >= 2:
        prev_position = 1.0 / (1.0 + np.exp(-3.0 * (
            0.50 * np.clip(prev['conviction'] / 0.4, -1, 1)
            + 0.30 * np.clip((prev['conviction'] - prev2['conviction']) / 0.3, -1, 1)
            + 0.20 * 0  # prev acceleration unknown
        )))
        position = np.clip(position, prev_position - 0.30, prev_position + 0.30)

    position = np.clip(position, 0.02, 0.98)

    # 趋势判定
    if trend > 0.05:
        trend_label = '国家队在加仓'
    elif trend < -0.05:
        trend_label = '国家队在减仓'
    else:
        trend_label = '国家队仓位平稳'

    return {
        'position': round(position, 3),
        'position_pct': f'{position:.0%}',
        'conviction': conv,
        'trend': round(trend, 4),
        'acceleration': round(acceleration, 4),
        'raw_score': round(raw, 4),
        'trend_label': trend_label,
        'quarters_used': len(df),
        'latest_quarter': latest['quarter'],
    }


def run_backtest(df: pd.DataFrame) -> dict:
    """回测仓位策略 vs 满仓 vs 空仓"""
    positions = []
    for i in range(2, len(df)):
        subset = df.iloc[:i + 1]
        result = calculate_position(subset)
        positions.append({
            'quarter': subset.iloc[-1]['quarter'],
            'conviction': result['conviction'],
            'position': result['position'],
        })

    pos_df = pd.DataFrame(positions)

    # 模拟：按建议仓位持有沪深300 vs 满仓 vs 空仓
    # 用实际季度收益（从 ETF 数据取）
    etf_cache = PARENT / '.cache' / 'etf_sector'
    hs300_path = etf_cache / 'hs300_2022.parquet'
    if hs300_path.exists():
        hs300 = pd.read_parquet(hs300_path)
        hs300_q = hs300['close'].resample('QE').last().pct_change()

        nav_positioned = 1.0
        nav_full = 1.0
        for _, row in pos_df.iterrows():
            q = row['quarter']
            yr = q[:4]
            q_num = q[-1]
            date_str = yr + {'1': '0331', '2': '0630', '3': '0930', '4': '1231'}[q_num]
            ts = pd.Timestamp(date_str)
            if ts in hs300_q.index:
                q_ret = hs300_q[ts]
                pos = row['position']
                nav_positioned *= (1 + q_ret * pos)       # 按仓位比例参与
                nav_full *= (1 + q_ret)                    # 满仓

        years = len(pos_df) / 4
        cagr_pos = nav_positioned ** (1 / max(years, 0.1)) - 1
        cagr_full = nav_full ** (1 / max(years, 0.1)) - 1

        return {
            'positioned_return': f'{nav_positioned - 1:+.2%}',
            'full_return': f'{nav_full - 1:+.2%}',
            'positioned_cagr': f'{cagr_pos:+.2%}',
            'full_cagr': f'{cagr_full:+.2%}',
            'outperformance': f'{cagr_pos - cagr_full:+.2%}',
            'quarters': len(pos_df),
        }

    return {'error': '沪深300 数据不可用'}


def print_report(result: dict):
    """打印仓位报告"""
    print()
    print('=' * 50)
    print('  国家队仓位判定器')
    print('=' * 50)
    print(f'  最新季度: {result["latest_quarter"]}')
    print(f'  使用数据: {result["quarters_used"]} 个季度')
    print()
    print(f'  当前信心指数: {result["conviction"]:+.4f}')
    print(f'  趋势方向:     {result["trend_label"]} ({result["trend"]:+.4f})')
    print(f'  加速度:       {result["acceleration"]:+.4f}')
    print(f'  原始得分:     {result["raw_score"]:+.4f}')
    print()
    print(f'  ★ 建议仓位: {result["position_pct"]} ({result["position"]:.3f})')
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description='国家队仓位判定器')
    parser.add_argument('--backtest', action='store_true', help='回测仓位策略')
    parser.add_argument('--live', action='store_true', help='更新数据并输出仓位')
    args = parser.parse_args()

    # 加载数据
    if not CACHE_DIR.exists():
        print('[ERROR] 国家队数据未拉取。请先运行 national_team_tracker.py')
        return

    df = load_conviction_history()

    if df.empty:
        print('[ERROR] 未找到季度数据')
        return

    # 当前仓位
    result = calculate_position(df)
    print_report(result)

    # 回测
    if args.backtest:
        bt = run_backtest(df)
        if 'error' not in bt:
            print()
            print(f'  回测 ({bt["quarters"]} 个季度):')
            print(f'    仓位策略: {bt["positioned_return"]} (年化 {bt["positioned_cagr"]})')
            print(f'    满仓持有: {bt["full_return"]} (年化 {bt["full_cagr"]})')
            print(f'    超额收益: {bt["outperformance"]}')
        else:
            print(f'  [回测] {bt["error"]}')

    return result


if __name__ == '__main__':
    main()
