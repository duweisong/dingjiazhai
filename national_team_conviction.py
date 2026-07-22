"""
国家队信心指数
==============

从国家队持仓数据提取的宏观信心信号。

核心逻辑:
  - 国家队重点实体: 社保基金, 养老金, 证金, 汇金
  - 季度总持仓市值环比变化 → 信心指数 (-1 ~ +1)
  - 加仓 > 5%: 信心 +, 减仓 > 5%: 信心 -

用法:
  python national_team_conviction.py            # 计算最新信心指数
  python national_team_conviction.py --update   # 拉取新数据并计算
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVICTION_FILE = os.path.join(OUTPUT_DIR, '.cache', 'national_team', 'conviction.json')
os.makedirs(os.path.dirname(CONVICTION_FILE), exist_ok=True)

# 国家队关键实体（从股东名称中匹配）
NATIONAL_TEAM_KEYWORDS = [
    '全国社保基金',
    '基本养老保险基金',
    '中国证券金融',
    '中央汇金',
    '证金公司',
    '养老金',
]


def load_holdings() -> pd.DataFrame:
    """加载缓存的持仓数据"""
    cache = os.path.join(OUTPUT_DIR, '.cache', 'national_team', 'holdings_latest.parquet')
    if not os.path.exists(cache):
        raise FileNotFoundError('国家队数据未拉取，请先运行 national_team_tracker.py')
    return pd.read_parquet(cache)


def compute_conviction(df: pd.DataFrame) -> dict:
    """
    计算国家队信心指数。

    Returns
    -------
    dict: {
        'date': str,
        'total_market_value': float,
        'nt_market_value': float,       # 国家队持仓总市值
        'nt_value_pct': float,           # 国家队占总市值比例
        'nt_stock_count': int,           # 国家队持股数
        'buy_count': int,                # 增持股票数
        'sell_count': int,               # 减持股票数
        'conviction_score': float,       # 信心指数 (-1 ~ +1)
        'signal': str,                   # bullish / neutral / bearish
    }
    """
    # 列名解码
    col_map = {}
    for c in df.columns:
        decoded = c.encode('raw_unicode_escape').decode('gbk', errors='replace')
        col_map[decoded] = c

    holder_col = col_map.get('股东类型', col_map.get('股东名称', df.columns[1]))
    name_col = col_map.get('股东名称', df.columns[1])
    qty_change_col = col_map.get('期末持股-数量变化', df.columns[7])
    mkt_val_col = col_map.get('期末持股-流通市值', df.columns[10])
    change_col = col_map.get('期末持股-持股变动', df.columns[9])
    date_col = col_map.get('报告期', df.columns[5])

    # 筛选国家队
    def is_national_team(name):
        if pd.isna(name): return False
        s = str(name)
        return any(kw in s for kw in NATIONAL_TEAM_KEYWORDS)

    nt_mask = df[holder_col].apply(is_national_team) | df[name_col].apply(is_national_team)
    nt_df = df[nt_mask].copy()

    if len(nt_df) == 0:
        return {'signal': 'no_data', 'conviction_score': 0}

    # 统计
    total_mv = pd.to_numeric(df[mkt_val_col], errors='coerce').sum()
    nt_mv = pd.to_numeric(nt_df[mkt_val_col], errors='coerce').sum()
    nt_count = len(nt_df)

    # 增持/减持统计
    if change_col in df.columns:
        buys = (nt_df[change_col].astype(str).str.contains('增')).sum()
        sells = (nt_df[change_col].astype(str).str.contains('减')).sum()
    else:
        # Use quantity change
        nt_df['qty_chg'] = pd.to_numeric(nt_df[qty_change_col], errors='coerce').fillna(0)
        buys = (nt_df['qty_chg'] > 0).sum()
        sells = (nt_df['qty_chg'] < 0).sum()

    # 报告日期
    dates = pd.to_datetime(df[date_col]).dropna()
    report_date = str(dates.iloc[0].date()) if len(dates) > 0 else 'unknown'

    # 信心指数: 基于买卖比
    total_actions = buys + sells
    if total_actions > 0:
        conviction = (buys - sells) / total_actions  # -1 to +1
    else:
        conviction = 0

    if conviction > 0.15:
        signal = 'bullish'
    elif conviction < -0.10:
        signal = 'bearish'
    else:
        signal = 'neutral'

    result = {
        'date': report_date,
        'total_market_value': float(total_mv),
        'nt_market_value': float(nt_mv),
        'nt_value_pct': float(nt_mv / total_mv * 100) if total_mv > 0 else 0,
        'nt_stock_count': nt_count,
        'buy_count': int(buys),
        'sell_count': int(sells),
        'conviction_score': round(conviction, 3),
        'signal': signal,
    }

    return result


def get_conviction() -> dict:
    """获取最新信心指数（优先读缓存）"""
    if os.path.exists(CONVICTION_FILE):
        with open(CONVICTION_FILE) as f:
            cached = json.load(f)
        age = datetime.now() - datetime.fromisoformat(cached.get('computed_at', '2000-01-01'))
        if age.days < 90:  # 季度内有效
            return cached

    # 重新计算
    df = load_holdings()
    result = compute_conviction(df)
    result['computed_at'] = datetime.now().isoformat()

    with open(CONVICTION_FILE, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def print_conviction(result: dict):
    """打印信心指数报告"""
    em = {'bullish': 'BULL', 'bearish': 'BEAR', 'neutral': '==', 'no_data': '??'}.get(result['signal'], '??')

    print('=' * 50)
    print(f'  国家队信心指数 {em}')
    print('=' * 50)
    print(f'  报告期: {result["date"]}')
    print(f'  国家队持仓市值: {result["nt_market_value"]/1e8:.1f} 亿')
    print(f'  占总市值: {result["nt_value_pct"]:.2f}%')
    print(f'  持股数: {result["nt_stock_count"]}')
    print(f'  增持: {result["buy_count"]}  |  减持: {result["sell_count"]}')
    print(f'  信心指数: {result["conviction_score"]:+.3f}  ({result["signal"]})')
    print()

    if result['signal'] == 'bullish':
        print('  [BULL] 国家队整体加仓，对后市偏乐观 -> ETF +0.10 boost')
    elif result['signal'] == 'bearish':
        print('  [BEAR] 国家队整体减仓，对后市偏谨慎 -> ETF -0.05 discount')
    else:
        print('  [==] 国家队仓位平稳，维持中性')

    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description='国家队信心指数')
    parser.add_argument('--update', action='store_true', help='强制重新计算')
    args = parser.parse_args()

    if args.update and os.path.exists(CONVICTION_FILE):
        os.remove(CONVICTION_FILE)

    result = get_conviction()
    print_conviction(result)


if __name__ == '__main__':
    main()
