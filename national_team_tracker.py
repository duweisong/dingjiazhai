"""
国家队持仓追踪工具
==================

每周拉取一次国家队最新持仓，聚合到行业层面，追踪变化方向。

数据源: akshare stock_gdfx_holding_analyse_em (季度披露)
用法:
  python national_team_tracker.py           # 拉取最新数据并缓存
  python national_team_tracker.py --report  # 查看最新报告
  python national_team_tracker.py --compare # 对比上次变化

设计: 首次拉取 ~5 分钟 (全量247页), 后续缓存秒开。
      数据按季度更新，建议每季度初运行一次即可。
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, '.cache', 'national_team')
CACHE_FILE = os.path.join(CACHE_DIR, 'holdings_latest.parquet')
META_FILE = os.path.join(CACHE_DIR, 'meta.json')
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# 1. 数据拉取
# ============================================================

def fetch_holdings(date: str = None, max_retries: int = 2):
    """
    拉取国家队持股分析数据。

    Parameters
    ----------
    date : str
        报告期, 如 '20241231' (2024Q4). 不传则用最新.
    max_retries : int
        最大重试次数

    Returns
    -------
    pd.DataFrame
    """
    import akshare as ak
    import time

    for attempt in range(max_retries):
        try:
            print(f'[拉取] 国家队持股数据 (date={date or "latest"})...')
            print('[拉取] 数据量较大 (~247页), 预计 3-5 分钟, 请耐心等待...')

            if date:
                df = ak.stock_gdfx_holding_analyse_em(date=date)
            else:
                # 尝试最近几个季度（从最新开始）
                for q in ['20260331', '20251231', '20250331', '20241231']:
                    try:
                        df = ak.stock_gdfx_holding_analyse_em(date=q)
                        date = q
                        break
                    except:
                        continue

            print(f'[拉取] 成功! {len(df)} 条记录, {df["股东类型"].nunique()} 类股东')
            return df, date

        except Exception as e:
            print(f'[拉取] 尝试 {attempt+1}/{max_retries} 失败: {type(e).__name__}: {str(e)[:120]}')
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f'[拉取] {wait}s 后重试...')
                time.sleep(wait)

    raise RuntimeError('国家队数据拉取失败，请检查网络后重试')


# ============================================================
# 2. 行业聚合
# ============================================================

# 申万一级行业映射（从 akshare 返回的行业名到我们 ETF 策略的行业）
SW_SECTOR_MAP = {
    '银行': '金融', '非银金融': '金融', '证券': '金融', '保险': '金融',
    '电子': 'TMT', '计算机': 'TMT', '通信': 'TMT', '传媒': 'TMT',
    '有色金属': '周期', '煤炭': '周期', '钢铁': '周期', '石油石化': '周期',
    '基础化工': '周期', '建筑材料': '基建', '建筑装饰': '基建',
    '房地产': '基建', '公用事业': '公用事业', '环保': '公用事业',
    '电力设备': '新能源', '新能源': '新能源',
    '机械设备': '制造', '国防军工': '国防军工', '汽车': '制造',
    '食品饮料': '消费', '家用电器': '消费', '商贸零售': '消费',
    '社会服务': '消费', '美容护理': '消费', '纺织服饰': '消费',
    '医药生物': '医药', '农林牧渔': '农业',
}

HOLDER_TYPE_MAP = {
    '中央汇金': '中央汇金',
    '证金公司': '证金',
    '社保基金': '社保',
    '养老金': '养老金',
    '企业年金': '企业年金',
}


def aggregate_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    将持股明细聚合到行业层面。

    Returns
    -------
    pd.DataFrame with columns:
        sector, holder_type, stock_count, total_market_value, avg_holding_pct
    """
    # 映射行业
    if '行业' in df.columns:
        df['sector_group'] = df['行业'].map(SW_SECTOR_MAP).fillna('其他')
    elif '所属行业' in df.columns:
        df['sector_group'] = df['所属行业'].map(SW_SECTOR_MAP).fillna('其他')
    else:
        # 尝试从行业列推断
        sector_cols = [c for c in df.columns if '行业' in str(c)]
        if sector_cols:
            df['sector_group'] = df[sector_cols[0]].map(SW_SECTOR_MAP).fillna('其他')
        else:
            df['sector_group'] = '未知'

    # 映射股东类型
    if '股东类型' in df.columns:
        df['holder_group'] = df['股东类型'].apply(
            lambda x: next((v for k, v in HOLDER_TYPE_MAP.items() if k in str(x)), '其他')
        )
    else:
        df['holder_group'] = '未知'

    # 数值列检测
    value_col = None
    for c in ['持股市值', '持仓市值', '参考市值', '市值']:
        if c in df.columns:
            value_col = c
            break
    # 尝试发现数值列
    if value_col is None:
        for c in df.columns:
            if df[c].dtype in ('float64', 'int64') and df[c].abs().max() > 1000:
                value_col = c
                break

    agg = df.groupby(['sector_group', 'holder_group']).agg(
        stock_count=('股票代码' if '股票代码' in df.columns else df.columns[0], 'count'),
        **{f'total_value': (value_col, 'sum')} if value_col else {},
    ).reset_index()

    if value_col:
        total = agg['total_value'].sum()
        agg['value_pct'] = agg['total_value'] / total * 100 if total > 0 else 0

    return agg


# ============================================================
# 3. 变化检测
# ============================================================

def compare_holdings(new_df: pd.DataFrame, old_df: pd.DataFrame) -> pd.DataFrame:
    """对比两期持仓变化"""
    new_agg = aggregate_by_sector(new_df)
    old_agg = aggregate_by_sector(old_df)

    if 'total_value' in new_agg.columns and 'total_value' in old_agg.columns:
        merged = new_agg.merge(
            old_agg, on=['sector_group', 'holder_group'],
            how='outer', suffixes=('_new', '_old')
        ).fillna(0)

        merged['value_change'] = merged['total_value_new'] - merged['total_value_old']
        merged['value_change_pct'] = (
            (merged['total_value_new'] - merged['total_value_old'])
            / merged['total_value_old'].replace(0, 1) * 100
        )
        merged['stock_change'] = merged['stock_count_new'] - merged['stock_count_old']

        merged = merged.sort_values('value_change', ascending=False)
        return merged

    return new_agg


# ============================================================
# 4. 报告生成
# ============================================================

def generate_report(df: pd.DataFrame, date: str) -> str:
    """生成可读报告"""
    agg = aggregate_by_sector(df)

    lines = []
    lines.append('=' * 60)
    lines.append(f'  国家队持仓分析报告')
    lines.append(f'  报告期: {date[:4]}Q{int(date[4:6])//3} ({date})')
    lines.append(f'  生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('=' * 60)

    # 按行业汇总
    if 'value_pct' in agg.columns:
        lines.append('\n[ 行业配置分布:')
        lines.append(f'  {"行业":<10} {"持股市值占比":>12} {"持股数":>8}')
        lines.append('  ' + '-' * 32)
        sector_sum = agg.groupby('sector_group').agg(
            value_sum=('total_value', 'sum'),
            stock_sum=('stock_count', 'sum')
        ).sort_values('value_sum', ascending=False)
        total_val = sector_sum['value_sum'].sum()
        for sector, row in sector_sum.iterrows():
            pct = row['value_sum'] / total_val * 100 if total_val > 0 else 0
            bar = '█' * max(1, int(pct))
            lines.append(f'  {sector:<10} {pct:>10.1f}% {row["stock_sum"]:>8}  {bar}')

    # 按股东类型汇总
    if '持有市值' in df.columns or 'total_value' in agg.columns:
        lines.append('\n[ 股东类型分布:')
        holder_sum = agg.groupby('holder_group')['total_value'].sum().sort_values(ascending=False)
        for holder, val in holder_sum.items():
            lines.append(f'  {holder:<12} {val/1e8:>10.1f} 亿')

    # 增持/减持最多的行业（如有对比数据）
    if 'value_change' in agg.columns:
        lines.append('\n🔄 环比变化 (Top 5):')
        top = agg.nlargest(5, 'value_change')
        for _, row in top.iterrows():
            lines.append(f'  📈 {row["sector_group"]:<10} {row["holder_group"]:<8} '
                        f'{row["value_change"]/1e8:>+8.1f}亿')

        bottom = agg.nsmallest(5, 'value_change')
        for _, row in bottom.iterrows():
            lines.append(f'  📉 {row["sector_group"]:<10} {row["holder_group"]:<8} '
                        f'{row["value_change"]/1e8:>+8.1f}亿')

    lines.append('\n' + '=' * 60)
    lines.append('⚠️ 数据基于季报披露，滞后 1-2 个月。仅供参考，不构成投资建议。')
    lines.append('=' * 60)

    return '\n'.join(lines)


# ============================================================
# 5. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='国家队持仓追踪')
    parser.add_argument('--report', action='store_true', help='查看最新报告')
    parser.add_argument('--compare', action='store_true', help='与上次数据对比')
    parser.add_argument('--date', type=str, default=None, help='指定报告期 (如 20241231)')
    parser.add_argument('--export', type=str, default=None, help='导出 CSV 路径')
    args = parser.parse_args()

    # 加载缓存
    has_cache = os.path.exists(CACHE_FILE)

    if args.report and has_cache:
        df = pd.read_parquet(CACHE_FILE)
        with open(META_FILE) as f:
            meta = json.load(f)
        report = generate_report(df, meta['date'])
        print(report)
        return

    if args.compare and has_cache:
        # 对比模式：拉取新数据，与缓存对比
        print('[对比] 拉取最新数据...')
        new_df, new_date = fetch_holdings(args.date)
        old_df = pd.read_parquet(CACHE_FILE)
        with open(META_FILE) as f:
            old_meta = json.load(f)

        # 如果日期相同，无需对比
        if new_date == old_meta['date']:
            print(f'[对比] 数据已是同一报告期 ({new_date})，无需对比')
            report = generate_report(new_df, new_date)
            print(report)
            return

        # 生成对比
        print(f'[对比] {old_meta["date"]} → {new_date}')
        # 保存新数据
        new_df.to_parquet(CACHE_FILE)
        with open(META_FILE, 'w') as f:
            json.dump({'date': new_date, 'fetched_at': datetime.now().isoformat()}, f)

        report = generate_report(new_df, new_date)
        print(report)
        return

    # 首次拉取 / 刷新
    if not has_cache or not (args.report or args.compare):
        print('[首次] 拉取国家队持仓数据...')
        df, date = fetch_holdings(args.date)

        # 缓存
        df.to_parquet(CACHE_FILE)
        with open(META_FILE, 'w') as f:
            json.dump({'date': date, 'fetched_at': datetime.now().isoformat()}, f)
        print(f'[缓存] 已保存至 {CACHE_FILE}')

        # 导出
        if args.export:
            df.to_csv(args.export, index=False, encoding='utf-8-sig')
            print(f'[导出] CSV 已保存至 {args.export}')

        # 生成报告
        report = generate_report(df, date)
        print(report)


if __name__ == '__main__':
    main()
