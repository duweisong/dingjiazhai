"""
ETF 行情数据批量拉取脚本
========================

从 efinance 拉取 35 只候选 ETF 的日线数据，校验后缓存为 parquet。

用法:
  python fetch_etf_data.py              # 使用缓存，缺失的补拉
  python fetch_etf_data.py --force      # 强制全量重新拉取
  python fetch_etf_data.py --validate   # 仅校验已有缓存
"""
import sys
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import os, time, argparse
import pandas as pd
import efinance as ef
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / '.cache' / 'etf_sector'
CACHE_FILE = CACHE_DIR / 'nt_rotation_35.parquet'

START_DATE = '20190101'
END_DATE = '20260612'
MIN_DAYS = 100  # 最少需要这么多交易日

# ============================================================
# ETF 候选池 (35只)
# ============================================================
ETF_POOL = {
    # 宽基 (8)
    '510300': ('沪深300ETF', '宽基'),
    '510500': ('中证500ETF', '宽基'),
    '510050': ('上证50ETF', '宽基'),
    '159915': ('创业板ETF', '宽基'),
    '588000': ('科创50ETF', '宽基'),
    '159949': ('创业板50ETF', '宽基'),
    '512100': ('中证1000ETF', '宽基'),
    '510880': ('红利ETF', '宽基'),
    # 行业/主题 (27)
    '512880': ('证券ETF', '金融'),
    '512800': ('银行ETF', '金融'),
    '512660': ('军工ETF', '军工'),
    '512670': ('国防ETF', '军工'),
    '512690': ('酒ETF', '消费'),
    '159736': ('食品饮料ETF', '消费'),
    '159996': ('家电ETF', '消费'),
    '159995': ('芯片ETF', '科技'),
    '512480': ('半导体ETF', '科技'),
    '512760': ('半导体50ETF', '科技'),
    '159869': ('游戏ETF', '科技'),
    '512980': ('传媒ETF', '传媒'),
    '515050': ('5GETF', '科技'),
    '516510': ('云计算ETF', '科技'),
    '159865': ('人工智能ETF', '科技'),
    '512010': ('医药ETF', '医药'),
    '512170': ('医疗ETF', '医药'),
    '159755': ('新能源车ETF', '新能源'),
    '515790': ('光伏ETF', '新能源'),
    '561910': ('电池ETF', '新能源'),
    '159611': ('电力ETF', '公用事业'),
    '515220': ('煤炭ETF', '周期'),
    '516970': ('建材ETF', '周期'),
    '512200': ('房地产ETF', '基建'),
    '516950': ('基建ETF', '基建'),
    '159766': ('旅游ETF', '消费'),
    '561330': ('矿业ETF', '周期'),
}


def fetch_single(code: str, name: str) -> pd.Series | None:
    """拉取单个 ETF 的累计净值序列。失败返回 None。"""
    try:
        df = ef.fund.get_quote_history(code)
        if df is None or df.empty:
            return None

        df = df.rename(columns={'日期': 'date', '累计净值': 'cum_nav'})
        df['date'] = pd.to_datetime(df['date'])
        df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce').ffill()
        df = df.sort_values('date').set_index('date')
        df = df.loc[START_DATE:END_DATE]

        if len(df) < MIN_DAYS:
            return None

        return df['cum_nav'].rename(code)
    except Exception:
        return None


def fetch_all(force: bool = False) -> dict:
    """
    批量拉取所有 ETF 数据。

    Returns
    -------
    dict: {code: Series} 成功拉取的 ETF，以及 {'failed': [...], 'skipped': [...]}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if CACHE_FILE.exists() and not force:
        cached = pd.read_parquet(CACHE_FILE)
        existing = {c: cached[c] for c in cached.columns if c in ETF_POOL}
        print(f'[Cache] 已有 {len(existing)} 只 ETF 的缓存数据')

    results = {}
    failed = []
    skipped = []

    codes = list(ETF_POOL.keys())
    for i, code in enumerate(codes):
        name, sector = ETF_POOL[code]

        if code in existing and not force:
            results[code] = existing[code]
            skipped.append(code)
            continue

        print(f'[{i+1}/{len(codes)}] {code} {name}...', end=' ', flush=True)
        series = fetch_single(code, name)

        if series is not None:
            results[code] = series
            print(f'OK ({len(series)}天)')
        else:
            failed.append(code)
            print(f'SKIP (数据不足或拉取失败)')

        time.sleep(0.6)  # 限速，避免被封

    # 保存
    if results:
        data = pd.DataFrame(results).sort_index().ffill().bfill()
        data.to_parquet(CACHE_FILE)
        print(f'\n[Save] {len(data)} 天 × {len(results)} 只 → {CACHE_FILE}')

    return {'data': results, 'failed': failed, 'skipped': skipped}


def validate_cache() -> dict:
    """校验已有缓存的数据完整性。"""
    if not CACHE_FILE.exists():
        return {'exists': False, 'error': '缓存文件不存在'}

    data = pd.read_parquet(CACHE_FILE)
    codes = list(data.columns)

    report = {
        'exists': True,
        'file': str(CACHE_FILE),
        'days': len(data),
        'date_range': f'{data.index[0].date()} ~ {data.index[-1].date()}',
        'etfs_total': len(codes),
        'etfs_ok': 0,
        'etfs_short': [],
        'etfs_ok_list': [],
        'etfs_missing': [],
    }

    for code in ETF_POOL:
        if code in data.columns:
            n = data[code].dropna().count()
            if n >= MIN_DAYS:
                report['etfs_ok'] += 1
                report['etfs_ok_list'].append(code)
            else:
                report['etfs_short'].append((code, n))
        else:
            report['etfs_missing'].append(code)

    return report


def main():
    parser = argparse.ArgumentParser(description='ETF 行情数据批量拉取')
    parser.add_argument('--force', action='store_true', help='强制全量重新拉取')
    parser.add_argument('--validate', action='store_true', help='仅校验已有缓存')
    args = parser.parse_args()

    if args.validate:
        report = validate_cache()
        print('=' * 60)
        print('  ETF 数据缓存校验')
        print('=' * 60)
        if not report['exists']:
            print(f'  [FAIL] {report["error"]}')
            return 1

        print(f'  文件: {report["file"]}')
        print(f'  时间范围: {report["date_range"]}')
        print(f'  总交易日: {report["days"]}')
        print(f'  通过验证: {report["etfs_ok"]}/{len(ETF_POOL)} 只')
        print(f'  通过列表: {", ".join(report["etfs_ok_list"])}')

        if report['etfs_short']:
            print(f'\n  数据不足 ({len(report["etfs_short"])}只):')
            for code, n in report['etfs_short']:
                print(f'    {code} {ETF_POOL[code][0]}: {n}天')

        if report['etfs_missing']:
            print(f'\n  缺失 ({len(report["etfs_missing"])}只):')
            for code in report['etfs_missing']:
                print(f'    {code} {ETF_POOL[code][0]}')

        if report['etfs_ok'] >= 30:
            print(f'\n  [OK] 达标 (>=30只)')
            return 0
        else:
            print(f'\n  [FAIL] 不达标 (<30只)，请运行 python fetch_etf_data.py --force')
            return 1

    # 拉取模式
    print(f'[Fetch] 目标: {len(ETF_POOL)} 只 ETF')
    print(f'[Fetch] 区间: {START_DATE} ~ {END_DATE}')
    print(f'[Fetch] 最低要求: {MIN_DAYS} 个交易日')
    if args.force:
        print('[Fetch] 模式: 强制全量重新拉取')
    print()

    result = fetch_all(force=args.force)

    data = result['data']
    failed = result['failed']
    skipped = result['skipped']

    print(f'\n{"="*60}')
    print(f'  拉取完成: {len(data)}/{len(ETF_POOL)} 只成功')
    if skipped:
        print(f'  缓存命中: {len(skipped)} 只')
    if failed:
        print(f'  失败: {len(failed)} 只 → 记录到踩坑记录')
        for code in failed:
            print(f'    - {code} {ETF_POOL[code][0]}')
    print(f'{"="*60}')

    return 0 if len(data) >= 30 else 1


if __name__ == '__main__':
    sys.exit(main())
