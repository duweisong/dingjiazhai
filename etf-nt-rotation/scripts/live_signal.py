"""
ETF 国家队轮动策略 — 月度实盘信号生成 + PushPlus 推送
========================================================

每月最后一个交易日运行，输出下月持仓建议。

用法:
  python live_signal.py              # 生成信号（控制台输出）
  python live_signal.py --push       # 生成信号 + PushPlus 推送
  python live_signal.py --dry-run    # 仅预览，不推送
"""
import sys
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import os, json, argparse
import numpy as np
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / '.cache'
DATA_FILE = CACHE_DIR / 'etf_sector' / 'nt_rotation_35.parquet'
CONFIG_FILE = PROJECT_ROOT / 'pushplus_config.json'
PUSHPLUS_URL = 'http://www.pushplus.plus/send'

# ============================================================
# 优化后参数（来自 T05 网格搜索）
# ============================================================
PARAMS = {
    'momentum_weight': 0.50,
    'volatility_weight': 0.50,
    'mom_bars': 14,
    'vol_short': 5,
    'vol_long': 44,
    'rank_min': 1,
    'rank_max': 10,
    'top_n': 5,
}

ETF_INFO = {
    '510300': ('沪深300ETF', '宽基'), '510500': ('中证500ETF', '宽基'),
    '510050': ('上证50ETF', '宽基'), '159915': ('创业板ETF', '宽基'),
    '588000': ('科创50ETF', '宽基'), '159949': ('创业板50ETF', '宽基'),
    '512100': ('中证1000ETF', '宽基'), '510880': ('红利ETF', '宽基'),
    '512880': ('证券ETF', '金融'), '512800': ('银行ETF', '金融'),
    '512660': ('军工ETF', '军工'), '512670': ('国防ETF', '军工'),
    '512690': ('酒ETF', '消费'), '159736': ('食品饮料ETF', '消费'),
    '159996': ('家电ETF', '消费'), '159995': ('芯片ETF', '科技'),
    '512480': ('半导体ETF', '科技'), '512760': ('半导体50ETF', '科技'),
    '159869': ('游戏ETF', '科技'), '512980': ('传媒ETF', '传媒'),
    '515050': ('5GETF', '科技'), '516510': ('云计算ETF', '科技'),
    '159865': ('人工智能ETF', '科技'), '512010': ('医药ETF', '医药'),
    '512170': ('医疗ETF', '医药'), '159755': ('新能源车ETF', '新能源'),
    '515790': ('光伏ETF', '新能源'), '561910': ('电池ETF', '新能源'),
    '159611': ('电力ETF', '公用事业'), '515220': ('煤炭ETF', '周期'),
    '516970': ('建材ETF', '周期'), '512200': ('房地产ETF', '基建'),
    '516950': ('基建ETF', '基建'), '159766': ('旅游ETF', '消费'),
    '561330': ('矿业ETF', '周期'),
}


def load_data():
    """加载最新 ETF 数据。"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'数据文件不存在: {DATA_FILE}\n请先运行: python scripts/fetch_etf_data.py')

    data = pd.read_parquet(DATA_FILE).ffill().bfill()
    active = [c for c in data.columns if c in ETF_INFO and data[c].dropna().count() >= 100]
    return data[active], active


def calc_factors(data: pd.DataFrame):
    """计算因子得分（与回测引擎相同逻辑）。"""
    p = PARAMS
    momentum = pd.DataFrame(index=data.index, dtype=float)
    vol_ratio = pd.DataFrame(index=data.index, dtype=float)

    for c in data.columns:
        momentum[c] = data[c] / data[c].shift(p['mom_bars']) - 1.0
        abs_ret = data[c].pct_change().abs()
        vol_ratio[c] = (abs_ret.rolling(p['vol_short']).mean() /
                        abs_ret.rolling(p['vol_long']).mean().replace(0, np.nan))

    scores = pd.DataFrame(index=data.index, dtype=float)
    for date in momentum.index:
        mom_row = momentum.loc[date].dropna()
        vol_row = vol_ratio.loc[date].dropna()
        common = mom_row.index.intersection(vol_row.index)
        if len(common) < p['top_n']:
            continue
        mom_rank = mom_row[common].rank(pct=True)
        vol_rank = vol_row[common].rank(pct=True)
        scores.loc[date, common] = (p['momentum_weight'] * mom_rank +
                                    p['volatility_weight'] * vol_rank)
    return scores.dropna(how='all')


def select_etfs(scores_row: pd.Series) -> list:
    """从最新得分中选出持仓 ETF。"""
    p = PARAMS
    ranked = scores_row.dropna().sort_values(ascending=False)
    n = len(ranked)
    if n < p['top_n']:
        return []
    candidates = ranked.iloc[:min(p['rank_max'], n)]
    return list(candidates.index[:p['top_n']])


def generate_signal() -> dict:
    """生成当前月份的实盘信号。"""
    data, active = load_data()
    scores = calc_factors(data)
    latest_date = scores.index[-1]
    latest_scores = scores.loc[latest_date]

    # 选股
    selected_codes = select_etfs(latest_scores)
    weight = 1.0 / len(selected_codes) if selected_codes else 0

    # 完整排名
    ranked = latest_scores.sort_values(ascending=False)
    ranking = []
    for rank, (code, score) in enumerate(ranked.items(), 1):
        if code in ETF_INFO:
            name, sector = ETF_INFO[code]
            ranking.append({
                'rank': rank, 'code': code, 'name': name,
                'sector': sector, 'score': float(score),
                'selected': code in selected_codes,
                'weight': f'{weight:.1%}' if code in selected_codes else '-',
            })

    # 持仓汇总
    holdings = []
    for code in selected_codes:
        name, sector = ETF_INFO[code]
        score = float(latest_scores.get(code, 0))
        holdings.append({
            'code': code, 'name': name, 'sector': sector,
            'score': score, 'weight': f'{weight:.1%}',
        })

    return {
        'date': str(latest_date.date()),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'selected': holdings,
        'ranking': ranking[:15],
        'params': PARAMS.copy(),
        'data_info': {'days': len(data), 'etfs': len(active),
                      'range': f'{data.index[0].date()}~{data.index[-1].date()}'},
    }


def push_to_wechat(signal: dict, token: str):
    """通过 PushPlus 推送到微信。"""
    html = build_html(signal)
    title = f"ETF轮动·NT · {signal['date']} · {len(signal['selected'])}只持仓"

    try:
        resp = requests.post(PUSHPLUS_URL, json={
            'token': token, 'title': title,
            'content': html, 'template': 'html',
        }, timeout=10, verify=False)
        ok = resp.json().get('code') == 200
        print(f'[PushPlus] {"OK" if ok else "FAIL"} {title}')
        return ok
    except Exception as e:
        print(f'[PushPlus] ERROR: {e}')
        return False


def build_html(signal: dict) -> str:
    """生成 PushPlus HTML 推送内容。"""
    h = '<div style="font-family:Microsoft YaHei,sans-serif;max-width:620px">'

    # Header
    h += f'<h2 style="color:#FF8124;border-bottom:2px solid #FF8124;padding-bottom:10px">'
    h += f'ETF 国家队轮动 · 月度信号</h2>'
    h += f'<p style="color:#666">信号日: {signal["date"]} | 生成: {signal["generated_at"]}<br>'
    h += f'参数: 权重={signal["params"]["momentum_weight"]}/{signal["params"]["volatility_weight"]} '
    h += f'| 动量{signal["params"]["mom_bars"]}d '
    h += f'| 前{signal["params"]["rank_max"]}选{signal["params"]["top_n"]}只</p>'

    # Holdings table
    h += f'<h3>本月持仓 ({len(signal["selected"])}只，等权重)</h3>'
    h += '<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:13px">'
    h += '<tr style="background:#f5f5f5"><th>#</th><th>ETF</th><th>代码</th><th>板块</th><th>得分</th><th>仓位</th></tr>'
    for i, etf in enumerate(signal['selected']):
        h += f'<tr><td>#{i+1}</td><td><b>{etf["name"]}</b></td>'
        h += f'<td>{etf["code"]}</td><td>{etf["sector"]}</td>'
        h += f'<td>{etf["score"]:.4f}</td><td><b>{etf["weight"]}</b></td></tr>'
    h += '</table>'

    # Ranking table
    h += f'<h3 style="margin-top:18px">Top 15 排名</h3>'
    h += '<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:12px">'
    h += '<tr style="background:#f5f5f5"><th>排名</th><th>ETF</th><th>代码</th><th>板块</th><th>得分</th></tr>'
    for r in signal['ranking']:
        mark = '★' if r['selected'] else ''
        color = '#FF8124' if r['selected'] else '#999'
        h += f'<tr style="color:{color}"><td>{mark}#{r["rank"]}</td>'
        h += f'<td>{r["name"]}</td><td>{r["code"]}</td>'
        h += f'<td>{r["sector"]}</td><td>{r["score"]:.4f}</td></tr>'
    h += '</table>'

    # Footer
    h += '<div style="margin-top:18px;padding:10px;background:#fff3e0;border-left:4px solid #FF8124;font-size:12px">'
    h += '<b>策略:</b> 50/50权重 | 14d动量 | Top10选8只 | 月度轮动 | 等权重<br>'
    h += f'<b>数据:</b> {signal["data_info"]["etfs"]}只ETF | {signal["data_info"]["range"]}'
    h += '</div>'
    h += '<p style="color:#999;font-size:11px">ETF国家队轮动策略 · 仅供学习参考，不构成投资建议</p>'
    h += '</div>'
    return h


def print_signal(signal: dict):
    """终端彩色输出信号。"""
    print()
    print('=' * 60)
    print('  ETF 国家队轮动策略 · 月度实盘信号')
    print('=' * 60)
    print(f'  信号日: {signal["date"]}')
    print(f'  参数: 权重={signal["params"]["momentum_weight"]}/{signal["params"]["volatility_weight"]}'
          f' | 动量{signal["params"]["mom_bars"]}d'
          f' | 前{signal["params"]["rank_max"]}选{signal["params"]["top_n"]}只')
    print(f'  数据: {signal["data_info"]["etfs"]}只ETF | {signal["data_info"]["range"]}')
    print()
    print(f'  *** 本月持仓 ({len(signal["selected"])}只，等权重) ***')
    print(f'  {"#":<4} {"代码":<8} {"名称":<14} {"板块":<6} {"得分":>8} {"仓位":>6}')
    print(f'  {"-"*50}')
    for i, etf in enumerate(signal['selected']):
        print(f'  {i+1:<4} {etf["code"]:<8} {etf["name"]:<14} '
              f'{etf["sector"]:<6} {etf["score"]:>8.4f} {etf["weight"]:>6}')

    print(f'\n  Top 15 排名:')
    for r in signal['ranking']:
        mark = '>>>' if r['selected'] else '   '
        print(f'  {mark} #{r["rank"]:<3} {r["code"]} {r["name"]:<14} '
              f'{r["sector"]:<6} {r["score"]:.4f}')

    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='ETF 国家队轮动 · 月度信号')
    parser.add_argument('--push', action='store_true', help='推送到微信')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不推送')
    args = parser.parse_args()

    # 加载配置
    config = {'pushplus_token': ''}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    elif (PROJECT_ROOT / 'pushplus_config_v4.2.json').exists():
        # 复用 V4.2 的 Token
        with open(PROJECT_ROOT / 'pushplus_config_v4.2.json') as f:
            config = json.load(f)

    # 生成信号
    signal = generate_signal()
    print_signal(signal)

    # 推送
    token = config.get('pushplus_token', '')
    if args.push and token:
        push_to_wechat(signal, token)
    elif args.push and not token:
        print('\n[PushPlus] Token 未配置，请在 pushplus_config.json 中设置')
    elif not args.dry_run and token:
        # 默认：有 Token 就推
        push_to_wechat(signal, token)

    return signal


if __name__ == '__main__':
    main()
