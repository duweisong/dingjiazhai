"""
ETF轮动策略 · 优化版 + PushPlus 信号推送
=========================================

优化要点:
1. 双时间框架趋势确认 (25日 + 50日) — 过滤短期噪声
2. Top-2 70/30 持仓 — 夏普最优, 回撤最低
3. 周频信号生成 — 降低噪声和换手
4. PushPlus 微信推送买卖信号

用法:
  python etf_rotation_live.py              # 检查信号并推送
  python etf_rotation_live.py --backtest   # 回测模式
  python etf_rotation_live.py --dry-run    # 仅打印信号，不推送

依赖:
  pip install efinance pandas numpy scikit-learn requests

配置:
  创建 pushplus_config.json 填入 PushPlus Token
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
import efinance as ef
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 修复Windows GBK编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 全局配置
# ============================================================

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pushplus_config.json')

# ETF候选池
ETF_POOL = {
    '510880': {'name': '红利ETF',   'type': 'A股价值', 'risk': 'low'},
    '159915': {'name': '创业板ETF',  'type': 'A股成长', 'risk': 'high'},
    '513100': {'name': '纳指ETF',    'type': '海外科技', 'risk': 'high'},
    '518880': {'name': '黄金ETF',    'type': '大宗商品', 'risk': 'mid'},
    '511010': {'name': '国债ETF',    'type': '固定收益', 'risk': 'safe'},
}
CODE_LIST = list(ETF_POOL.keys())
SAFE_CODE = '511010'

# 策略参数 (可被config覆盖)
STRATEGY_PARAMS = {
    'trend_short': 25,     # 短期趋势窗口
    'trend_long': 50,      # 长期趋势窗口
    'mom_window': 20,      # 动量窗口
    'vol_window': 60,      # 波动率窗口
    'w_trend_short': 0.30, # 短期趋势权重
    'w_trend_long': 0.25,  # 长期趋势权重
    'w_momentum': 0.25,    # 动量权重
    'w_anti_vol': 0.20,    # 低波动权重
    'use_top2': True,       # Top-2分散持仓 (Sharpe最优, 回撤最低)
    'top1_weight': 0.70,   # Top-1 权重
    'top2_weight': 0.30,   # Top-2 权重
    'score_buffer': 0.0,    # 换仓缓冲
    'min_score': 0.0,       # 最低得分阈值
    'backtest_start': '20150101',
    'backtest_end': '20250828',
}

# PushPlus API
PUSHPLUS_URL = 'http://www.pushplus.plus/send'


# ============================================================
# 配置管理
# ============================================================

def load_config():
    """加载PushPlus配置文件"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 合并策略参数
        if 'strategy' in config:
            STRATEGY_PARAMS.update(config['strategy'])

        return config

    # 默认配置模板
    return {
        'pushplus_token': '',
        'notify_on_signal': True,
        'notify_on_error': True,
        'notify_daily_summary': False,
        'strategy': STRATEGY_PARAMS,
    }


def save_config(config):
    """保存配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# PushPlus 推送模块
# ============================================================

def pushplus_send(token, title, content, template='html'):
    """发送PushPlus消息"""
    if not token:
        print('[PushPlus] 未配置Token，跳过推送')
        return False

    payload = {
        'token': token,
        'title': title,
        'content': content,
        'template': template,
    }

    try:
        resp = requests.post(PUSHPLUS_URL, json=payload, timeout=10)
        result = resp.json()
        if result.get('code') == 200:
            print(f'[PushPlus] 推送成功: {title}')
            return True
        else:
            print(f'[PushPlus] 推送失败: {result}')
            return False
    except Exception as e:
        print(f'[PushPlus] 推送异常: {e}')
        return False


def format_signal_html(signals, scores_df, params):
    """格式化买卖信号为HTML"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    rows_html = ''

    # 所有ETF得分排名
    if scores_df is not None and len(scores_df) > 0:
        rank_df = scores_df.sort_values('综合得分', ascending=False)
        for i, (_, row) in enumerate(rank_df.iterrows()):
            color = '#FF8124' if i < 2 else '#999'
            weight = f"{params['top1_weight']:.0%}" if i == 0 else (f"{params['top2_weight']:.0%}" if i == 1 else '-')
            rows_html += f"""
            <tr style="color:{color}">
                <td>#{i+1}</td>
                <td>{row['名称']}</td>
                <td>{row['代码']}</td>
                <td>{row['类型']}</td>
                <td>{row['综合得分']:.4f}</td>
                <td>{row['趋势_短']:.4f}</td>
                <td>{row['趋势_长']:.4f}</td>
                <td><b>{weight}</b></td>
            </tr>"""

    # 信号详情
    signal_rows = ''
    for s in signals:
        icon = '🟢 买入' if s['action'] == 'BUY' else ('🔴 卖出' if s['action'] == 'SELL' else '🟡 持有')
        signal_rows += f"""
            <tr>
                <td>{icon}</td>
                <td>{s['name']}</td>
                <td>{s['code']}</td>
                <td>{s['weight']}</td>
                <td>{s['reason']}</td>
            </tr>"""

    html = f"""
    <div style="font-family: 'Microsoft YaHei', sans-serif; max-width: 600px;">
        <h2 style="color:#FF8124; border-bottom:2px solid #FF8124; padding-bottom:10px;">
            📊 ETF轮动策略 · 调仓信号
        </h2>
        <p style="color:#666;">生成时间: {now}</p>

        <h3>🎯 建议持仓</h3>
        <table style="width:100%; border-collapse:collapse; border:1px solid #ddd;">
            <tr style="background:#f5f5f5;">
                <th>排名</th><th>ETF</th><th>代码</th><th>类型</th>
                <th>综合得分</th><th>短期趋势</th><th>长期趋势</th><th>建议仓位</th>
            </tr>
            {rows_html}
        </table>

        <h3 style="margin-top:20px;">📋 操作信号</h3>
        <table style="width:100%; border-collapse:collapse; border:1px solid #ddd;">
            <tr style="background:#f5f5f5;">
                <th>操作</th><th>ETF</th><th>代码</th><th>仓位</th><th>原因</th>
            </tr>
            {signal_rows}
        </table>

        <div style="margin-top:20px; padding:10px; background:#fff3e0; border-left:4px solid #FF8124;">
            <b>策略参数:</b> 短趋势{params['trend_short']}日 |
            长趋势{params['trend_long']}日 | Top-2 70/30 |
            周频执行
        </div>

        <p style="color:#999; font-size:12px; margin-top:20px;">
            ⚠️ 本信号仅供学习参考，不构成投资建议<br>
            ETF轮动策略 · 自动生成
        </p>
    </div>
    """
    return html


# ============================================================
# 数据获取
# ============================================================

def fetch_latest_data():
    """获取最新ETF净值数据(用于实盘信号)"""
    print('[数据] 获取最新ETF净值...')

    dfs = {}
    for code in CODE_LIST:
        try:
            df = ef.fund.get_quote_history(code)
            df = df.rename(columns={
                '日期': 'date', '单位净值': 'nav',
                '累计净值': 'cum_nav', '涨跌幅': 'daily_ret'
            })
            df['date'] = pd.to_datetime(df['date'])
            df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce').ffill()
            df = df.sort_values('date').set_index('date')
            dfs[code] = df['cum_nav']
            print(f'  [{code}] {ETF_POOL[code]["name"]}: {len(dfs[code])} 条, '
                  f'最新: {dfs[code].index[-1].date()} NAV={dfs[code].iloc[-1]:.4f}')
        except Exception as e:
            print(f'  [{code}] 获取失败: {e}')

    if not dfs:
        raise RuntimeError('未获取到任何数据')

    data = pd.DataFrame(dfs)
    data = data.ffill().bfill()  # 处理跨市场节假日
    data = data[list(dfs.keys())]
    return data


def fetch_backtest_data():
    """获取历史数据(用于回测)"""
    print('[数据] 获取历史ETF净值...')

    dfs = {}
    for code in CODE_LIST:
        try:
            df = ef.fund.get_quote_history(code)
            df = df.rename(columns={
                '日期': 'date', '累计净值': 'cum_nav'
            })
            df['date'] = pd.to_datetime(df['date'])
            df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce').ffill()
            df = df.sort_values('date').set_index('date')
            dfs[code] = df['cum_nav']
        except Exception as e:
            print(f'  [{code}] 获取失败: {e}')

    data = pd.DataFrame(dfs)
    data = data.ffill().bfill()

    start = STRATEGY_PARAMS['backtest_start']
    end = STRATEGY_PARAMS['backtest_end']
    data = data.loc[start:end]
    print(f'[数据] 范围: {data.index[0].date()} ~ {data.index[-1].date()}, '
          f'{len(data)} 个交易日')
    return data


# ============================================================
# 因子计算
# ============================================================

def calc_trend_score(prices, N):
    """计算趋势得分 = 斜率 × R² (polyfit实现, 兼容rolling)"""
    arr = np.asarray(prices, dtype=float)
    if len(arr) < N or np.isnan(arr).any():
        return np.nan
    x = np.arange(1, N + 1, dtype=float)
    y = arr[-N:] / arr[-N]  # 归一化
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(slope * r2)


def calculate_scores(data, params):
    """计算所有ETF的综合得分"""
    codes = [c for c in CODE_LIST if c in data.columns]
    scores = pd.DataFrame(index=data.index)

    for code in codes:
        prices = data[code]

        # 短期趋势得分
        scores[f'trend_s_{code}'] = (
            prices.rolling(params['trend_short'])
            .apply(lambda x: calc_trend_score(x, params['trend_short']))
        )

        # 长期趋势得分
        scores[f'trend_l_{code}'] = (
            prices.rolling(params['trend_long'])
            .apply(lambda x: calc_trend_score(x, params['trend_long']))
        )

        # 动量得分 (N日收益率)
        scores[f'mom_{code}'] = prices / prices.shift(params['mom_window']) - 1.0

        # 波动率 (年化, 取负值→低波动高得分)
        daily_ret = prices.pct_change()
        scores[f'vol_{code}'] = -daily_ret.rolling(params['vol_window']).std() * np.sqrt(252)

    # 横截面z-score归一化
    for prefix in ['trend_s_', 'trend_l_', 'mom_', 'vol_']:
        cols = [prefix + c for c in codes]
        row_mean = scores[cols].mean(axis=1)
        row_std = scores[cols].std(axis=1).replace(0, 1)
        for col in cols:
            scores[col + '_z'] = (scores[col] - row_mean) / row_std

    # 综合得分
    w = params
    for code in codes:
        scores[f'score_{code}'] = (
            w['w_trend_short'] * scores[f'trend_s_{code}_z'] +
            w['w_trend_long']  * scores[f'trend_l_{code}_z'] +
            w['w_momentum']    * scores[f'mom_{code}_z'] +
            w['w_anti_vol']    * scores[f'vol_{code}_z']
        )

    return scores, codes


# ============================================================
# 信号生成
# ============================================================

def generate_signals(data, scores, active_codes, params, current_holdings=None):
    """
    生成买卖信号

    规则:
    1. 双时间框架得分排名 (25日短趋势 + 50日长趋势)
    2. 选择得分最高的Top-2 ETF
    3. 70/30 固定权重分配
    4. 全市场下行→全仓避险国债

    返回:
        signals: 操作信号列表
        holdings: 最新建议持仓 {code: weight}
        scores_df: 得分排名表
    """
    if current_holdings is None:
        current_holdings = {}

    latest = scores.iloc[-1]
    trend_s_cols = [f'trend_s_{c}' for c in active_codes]
    trend_l_cols = [f'trend_l_{c}' for c in active_codes]

    # 收集得分
    score_list = []
    for code in active_codes:
        score_list.append({
            '代码': code,
            '名称': ETF_POOL[code]['name'],
            '类型': ETF_POOL[code]['type'],
            '综合得分': latest[f'score_{code}'],
            '趋势_短': latest[f'trend_s_{code}'],
            '趋势_长': latest[f'trend_l_{code}'],
        })
    scores_df = pd.DataFrame(score_list).sort_values('综合得分', ascending=False)

    # 全市场下行检查
    all_short_neg = all(latest[c] < 0 for c in trend_s_cols)
    all_long_neg = all(latest[c] < 0 for c in trend_l_cols)

    # 确定持仓
    new_holdings = {}

    if all_short_neg and all_long_neg:
        new_holdings[SAFE_CODE] = 1.0
    else:
        valid = [c for c in scores_df['代码'].tolist()
                 if latest[f'score_{c}'] > params['min_score']]
        if not valid:
            new_holdings[SAFE_CODE] = 1.0
        elif len(valid) == 1 or not params.get('use_top2'):
            new_holdings[valid[0]] = 1.0
        else:
            new_holdings[valid[0]] = params['top1_weight']
            new_holdings[valid[1]] = params['top2_weight']

    if not new_holdings:
        new_holdings[SAFE_CODE] = 1.0

    # 对比生成信号
    signals = []
    old_codes = set(current_holdings.keys())
    new_codes = set(new_holdings.keys())

    # 卖
    for code in old_codes - new_codes:
        signals.append({
            'action': 'SELL',
            'code': code, 'name': ETF_POOL[code]['name'],
            'weight': f'{current_holdings[code]:.0%}->0%',
            'reason': '跌出Top-2'
        })

    # 买
    for code in new_codes - old_codes:
        if old_codes:
            old_best_score = max(
                latest.get(f'score_{c}', -999) for c in old_codes
                if f'score_{c}' in latest.index
            )
            new_score = latest.get(f'score_{code}', 0)
            if new_score < old_best_score * (1 + params['score_buffer']):
                continue
        signals.append({
            'action': 'BUY',
            'code': code, 'name': ETF_POOL[code]['name'],
            'weight': f'{new_holdings[code]:.0%}',
            'reason': '新入选Top-2'
        })

    # 持有
    for code in old_codes & new_codes:
        signals.append({
            'action': 'HOLD',
            'code': code, 'name': ETF_POOL[code]['name'],
            'weight': f'{new_holdings[code]:.0%}',
            'reason': '维持持仓'
        })

    return signals, new_holdings, scores_df


# ============================================================
# 回测引擎 (日频计算 + 周频执行)
# ============================================================

def run_backtest(data, params):
    """
    回测引擎: 每日计算因子 → 每周五生成信号 → 持有一周
    """
    print('\n[回测] 开始回测 (日频因子 + 周频执行)...')

    # 计算日频因子
    daily_scores, active_codes = calculate_scores(data, params)
    daily_scores = daily_scores.dropna()

    # 日收益率
    daily_returns = data[active_codes].pct_change()

    # 对齐
    common_idx = daily_scores.index.intersection(daily_returns.index)
    daily_scores = daily_scores.loc[common_idx]
    daily_returns = daily_returns.loc[common_idx]

    # 周五列表
    friday_mask = pd.Series(common_idx, index=common_idx).apply(
        lambda d: d.dayofweek == 4
    )
    fridays = set(common_idx[friday_mask])

    print(f'[回测] 有效数据: {len(common_idx)} 天, 信号日(周五): {len(fridays)} 个')

    holdings = {}      # 当前持仓 {code: weight}
    nav = 1.0
    nav_history = pd.Series(1.0, index=common_idx, dtype=float)
    trades_log = []

    for i, date in enumerate(common_idx):
        # 周五检查信号
        if date in fridays and i > 0:
            signals, suggested, _ = generate_signals(
                data, daily_scores.loc[:date], active_codes, params, holdings
            )

            # 检查是否有实质换仓 (品种变化)
            old_set = set(holdings.keys())
            new_set = set(suggested.keys())
            has_change = old_set != new_set

            if has_change:
                # 记录交易 (不含HOLD)
                for s in signals:
                    if s['action'] in ('BUY', 'SELL'):
                        trades_log.append({'date': date, **s})
                # 换仓成本
                turnover = sum(abs(holdings.get(c, 0) - suggested.get(c, 0))
                              for c in old_set | new_set)
                nav *= (1 - turnover * 0.001)
                holdings = suggested
            # 无品种变化 → 维持原持仓,不产生交易

        # 计算当日收益
        if holdings:
            day_ret = sum(
                holdings.get(code, 0) * daily_returns[code].iloc[i]
                for code in holdings
                if not np.isnan(daily_returns[code].iloc[i])
            )
            nav *= (1 + day_ret)

        nav_history[date] = nav

    # 绩效统计
    nav_history = nav_history.dropna()
    total_ret = nav_history.iloc[-1] - 1
    years = len(nav_history) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1
    rets = nav_history.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = nav_history.expanding().max()
    max_dd = float(((nav_history / peak) - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    print(f'\n{"="*60}')
    print(f'  回测结果 (日频因子 + 周频执行)')
    print(f'{"="*60}')
    print(f'  区间: {nav_history.index[0].date()} ~ {nav_history.index[-1].date()}')
    print(f'  年数: {years:.1f}, 信号日: {len(fridays)}')
    print(f'  累计收益: {total_ret:.2%}')
    print(f'  年化收益: {cagr:.2%}')
    print(f'  夏普比率: {sharpe:.2f}')
    print(f'  卡玛比率: {calmar:.2f}')
    print(f'  最大回撤: {max_dd:.2%}')
    print(f'  交易次数: {len(trades_log)}')
    print(f'  年均交易: {len(trades_log)/max(years,0.1):.1f}')

    # 年度收益
    annual = nav_history.resample('YE').last()
    print(f'\n  年度收益:')
    prev_v = 1.0
    for d, v in annual.items():
        print(f'    {d.year}: {(v/prev_v-1):+.2%}')
        prev_v = v

    # 与纳指ETF对比
    ndx = data['513100']
    ndx_nav = ndx / ndx.loc[nav_history.index[0]]
    ndx_nav = ndx_nav.loc[nav_history.index]
    ndx_ret = ndx_nav.iloc[-1] - 1
    yr = len(ndx_nav) / 252
    ndx_cagr = (1 + ndx_ret) ** (1 / yr) - 1
    print(f'\n  对比: BH纳指ETF 同期年化 {ndx_cagr:.2%}')
    print(f'  超额: {cagr - ndx_cagr:+.2%}')

    return nav_history, trades_log


# ============================================================
# 主程序
# ============================================================

def run_live(config):
    """实盘信号模式"""
    print('\n' + '=' * 60)
    print('  ETF轮动策略 · 实时信号检查')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    params = STRATEGY_PARAMS

    # 1. 获取数据
    data = fetch_latest_data()

    # 2. 计算得分
    scores, active_codes = calculate_scores(data, params)
    latest_date = scores.index[-1]
    print(f'\n[信号] 最新数据日期: {latest_date.date()}')

    # 3. 生成信号 (无当前持仓假设)
    signals, new_holdings, scores_df = generate_signals(
        data, scores, active_codes, params
    )

    # 4. 打印信号
    print(f'\n{"="*60}')
    print(f'  📊 综合得分排名')
    print(f'{"="*60}')
    for _, row in scores_df.iterrows():
        bar = '█' * max(1, int(row['综合得分'] * 50)) if row['综合得分'] > 0 else ''
        print(f'  {row["代码"]} {row["名称"]:<8} 得分:{row["综合得分"]:+7.4f}  '
              f'短:{row["趋势_短"]:+7.4f} 长:{row["趋势_长"]:+7.4f}  {bar}')

    print(f'\n{"="*60}')
    print(f'  📋 建议持仓')
    print(f'{"="*60}')
    for code, weight in new_holdings.items():
        name = ETF_POOL[code]['name']
        type_ = ETF_POOL[code]['type']
        print(f'  {code} {name:<8} ({type_}) → {weight:.0%}')

    print(f'\n{"="*60}')
    print(f'  🔔 交易信号')
    print(f'{"="*60}')
    for s in signals:
        icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'ADJUST': '🔵'}.get(s['action'], '⚪')
        print(f'  {icon} {s["action"]:<8} {s["code"]} {s["name"]:<8} '
              f'仓位:{s["weight"]:<10} {s["reason"]}')

    # 5. PushPlus推送
    if config.get('pushplus_token') and config.get('notify_on_signal'):
        has_trade = any(s['action'] in ('BUY', 'SELL') for s in signals)

        if has_trade:
            title = '🔔 ETF轮动调仓信号'
            content = format_signal_html(signals, scores_df, params)
            pushplus_send(config['pushplus_token'], title, content)
        else:
            print('\n[PushPlus] 无调仓信号，不推送')

    return signals, new_holdings, scores_df


def run_dry_run(config):
    """试运行模式 (不推送)"""
    # 临时禁用推送
    config['notify_on_signal'] = False
    return run_live(config)


def main():
    parser = argparse.ArgumentParser(description='ETF轮动策略 · 优化版')
    parser.add_argument('--backtest', action='store_true', help='回测模式')
    parser.add_argument('--dry-run', action='store_true', help='试运行(不推送)')
    parser.add_argument('--init-config', action='store_true', help='初始化配置文件')
    args = parser.parse_args()

    # 初始化配置
    if args.init_config or not os.path.exists(CONFIG_FILE):
        config = load_config()
        if not config.get('pushplus_token') and sys.stdin.isatty():
            print('=' * 60)
            print('  首次使用，需要配置 PushPlus Token')
            print('=' * 60)
            print()
            print('  1. 访问 http://www.pushplus.plus 注册账号')
            print('  2. 在"发送消息"页面获取你的 Token')
            print('  3. 将 Token 填入配置文件')
            print()
            token = input('  请输入 PushPlus Token (回车跳过): ').strip()
            if token:
                config['pushplus_token'] = token
                save_config(config)
                print(f'  Token已保存到 {CONFIG_FILE}')
            else:
                print('  未设置Token，将在本地模式运行')
                save_config(config)
        elif not config.get('pushplus_token'):
            print(f'  配置文件 {CONFIG_FILE} 已创建，请编辑填入 pushplus_token')
            save_config(config)
        if args.init_config:
            return

    config = load_config()

    if args.backtest:
        # 回测模式
        data = fetch_backtest_data()
        nav, trades = run_backtest(data, STRATEGY_PARAMS)
    elif args.dry_run:
        # 试运行
        run_dry_run(config)
    else:
        # 实盘信号模式
        run_live(config)


if __name__ == '__main__':
    main()
