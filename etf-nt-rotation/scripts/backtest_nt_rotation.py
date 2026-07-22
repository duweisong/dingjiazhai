"""
ETF 国家队轮动策略 — 核心回测脚本
==================================

基于月度动量+量比双因子排名，选取 5-20 名中段 5 只 ETF，
月初等权买入、月末卖出。集成预测仓位信号。

用法:
  python backtest_nt_rotation.py              # 默认参数回测
  python backtest_nt_rotation.py --baseline   # 仅跑等权基准
  python backtest_nt_rotation.py --compare    # 对比多种参数组合
"""
import sys
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / '.cache'
DATA_FILE = CACHE_DIR / 'etf_sector' / 'nt_rotation_35.parquet'
POSITION_FILE = CACHE_DIR / 'national_team' / 'predicted_position.json'

# ============================================================
# ETF 池定义
# ============================================================
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

# ============================================================
# 参数配置
# ============================================================
@dataclass
class Config:
    # 因子参数（v2: 经 152 组网格搜索优化）
    momentum_weight: float = 0.50  # 50/50 > 60/40
    volatility_weight: float = 0.50
    mom_bars: int = 14        # 14d > 21d，更快捕获轮动
    vol_short: int = 5
    vol_long: int = 44

    # 选股参数（v3: Top5 从 Top10 中选）
    rank_min: int = 1
    rank_max: int = 10        # 前 10 名候选
    top_n: int = 5            # 持仓 5 只
    selection_mode: str = 'top'  # 'top'=取头部, 'mid'=取中段(已证伪)

    # 交易参数
    trade_cost: float = 0.001  # 单边手续费
    use_position_scaling: bool = True  # 是否启用预测仓位缩放

    # 回测区间
    start_date: str = '20220101'
    end_date: str = '20260610'

    def __repr__(self):
        return (f'Config(w={self.momentum_weight}/{self.volatility_weight}, '
                f'rank={self.rank_min}-{self.rank_max}, n={self.top_n}, '
                f'mom={self.mom_bars}d, vol={self.vol_short}/{self.vol_long}d)')


# ============================================================
# 数据加载
# ============================================================
def load_data(config: Config) -> Tuple[pd.DataFrame, List[str]]:
    """加载 ETF 净值数据，返回 (data, active_codes)。"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'数据文件不存在: {DATA_FILE}\n请先运行: python scripts/fetch_etf_data.py')

    data = pd.read_parquet(DATA_FILE)
    data = data.loc[config.start_date:config.end_date]
    data = data.ffill().bfill()

    # 筛选有数据的 ETF
    active = [c for c in data.columns if c in ETF_INFO and data[c].dropna().count() >= 100]
    data = data[active]

    print(f'[Data] {len(data)} days x {len(active)} ETFs ({data.index[0].date()} ~ {data.index[-1].date()})')
    return data, active


def load_position_signal() -> float:
    """加载预测仓位信号。不可用时返回 1.0（满仓）。"""
    if not POSITION_FILE.exists():
        return 1.0
    try:
        data = json.loads(POSITION_FILE.read_text())
        age = (datetime.now() - datetime.fromisoformat(data['date'])).days
        if age < 7:
            return data['predicted_position']
    except Exception:
        pass
    return 1.0


# ============================================================
# 因子计算
# ============================================================
def calc_factors(data: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    计算动量 + 量比因子。

    动量: price / price.shift(21) - 1
    量比: abs_ret.rolling(5).mean() / abs_ret.rolling(44).mean()
          (注意：用绝对值收益波动率替代实际成交量，因为基金净值数据不含成交量)

    Returns: scores, momentum, volatility_ratio
    """
    p = config
    momentum = pd.DataFrame(index=data.index, dtype=float)
    vol_ratio = pd.DataFrame(index=data.index, dtype=float)

    for c in data.columns:
        # 动量因子
        momentum[c] = data[c] / data[c].shift(p.mom_bars) - 1.0

        # 量比因子（用绝对值收益波动率替代实际成交量）
        abs_ret = data[c].pct_change().abs()
        short_ma = abs_ret.rolling(p.vol_short).mean()
        long_ma = abs_ret.rolling(p.vol_long).mean().replace(0, np.nan)
        vol_ratio[c] = short_ma / long_ma

    # 综合得分：percentile rank 加权
    scores = pd.DataFrame(index=data.index, dtype=float)
    for date in momentum.index:
        mom_row = momentum.loc[date].dropna()
        vol_row = vol_ratio.loc[date].dropna()
        common = mom_row.index.intersection(vol_row.index)

        if len(common) < p.top_n:
            continue

        # Percentile rank (0~1, 越高越好)
        mom_rank = mom_row[common].rank(pct=True)
        vol_rank = vol_row[common].rank(pct=True)

        scores.loc[date, common] = (
            p.momentum_weight * mom_rank +
            p.volatility_weight * vol_rank
        )

    scores = scores.dropna(how='all')
    print(f'[Factors] {len(scores)} valid days with scores')
    return scores, momentum, vol_ratio


# ============================================================
# 选股逻辑
# ============================================================
def select_etfs(scores_row: pd.Series, config: Config, position: float = 1.0) -> Dict[str, float]:
    """
    从当日得分中选出 top_n 只 ETF。

    mode='top': 取排名 rank_min~rank_max 的前 top_n 只（默认 1-5，即头部 5 只）
    mode='mid': 取排名 rank_min~rank_max 的中间 top_n 只（原方案，已被数据否决）
    """
    ranked = scores_row.dropna().sort_values(ascending=False)
    n = len(ranked)

    if n < config.top_n:
        return {}

    lo = min(config.rank_min, n)
    hi = min(config.rank_max, n)
    if hi <= lo:
        hi = n
        lo = 1
    candidates = ranked.iloc[lo - 1:hi]
    m = len(candidates)

    if config.selection_mode == 'mid':
        # 中段选股（已证伪：A股动量头部集中，跳过头部=放弃收益）
        if m <= config.top_n:
            selected_codes = list(candidates.index)
        else:
            start = (m - config.top_n) // 2
            selected_codes = list(candidates.index[start:start + config.top_n])
    else:
        # 头部选股（推荐）：直接取前 top_n 名
        selected_codes = list(candidates.index[:config.top_n])

    weight = (1.0 / len(selected_codes)) * position
    return {c: weight for c in selected_codes}


# ============================================================
# 交易日历
# ============================================================
def get_trade_schedule(dates: pd.DatetimeIndex) -> List[dict]:
    """生成月度调仓时间表。返回 [{month, buy_date, sell_date}, ...]"""
    schedule = []
    months = sorted(set((d.year, d.month) for d in dates))
    for yr, mo in months:
        month_dates = [d for d in dates if d.year == yr and d.month == mo]
        if not month_dates:
            continue
        schedule.append({
            'month': f'{yr}-{mo:02d}',
            'buy': min(month_dates),
            'sell': max(month_dates),
        })
    return schedule


# ============================================================
# 回测主循环
# ============================================================
def run_backtest(data: pd.DataFrame, active_codes: List[str],
                 config: Config, position_signal: float = 1.0) -> Dict:
    """执行完整回测。"""
    p = config

    # 计算因子
    scores, momentum, vol_ratio = calc_factors(data[active_codes], config)

    # 日收益率（用于持仓期间计算）
    daily_ret = data[active_codes].pct_change()

    # 调仓时间表
    schedule = get_trade_schedule(scores.index)
    print(f'[Schedule] {len(schedule)} months')

    # 回测状态
    holdings: Dict[str, float] = {}  # {code: weight}
    nav = 1.0
    nav_history = pd.Series(1.0, index=scores.index, dtype=float)
    trades: List[Dict] = []

    for i, date in enumerate(scores.index):
        is_buy = any(t['buy'].date() == date.date() for t in schedule)
        is_sell = any(t['sell'].date() == date.date() for t in schedule)

        # === 卖出（先结算当日收益，再清仓） ===
        if is_sell and holdings:
            if date in daily_ret.index:
                day_ret = 0.0
                for c, w in holdings.items():
                    r = daily_ret[c].loc[date]
                    if not np.isnan(r):
                        day_ret += w * r
                nav *= (1 + day_ret)
            for c in list(holdings.keys()):
                trades.append({'date': date, 'action': 'SELL', 'code': c})
            holdings = {}

        # === 买入（月初建仓） ===
        if is_buy and not holdings:
            sr = scores.loc[date].dropna()
            if len(sr) >= p.top_n:
                selected = select_etfs(sr, config, position_signal)
                if selected:
                    for c in selected:
                        trades.append({'date': date, 'action': 'BUY', 'code': c})
                    nav *= (1 - p.trade_cost)
                    holdings = selected

        # === 持仓期净值变动 ===
        if holdings and not is_sell:
            if date in daily_ret.index:
                day_ret = 0.0
                for c, w in holdings.items():
                    r = daily_ret[c].loc[date]
                    if not np.isnan(r):
                        day_ret += w * r
                nav *= (1 + day_ret)

        nav_history[date] = nav

    # === 绩效计算 ===
    nav_history = nav_history.dropna()
    returns = nav_history.pct_change().dropna()

    total_return = nav_history.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / max(years, 0.1)) - 1
    annual_vol = float(returns.std() * np.sqrt(252))

    # 夏普（无风险利率 2%）
    excess = returns - 0.02 / 252
    sharpe = float(excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # 最大回撤
    peak = nav_history.expanding().max()
    drawdown = (nav_history / peak) - 1
    max_dd = float(drawdown.min())
    max_dd_days = 0
    dd_start = None
    for d in drawdown.index:
        if drawdown[d] < 0 and dd_start is None:
            dd_start = d
        elif drawdown[d] >= 0 and dd_start is not None:
            max_dd_days = max(max_dd_days, (d - dd_start).days)
            dd_start = None

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # 胜率
    win_rate = float((returns > 0).sum() / len(returns))
    pos_returns = returns[returns != 0]
    pos_win_rate = float((pos_returns > 0).sum() / len(pos_returns)) if len(pos_returns) > 0 else 0

    # 月度统计
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    month_win_rate = float((monthly > 0).sum() / len(monthly)) if len(monthly) > 0 else 0

    # 等权基准
    bench_ret = daily_ret[active_codes].mean(axis=1).loc[nav_history.index]
    bench_nav = (1.0 + bench_ret).cumprod()
    bench_total = bench_nav.iloc[-1] - 1
    bench_cagr = (1 + bench_total) ** (1 / max(years, 0.1)) - 1

    # 仓位统计
    months_in_position = sum(1 for t in schedule if any(
        tr['date'].date() == t['buy'].date() and tr['action'] == 'BUY'
        for tr in trades))
    avg_holdings = months_in_position / max(len(schedule), 1) * config.top_n

    return {
        'config': repr(config),
        'nav': nav_history,
        'bench_nav': bench_nav,
        'returns': returns,
        'monthly_returns': monthly,
        'trades': trades,
        'schedule_months': len(schedule),
        'position_signal': position_signal,
        'metrics': {
            'total_return': total_return,
            'cagr': cagr,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'max_dd_days': max_dd_days,
            'calmar': calmar,
            'win_rate_daily': win_rate,
            'win_rate_pos': pos_win_rate,
            'win_rate_monthly': month_win_rate,
            'total_trades': len(trades),
            'bench_cagr': bench_cagr,
            'bench_total': bench_total,
            'avg_holdings': avg_holdings,
        },
    }


# ============================================================
# 报告输出
# ============================================================
def print_report(result: Dict):
    """打印格式化的绩效报告。"""
    m = result['metrics']
    cfg = result['config']

    print()
    print('=' * 65)
    print('  ETF 国家队轮动策略 — 回测绩效报告')
    print('=' * 65)
    print(f'  参数: {cfg}')
    print(f'  调仓: {result["schedule_months"]} 个月')
    print(f'  仓位信号: {result["position_signal"]:.0%} (预测仓位)')
    print(f'  交易笔数: {m["total_trades"]}')

    print()
    print(f'  {"指标":<20} {"策略":>12} {"基准(等权)":>12} {"差异":>10}')
    print(f'  {"-"*54}')
    items = [
        ('累计收益', m['total_return'], m['bench_total'], '.2%'),
        ('年化收益', m['cagr'], m['bench_cagr'], '.2%'),
        ('年化波动', m['annual_vol'], None, '.2%'),
        ('夏普比率', m['sharpe'], None, '.2f'),
        ('最大回撤', m['max_drawdown'], None, '.2%'),
    ]
    for label, v, b, fmt in items:
        if b is not None:
            print(f'  {label:<20} {v:>12{fmt}}  {b:>12{fmt}}  {v-b:>+10{fmt}}')
        else:
            print(f'  {label:<20} {v:>12{fmt}}  {"N/A":>12}')

    print(f'  {"最大回撤持续":<20} {m["max_dd_days"]:>11}天')
    print(f'  {"Calmar比率":<20} {m["calmar"]:>12.2f}')
    print(f'  {"日胜率":<20} {m["win_rate_daily"]:>11.1%}')
    print(f'  {"持仓日胜率":<20} {m["win_rate_pos"]:>11.1%}')
    print(f'  {"月胜率":<20} {m["win_rate_monthly"]:>11.1%}')
    print(f'  {"平均持仓":<20} {m["avg_holdings"]:>11.1f} 只/月')
    print('=' * 65)

    # 年度收益
    if result['monthly_returns'] is not None and len(result['monthly_returns']) > 0:
        annual = result['monthly_returns'].groupby(result['monthly_returns'].index.year).apply(
            lambda x: (1 + x).prod() - 1)
        print(f'\n  年度收益:')
        for yr, r in annual.items():
            marker = '  *** BEST' if r == annual.max() else ('  --- WORST' if r == annual.min() else '')
            print(f'    {yr}: {r:>+8.2%}{marker}')

    # vs 基准
    print(f'\n  超额收益: {m["cagr"] - m["bench_cagr"]:+.2%} (vs 等权持有)')

    return result


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='ETF 国家队轮动策略回测')
    parser.add_argument('--baseline', action='store_true', help='仅跑等权基准')
    parser.add_argument('--no-position', action='store_true', help='不使用预测仓位信号')
    parser.add_argument('--compare', action='store_true', help='对比多种参数组合')
    args = parser.parse_args()

    # 加载数据
    print('=' * 65)
    print('  ETF 国家队轮动策略 回测 v1.0')
    print('=' * 65)
    data, active = load_data(Config())
    print(f'[Data] ETF pool: {len(active)} active')

    # 仓位信号
    position = 1.0
    if not args.no_position:
        position = load_position_signal()
        print(f'[Signal] Predicted position: {position:.0%}')

    if args.baseline:
        # 仅跑等权基准
        daily_ret = data[active].pct_change()
        bench_ret = daily_ret.mean(axis=1)
        bench_nav = (1.0 + bench_ret).cumprod()
        bench_total = bench_nav.iloc[-1] - 1
        yrs = len(bench_nav) / 252
        bench_cagr = (1 + bench_total) ** (1 / max(yrs, 0.1)) - 1
        bench_dd = float(((bench_nav / bench_nav.expanding().max()) - 1).min())
        print(f'\n  等权基准: CAGR={bench_cagr:.2%}, MaxDD={bench_dd:.2%}, '
              f'Total={bench_total:.2%}')
        return

    if args.compare:
        # 对比多种参数组合
        configs = [
            Config(momentum_weight=0.6, volatility_weight=0.4, rank_min=1, rank_max=5, top_n=5),
            Config(momentum_weight=0.5, volatility_weight=0.5, rank_min=1, rank_max=5, top_n=5),
            Config(momentum_weight=0.7, volatility_weight=0.3, rank_min=1, rank_max=5, top_n=5),
            Config(momentum_weight=0.6, volatility_weight=0.4, rank_min=2, rank_max=7, top_n=5),
            Config(momentum_weight=0.6, volatility_weight=0.4, rank_min=1, rank_max=5, top_n=3),
            # 中段选股对照（已证伪，保留作对比）
            Config(momentum_weight=0.6, volatility_weight=0.4, rank_min=5, rank_max=20, top_n=5, selection_mode='mid'),
        ]
        results = []
        for cfg in configs:
            print(f'\n{"-"*50}')
            print(f'  Testing: {cfg}')
            r = run_backtest(data, active, cfg, position)
            m = r['metrics']
            results.append((cfg, m))
            print(f'  CAGR={m["cagr"]:.2%}  Sharpe={m["sharpe"]:.2f}  '
                  f'MaxDD={m["max_drawdown"]:.2%}  WinRate={m["win_rate_monthly"]:.1%}')

        # 排名
        print(f'\n{"="*65}')
        print(f'  参数对比排名 (按夏普)')
        print(f'{"="*65}')
        ranked = sorted(results, key=lambda x: x[1]['sharpe'], reverse=True)
        for i, (cfg, m) in enumerate(ranked):
            print(f'  #{i+1} Sharpe={m["sharpe"]:.2f} CAGR={m["cagr"]:.2%} '
                  f'MaxDD={m["max_drawdown"]:.2%} | {cfg}')
        print(f'\n  Best: {ranked[0][0]}')
        return

    # 默认：单次回测
    result = run_backtest(data, active, Config(), position)
    print_report(result)
    return result


if __name__ == '__main__':
    main()
