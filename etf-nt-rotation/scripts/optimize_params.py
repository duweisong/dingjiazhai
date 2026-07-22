"""
ETF 国家队轮动策略 — 参数优化（T05）
====================================

网格搜索 + 增强策略测试。找到全局最优参数组合。

用法:
  python optimize_params.py           # 全面网格搜索
  python optimize_params.py --quick   # 快速粗搜索（~50组）
  python optimize_params.py --full    # 完整搜索（~300组）
"""
import sys
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import itertools, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'etf-nt-rotation' / 'scripts'))
from backtest_nt_rotation import (
    load_data, calc_factors, select_etfs, get_trade_schedule,
    Config, ETF_INFO,
)

DATA_FILE = PROJECT_ROOT / '.cache' / 'etf_sector' / 'nt_rotation_35.parquet'
REPORT_FILE = PROJECT_ROOT / 'etf-nt-rotation' / 'docs' / 'specs' / 'optimization_report.md'


@dataclass
class EnhancedConfig:
    """增强配置：在基础 Config 上增加过滤器"""
    # 基础因子
    momentum_weight: float = 0.60
    volatility_weight: float = 0.40
    mom_bars: int = 21
    vol_short: int = 5
    vol_long: int = 44
    # 选股
    rank_min: int = 1
    rank_max: int = 5
    top_n: int = 5
    # 过滤器
    min_momentum: float = -1.0     # 绝对动量最低阈值（-1=不过滤）
    market_ma: int = 0             # 市场MA闸门（0=关闭）
    position_mode: str = 'fixed'   # 'fixed'/'predicted'/'optimized'
    position_value: float = 1.0    # 固定仓位值
    trade_cost: float = 0.001

    def to_base_config(self) -> Config:
        return Config(
            momentum_weight=self.momentum_weight,
            volatility_weight=self.volatility_weight,
            mom_bars=self.mom_bars,
            vol_short=self.vol_short,
            vol_long=self.vol_long,
            rank_min=self.rank_min,
            rank_max=self.rank_max,
            top_n=self.top_n,
            trade_cost=self.trade_cost,
        )

    def __repr__(self):
        parts = [f'w={self.momentum_weight}/{self.volatility_weight}',
                 f'mom={self.mom_bars}d', f'vol={self.vol_short}/{self.vol_long}d',
                 f'rank={self.rank_min}-{self.rank_max}', f'n={self.top_n}']
        if self.min_momentum > -1:
            parts.append(f'minMom>{self.min_momentum:.0%}')
        if self.market_ma > 0:
            parts.append(f'MA{self.market_ma}')
        parts.append(f'pos={self.position_mode}:{self.position_value:.0%}')
        return ' | '.join(parts)


def run_single_backtest(data: pd.DataFrame, active: List[str],
                        cfg: EnhancedConfig) -> Dict:
    """运行单次回测，返回绩效指标。"""
    base_cfg = cfg.to_base_config()
    scores, momentum, vol_ratio = calc_factors(data, base_cfg)
    daily_ret = data.pct_change()
    schedule = get_trade_schedule(scores.index)

    nav = 1.0
    holdings = {}

    for i, date in enumerate(scores.index):
        is_buy = any(t['buy'].date() == date.date() for t in schedule)
        is_sell = any(t['sell'].date() == date.date() for t in schedule)

        # 市场MA闸门
        market_ok = True
        if cfg.market_ma > 0:
            # 用第一只ETF作为市场代理
            proxy = data.iloc[:, 0]
            if i >= cfg.market_ma:
                ma = proxy.iloc[i - cfg.market_ma:i + 1].mean()
                market_ok = proxy.iloc[i] > ma

        # === 卖出 ===
        if is_sell and holdings:
            if date in daily_ret.index:
                day_ret = 0.0
                for c, w in holdings.items():
                    r = daily_ret[c].loc[date]
                    if not np.isnan(r):
                        day_ret += w * r
                nav *= (1 + day_ret)
            holdings = {}

        # === 买入 ===
        if is_buy and not holdings:
            if market_ok:
                sr = scores.loc[date].dropna()
                # 绝对动量过滤
                if cfg.min_momentum > -1 and date in momentum.index:
                    mr = momentum.loc[date]
                    sr = sr[sr.index.map(lambda c: mr.get(c, -999) >= cfg.min_momentum)]

                if len(sr) >= max(cfg.top_n, 1):
                    selected = select_etfs(sr, base_cfg, 1.0)
                    if selected:
                        nav *= (1 - cfg.trade_cost)
                        # 应用仓位缩放
                        pos_scale = cfg.position_value
                        holdings = {c: w * pos_scale for c, w in selected.items()}

        # === 持仓收益 ===
        if holdings and not is_sell:
            if date in daily_ret.index:
                day_ret = 0.0
                for c, w in holdings.items():
                    r = daily_ret[c].loc[date]
                    if not np.isnan(r):
                        day_ret += w * r
                nav *= (1 + day_ret)

    # 绩效
    nav_series = pd.Series(nav, index=scores.index).reindex(scores.index).ffill()
    rets = nav_series.pct_change().dropna()
    if len(rets) < 20:
        return {'cagr': -99, 'sharpe': -99, 'max_dd': -99, 'score': -99}

    total = nav - 1
    yrs = len(rets) / 252
    cagr = (1 + total) ** (1 / max(yrs, 0.1)) - 1
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = nav_series.expanding().max()
    max_dd = float(((nav_series / peak) - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    month_wr = float((monthly > 0).sum() / len(monthly)) if len(monthly) > 0 else 0

    # 综合评分（平衡收益与风险）
    score = cagr * 0.35 + sharpe * 0.30 + calmar * 0.20 + month_wr * 0.15

    return {
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'calmar': calmar, 'month_wr': month_wr, 'score': score,
        'total': total, 'months': len(schedule),
    }


def grid_search(data, active, param_grid: Dict, top_n: int = 30) -> pd.DataFrame:
    """网格搜索，返回按 score 排序的结果。"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f'[Grid] {len(keys)} dimensions, {total} combinations')

    results = []
    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        cfg = EnhancedConfig(**{k: v for k, v in params.items()
                                if k in EnhancedConfig.__dataclass_fields__})
        metrics = run_single_backtest(data, active, cfg)
        results.append({**params, **metrics})

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{total}...')

    df = pd.DataFrame(results).sort_values('score', ascending=False)
    return df.head(top_n)


def analyze_results(df: pd.DataFrame) -> str:
    """分析结果，生成优化报告。"""
    lines = []
    lines.append('# ETF 国家队轮动策略 — 参数优化报告')
    lines.append(f'\n> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'> 测试组合: {len(df)} 组\n')

    lines.append('## 1. Top 20 最优参数\n')
    lines.append('| # | Score | CAGR | Sharpe | MaxDD | Calmar | WinRate | 参数 |')
    lines.append('|---|------|------|--------|-------|--------|--------|------|')
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        cfg = EnhancedConfig(**{k: row[k] for k in EnhancedConfig.__dataclass_fields__
                                 if k in row.index})
        lines.append(f'| {i+1} | {row["score"]:.4f} | {row["cagr"]:.1%} | '
                     f'{row["sharpe"]:.2f} | {row["max_dd"]:.1%} | {row["calmar"]:.2f} | '
                     f'{row["month_wr"]:.1%} | {cfg} |')

    lines.append('\n## 2. 参数敏感性分析\n')

    # 分析各参数的影响
    for param in ['momentum_weight', 'mom_bars', 'top_n', 'min_momentum', 'market_ma']:
        if param not in df.columns or df[param].nunique() <= 1:
            continue
        grouped = df.groupby(param)['score'].mean()
        best_val = grouped.idxmax()
        lines.append(f'- **{param}**: 最优值={best_val}, 最佳平均 Score={grouped.max():.4f}')
        lines.append(f'  取值: {dict(grouped.round(4))}')

    lines.append('\n## 3. 过滤器效果\n')

    # 绝对动量过滤
    if 'min_momentum' in df.columns and df['min_momentum'].nunique() > 1:
        base = df[df['min_momentum'] == -1.0]['score'].mean()
        filtered = df[df['min_momentum'] > -1.0]['score'].mean()
        lines.append(f'- 绝对动量过滤: 关闭平均Score={base:.4f}, 开启平均Score={filtered:.4f} '
                     f'({"+" if filtered > base else ""}{filtered - base:+.4f})')

    # MA闸门
    if 'market_ma' in df.columns and df['market_ma'].nunique() > 1:
        base = df[df['market_ma'] == 0]['score'].mean()
        gated = df[df['market_ma'] > 0]['score'].mean()
        lines.append(f'- MA闸门: 关闭平均Score={base:.4f}, 开启平均Score={gated:.4f} '
                     f'({"+" if gated > base else ""}{gated - base:+.4f})')

    lines.append('\n## 4. 推荐配置\n')
    best = df.iloc[0]
    best_cfg = EnhancedConfig(**{k: best[k] for k in EnhancedConfig.__dataclass_fields__
                                  if k in best.index})
    lines.append(f'**综合最优 (Score={best["score"]:.4f})**:')
    lines.append(f'- CAGR={best["cagr"]:.1%}, Sharpe={best["sharpe"]:.2f}, '
                 f'MaxDD={best["max_dd"]:.1%}')
    lines.append(f'- 参数: {best_cfg}')

    # 不同目标的最优
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    best_cagr = df.loc[df['cagr'].idxmax()]
    best_calmar = df.loc[df['calmar'].idxmax()]
    lines.append(f'\n**夏普最优**: CAGR={best_sharpe["cagr"]:.1%}, '
                 f'Sharpe={best_sharpe["sharpe"]:.2f}')
    lines.append(f'**收益最优**: CAGR={best_cagr["cagr"]:.1%}, '
                 f'MaxDD={best_cagr["max_dd"]:.1%}')
    lines.append(f'**回撤最优**: CAGR={best_calmar["cagr"]:.1%}, '
                 f'MaxDD={best_calmar["max_dd"]:.1%}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='ETF轮动参数优化')
    parser.add_argument('--quick', action='store_true', help='快速粗搜索')
    parser.add_argument('--full', action='store_true', help='完整搜索')
    args = parser.parse_args()

    print('=' * 60)
    print('  ETF 国家队轮动 — 参数优化')
    print('=' * 60)

    data = pd.read_parquet(DATA_FILE).loc['20220101':'20260610'].ffill().bfill()
    active = [c for c in data.columns if c in ETF_INFO and data[c].dropna().count() >= 100]
    data = data[active]
    print(f'[Data] {len(data)} days x {len(active)} ETFs')

    if args.quick:
        param_grid = {
            'momentum_weight': [0.5, 0.6, 0.7, 0.8],
            'volatility_weight': [0.2, 0.3, 0.4, 0.5],
            'mom_bars': [14, 21, 33],
            'top_n': [3, 5, 8],
            'rank_max': [5, 8],
            'min_momentum': [-1.0, -0.03, -0.05],
            'market_ma': [0, 50],
            'position_value': [0.7, 0.85, 1.0],
        }
    elif args.full:
        param_grid = {
            'momentum_weight': [0.4, 0.5, 0.6, 0.7, 0.8],
            'volatility_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
            'mom_bars': [10, 14, 21, 33, 44],
            'vol_short': [3, 5],
            'vol_long': [22, 44, 66],
            'top_n': [3, 5, 8, 10],
            'rank_min': [1],
            'rank_max': [3, 5, 8, 10],
            'min_momentum': [-1.0, -0.02, -0.03, -0.05],
            'market_ma': [0, 50, 60],
            'position_value': [0.7, 0.85, 1.0],
        }
    else:
        # 默认：中等粒度搜索
        param_grid = {
            'momentum_weight': [0.5, 0.6, 0.7, 0.8],
            'volatility_weight': [0.2, 0.3, 0.4, 0.5],
            'mom_bars': [10, 14, 21, 33],
            'vol_long': [22, 44, 66],
            'top_n': [3, 5, 8],
            'rank_max': [5, 8],
            'min_momentum': [-1.0, -0.03],
            'market_ma': [0, 50],
            'position_value': [0.85, 1.0],
        }

    # 去重：volatility_weight = 1 - momentum_weight
    valid_combos = [(w, v) for w in param_grid['momentum_weight']
                    for v in param_grid['volatility_weight']
                    if abs(w + v - 1.0) < 0.01]
    if valid_combos:
        param_grid['momentum_weight'] = list(set(p[0] for p in valid_combos))
        param_grid['volatility_weight'] = list(set(p[1] for p in valid_combos))
        # Filter to valid pairs during iteration
        total = 1
        for k, v in param_grid.items():
            if k == 'volatility_weight':
                continue
            total *= len(v)
        print(f'[Grid] {len(param_grid)-1} dims, ~{total} combos (weight pairs: {len(valid_combos)})')

    df = grid_search(data, active, param_grid)

    # 只在权重和为1的组合中筛选
    df = df[df.apply(lambda r: abs(r['momentum_weight'] + r.get('volatility_weight', 0) - 1.0) < 0.01, axis=1)]

    print(f'\n[Results] {len(df)} valid results')
    print(f'\nTop 10:')
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        cfg = EnhancedConfig(**{k: row[k] for k in EnhancedConfig.__dataclass_fields__
                                 if k in row.index})
        print(f'  #{i+1} Score={row["score"]:.4f} | CAGR={row["cagr"]:.1%} '
              f'Sharpe={row["sharpe"]:.2f} MaxDD={row["max_dd"]:.1%} | {cfg}')

    # 保存报告
    report = analyze_results(df)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n[Report] {REPORT_FILE}')

    # 打印关键发现
    print('\n' + '=' * 60)
    print('  关键发现')
    print('=' * 60)

    # 各参数最优值
    for param, label in [('momentum_weight', '动量权重'), ('mom_bars', '动量周期'),
                          ('top_n', '持仓数'), ('min_momentum', '动量过滤'),
                          ('market_ma', 'MA闸门')]:
        if param in df.columns and df[param].nunique() > 1:
            grouped = df.groupby(param)['score'].mean()
            best = grouped.idxmax()
            print(f'  {label}: 最优={best} (score={grouped.max():.4f})')

    return df


if __name__ == '__main__':
    main()
