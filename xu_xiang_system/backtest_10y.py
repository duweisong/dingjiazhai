"""
Ten-Year Backtest (2015-2025) - Xu Xiang Upgraded System

Simulates Xu Xiang style across all major market regimes:
  2015: Leverage bull + crash
  2016: Meltdown recovery
  2017: White-horse bull (strategy failure zone)
  2018: Full-year bear
  2019-2020: Tech bull
  2021-2022: Quant emergence
  2023-2025: Theme rotation + quant dominance
"""

import numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from .data_loader import get_stock_data, get_index_data
from .strategy import XuXiangStrategy
from .market_env import MarketEnvClassifier, build_index_env_df


@dataclass
class YearlyPerformance:
    year: int = 0
    total_return_pct: float = 0
    benchmark_return_pct: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    total_trades: int = 0
    win_rate: float = 0
    avg_holding_days: float = 0
    profit_factor: float = 0
    env_A_days: int = 0
    env_B_days: int = 0
    env_C_days: int = 0


@dataclass
class BacktestReport:
    symbol: str = ""
    name: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0
    final_equity: float = 0
    total_return_pct: float = 0
    annualized_return_pct: float = 0
    benchmark_return_pct: float = 0
    max_drawdown_pct: float = 0
    max_drawdown_duration_days: int = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    total_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_win_pct: float = 0
    avg_loss_pct: float = 0
    avg_holding_days: float = 0
    yearly: List = field(default_factory=list)
    env_stats: Dict = field(default_factory=dict)
    equity_curve: object = None
    trades: object = None


class TenYearBacktest:
    """10-year backtest engine for Xu Xiang upgraded strategy"""

    def __init__(self, initial_capital=1000000, commission=0.00025,
                 stamp_tax=0.001, slippage=0.001, t_plus_one=True):
        self.ic = initial_capital
        self.cr = commission
        self.st = stamp_tax
        self.sl = slippage
        self.t1 = t_plus_one
        self.ec = MarketEnvClassifier()

    def run(self, symbol, name="", start="2015-01-01", end="2025-12-31",
            sp=None):
        print(f"\n{'='*70}")
        print(f"  Xu Xiang Upgraded 10-Year Backtest")
        print(f"  Stock: {name}({symbol})  |  {start} ~ {end}")
        print(f"{'='*70}")
        print("\n[1/5] Loading data...")
        df = get_stock_data(symbol, name, start, end)
        idx = get_index_data("000001", start, end)
        print(f"  Stock: {len(df)}d, Index: {len(idx)}d")
        print("\n[2/5] Classifying environments...")
        env_raw = build_index_env_df(idx)
        cd = set(df["date"].dt.date) & set(env_raw["date"].dt.date)
        env_raw = env_raw[env_raw["date"].dt.date.isin(cd)]
        df = df[df["date"].dt.date.isin(cd)]
        env = self.ec.classify_series(env_raw)
        a_d = int((env["env_level"] == "A").sum())
        b_d = int((env["env_level"] == "B").sum())
        c_d = int((env["env_level"] == "C").sum())
        print(f"  A:{a_d}d  B:{b_d}d  C:{c_d}d")
        print("\n[3/5] Generating signals...")
        s = XuXiangStrategy(sp or {})
        sig = s.generate_signals(df)
        bu = int((sig == 1).sum())
        se = int((sig == -1).sum())
        print(f"  Signals: {bu} buys, {se} sells")
        print("\n[4/5] Running simulation...")
        res = self._simulate(df, sig, env)
        eq_final = res["equity_curve"]["equity"].iloc[-1]
        print(f"  Final equity: {eq_final:,.0f}")
        print(f"  Trades: {len(res['trades'])}")
        print("\n[5/5] Building report...")
        rpt = self._build_report(symbol, name, start, end, df, res, env)
        return rpt

    def _simulate(self, df, sig, env):
        cash = self.ic
        pos = 0
        ep = 0.0
        ed = None
        cs = False
        lbd = None
        tl = []
        er = []
        em = dict(zip(env["date"], env["env_level"]))
        for i in range(len(df)):
            date = df["date"].iloc[i]
            cp = df["close"].iloc[i]
            si = sig.iloc[i]
            el = em.get(date, "B")
            if self.t1 and pos > 0 and lbd is not None:
                if date > lbd:
                    cs = True
            if pos > 0 and cs:
                stg = False
                sp = 0.0
                sr = ""
                if si == -1:
                    stg = True
                    sp = cp * (1 - self.sl)
                    sr = "signal"
                if el == "C":
                    stg = True
                    sp = cp * (1 - self.sl)
                    sr = "env_C"
                if stg:
                    pro = sp * pos
                    cm = max(pro * self.cr, 5)
                    smp = pro * self.st
                    cash += pro - cm - smp
                    pnl = (sp - ep) * pos - cm - smp
                    pp = (sp - ep) / ep
                    hd = (date - ed).days if ed else 0
                    tl.append(dict(
                        entry_date=ed, exit_date=date,
                        entry_price=ep, exit_price=sp,
                        shares=pos, pnl=pnl, pnl_pct=pp,
                        reason=sr, holding_days=max(1, hd),
                        env_at_entry=em.get(ed, "?"),
                        env_at_exit=el))
                    pos = 0
                    cs = False
            if si == 1 and pos == 0 and el != "C":
                bp = cp * (1 + self.sl)
                bdgt = cash * 0.8
                rs = int(bdgt / bp)
                sh = (rs // 100) * 100
                if sh >= 100:
                    cst = bp * sh
                    cm = max(cst * self.cr, 5)
                    tc = cst + cm
                    if tc <= cash:
                        cash -= tc
                        pos = sh
                        ep = bp
                        ed = date
                        lbd = date
                        cs = not self.t1
            mv = pos * cp if pos > 0 else 0
            eq = cash + mv
            er.append(dict(date=date, equity=eq, cash=cash,
                           position=pos, market_value=mv,
                           close=cp, env_level=el))
        if pos > 0:
            lc = df["close"].iloc[-1]
            pro = lc * pos
            cm = max(pro * self.cr, 5)
            smp = pro * self.st
            cash += pro - cm - smp
            pnl = (lc - ep) * pos - cm - smp
            pp = (lc - ep) / ep
            hd = (df["date"].iloc[-1] - ed).days
            tl.append(dict(
                entry_date=ed, exit_date=df["date"].iloc[-1],
                entry_price=ep, exit_price=lc,
                shares=pos, pnl=pnl, pnl_pct=pp,
                reason="force_close", holding_days=max(1, hd),
                env_at_entry=em.get(ed, "?"),
                env_at_exit=em.get(df["date"].iloc[-1], "?")))
        eqdf = pd.DataFrame(er)
        eqdf["ret"] = eqdf["equity"].pct_change()
        tdf = pd.DataFrame(tl) if tl else pd.DataFrame()
        wr = (tdf["pnl"] > 0).mean() if len(tdf) > 0 else 0.0
        return dict(equity_curve=eqdf, trades=tdf, win_rate=wr,
                    final_cash=cash)

    def _build_report(self, symbol, name, start, end, df, res, env):
        eq = res["equity_curve"]
        td = res["trades"]
        tr = (eq["equity"].iloc[-1] - self.ic) / self.ic
        tdy = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
        yr = tdy / 365.25
        ar = (1 + tr) ** (1 / yr) - 1 if yr > 0 else 0
        cmax = eq["equity"].cummax()
        dd = (eq["equity"] - cmax) / cmax
        mdd = dd.min()
        mdur = 0
        if pd.notna(dd.idxmin()):
            ds = cmax[:dd.idxmin()].idxmax() if dd.idxmin() > 0 else 0
            de = dd.idxmin()
            mdur = (eq["date"].iloc[de] - eq["date"].iloc[ds]).days
        rf = 0.03 / 252
        ex = eq["ret"].dropna() - rf
        sh = np.sqrt(252) * ex.mean() / ex.std() if ex.std() > 0 else 0
        dn = ex[ex < 0]
        so = np.sqrt(252) * ex.mean() / dn.std() if len(dn) > 0 and dn.std() > 0 else 0
        ca = ar / abs(mdd) if mdd != 0 else 0
        if len(td) > 0:
            wr = (td["pnl"] > 0).mean()
            ws = td[td["pnl"] > 0]
            ls = td[td["pnl"] <= 0]
            pf = ws["pnl"].sum() / abs(ls["pnl"].sum()) if len(ls) > 0 and ls["pnl"].sum() != 0 else float("inf")
            aw = ws["pnl_pct"].mean() if len(ws) > 0 else 0
            al = ls["pnl_pct"].mean() if len(ls) > 0 else 0
            ah = td["holding_days"].mean()
        else:
            wr = pf = aw = al = ah = 0
        br = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
        yl = self._yearly(eq, td, env, df)
        es = self.ec.get_env_stats(env)
        return BacktestReport(
            symbol=symbol, name=name, start_date=start, end_date=end,
            initial_capital=self.ic, final_equity=eq["equity"].iloc[-1],
            total_return_pct=tr * 100, annualized_return_pct=ar * 100,
            benchmark_return_pct=br * 100, max_drawdown_pct=mdd * 100,
            max_drawdown_duration_days=int(mdur),
            sharpe_ratio=sh, sortino_ratio=so, calmar_ratio=ca,
            total_trades=len(td), win_rate=wr * 100,
            profit_factor=pf,
            avg_win_pct=aw * 100 if aw else 0,
            avg_loss_pct=al * 100 if al else 0,
            avg_holding_days=ah,
            yearly=yl, env_stats=es,
            equity_curve=eq, trades=td)

    def _yearly(self, eq, td, env, pr):
        yl = []
        eqc = eq.copy()
        eqc["year"] = pd.to_datetime(eqc["date"]).dt.year
        for yr in sorted(eqc["year"].unique()):
            ye = eqc[eqc["year"] == yr]
            if len(ye) < 2:
                continue
            se = ye["equity"].iloc[0]
            ee = ye["equity"].iloc[-1]
            yt = (ee - se) / se
            cmax_y = ye["equity"].cummax()
            dy = (ye["equity"] - cmax_y) / cmax_y
            mdy = dy.min()
            rf = 0.03 / 252
            ex = ye["ret"].dropna() - rf
            sy = np.sqrt(252) * ex.mean() / ex.std() if ex.std() > 0 else 0
            ytd = td[pd.to_datetime(td["entry_date"]).dt.year == yr] if len(td) > 0 else pd.DataFrame()
            ye2 = env[pd.to_datetime(env["date"]).dt.year == yr]
            ea = int((ye2["env_level"] == "A").sum())
            eb = int((ye2["env_level"] == "B").sum())
            ec2 = int((ye2["env_level"] == "C").sum())
            yp = pr[pd.to_datetime(pr["date"]).dt.year == yr]
            by = 0.0
            if len(yp) >= 2:
                by = (yp["close"].iloc[-1] - yp["close"].iloc[0]) / yp["close"].iloc[0]
            yl.append(YearlyPerformance(
                year=int(yr), total_return_pct=yt * 100,
                benchmark_return_pct=by * 100,
                max_drawdown_pct=mdy * 100,
                sharpe_ratio=sy, total_trades=len(ytd),
                win_rate=(ytd["pnl"] > 0).mean() * 100 if len(ytd) > 0 else 0,
                avg_holding_days=ytd["holding_days"].mean() if len(ytd) > 0 else 0,
                env_A_days=ea, env_B_days=eb, env_C_days=ec2))
        return yl

    def print_report(self, rpt):
        print()
        print("=" * 80)
        print(f"  BACKTEST: {rpt.name}({rpt.symbol})  {rpt.start_date} ~ {rpt.end_date}")
        print("=" * 80)
        print(f"  Return: {rpt.total_return_pct:+.2f}%  Annual: {rpt.annualized_return_pct:+.2f}%")
        print(f"  Bench: {rpt.benchmark_return_pct:+.2f}%  MaxDD: {rpt.max_drawdown_pct:.2f}%")
        print(f"  Sharpe: {rpt.sharpe_ratio:.2f}  Calmar: {rpt.calmar_ratio:.2f}")
        print(f"  Trades: {rpt.total_trades}  WinRate: {rpt.win_rate:.1f}%  PF: {rpt.profit_factor:.2f}")
        print("  Yearly:")
        for y in rpt.yearly:
            print(f"    {y.year}: {y.total_return_pct:+.1f}% (bench {y.benchmark_return_pct:+.1f}%) DD={y.max_drawdown_pct:.1f}% env={y.env_A_days}/{y.env_B_days}/{y.env_C_days}")
        if len(rpt.trades) > 0:
            print("  Env-Conditional:")
            for ev in ["A", "B", "C"]:
                et = rpt.trades[rpt.trades["env_at_entry"] == ev]
                if len(et) > 0:
                    ap = et["pnl_pct"].mean() * 100
                    wr = (et["pnl"] > 0).mean() * 100
                    print(f"    {ev}: {len(et)} trades, PnL={ap:+.1f}%, win={wr:.0f}%")


def run_backtest(symbol="002036", name="LianChuang",
                 start="2015-01-01", end="2025-12-31"):
    """Convenience function to run a backtest"""
    bt = TenYearBacktest(initial_capital=1000000)
    rpt = bt.run(symbol, name, start, end)
    bt.print_report(rpt)
    return rpt


if __name__ == "__main__":
    run_backtest()