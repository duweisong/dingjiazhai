#!/usr/bin/env python3
"""沪深300 批量回测（独立运行版）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ao_kc_strategy import *

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=100000)
    p.add_argument("--risk", type=float, default=0.02)
    p.add_argument("--tdx-path", default=r"C:\zd_pazq_hy")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()

    print("Running CSI 300 batch backtest...")
    rdf = batch_backtest_hs300(args.tdx_path, args.capital, args.risk)
    print_batch_report(rdf)

    if args.export and not rdf.empty:
        rdf.to_csv("HS300_batch_results.csv", index=False, encoding="utf-8-sig")
        print("Saved: HS300_batch_results.csv")
