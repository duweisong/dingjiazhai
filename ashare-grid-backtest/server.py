"""
A股网格交易回测 — Web 可视化界面服务器

Flask 后端，封装 grid_engine 为 REST API，
浏览器打开 http://localhost:5000 即可使用。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# 确保项目根在路径
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, request, send_from_directory

from src.config import GridConfig, STOCK_POOL, ETF_POOL
from src.data_loader import get_stock_data
from src.grid_engine import GridEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ============================================================
# 静态文件
# ============================================================
@app.route("/")
def index():
    return send_from_directory("design", "Grid Backtest UI.html")


# ============================================================
# API: 标的池
# ============================================================
@app.route("/api/stocks")
def api_stocks():
    """返回标的池列表"""
    stocks = []
    for code, market, name in STOCK_POOL + ETF_POOL:
        stocks.append({"code": code, "market": market, "name": name})
    return jsonify(stocks)


# ============================================================
# API: 单次回测
# ============================================================
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """
    执行单次网格回测。

    Body (JSON):
        symbol, market, start_date, end_date,
        grid_period, grid_levels, grid_step,
        use_ma_gate, ma_gate_period
    """
    try:
        params = request.get_json() or {}

        # 构建配置
        symbol = params.get("symbol", "002036")
        market = params.get("market", "SZ")
        start_date = params.get("start_date", "2020-01-01")
        end_date = params.get("end_date", "2025-12-31")
        grid_period = int(params.get("grid_period", 240))
        grid_levels = int(params.get("grid_levels", 11))
        grid_step = float(params.get("grid_step", 0.005))
        use_ma_gate = bool(params.get("use_ma_gate", False))
        ma_gate_period = int(params.get("ma_gate_period", 60))

        half = (grid_levels - 1) / 2
        grid_top_pct = half * grid_step
        grid_bot_pct = -half * grid_step

        config = GridConfig(
            symbol=symbol, market=market,
            start_date=start_date, end_date=end_date,
            grid_period=grid_period, grid_levels=grid_levels,
            grid_step=grid_step,
            grid_top_pct=grid_top_pct, grid_bot_pct=grid_bot_pct,
            use_ma_gate=use_ma_gate, ma_gate_period=ma_gate_period,
        )

        # 加载数据
        logger.info("加载 %s.%s 数据...", symbol, market)
        df = get_stock_data(symbol, market, start_date, end_date)

        # 执行回测
        logger.info("运行回测: 周期=%d 档位=%d 步长=%.3f", grid_period, grid_levels, grid_step)
        engine = GridEngine()
        result = engine.run(df, config)

        # 构建权益曲线数据
        equity = result.equity_curve
        equity_data = []
        for _, row in equity.iterrows():
            equity_data.append({
                "date": str(row["date"]),
                "total_value": float(row["total_value"]),
            })

        # 构建交易日志
        trades_data = []
        for t in result.trades[-20:]:  # 最近 20 笔
            trades_data.append({
                "date": str(t.date.date()),
                "action": t.action,
                "price": t.price,
                "shares": t.shares,
                "reason": "网格触发" if "INIT" in t.action or "BUY" in t.action
                          else ("止盈" if t.price > 0 else "平仓"),
            })

        return jsonify({
            "success": True,
            "result": {
                "total_return_pct": result.total_return_pct,
                "annualized_return_pct": result.annualized_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "calmar_ratio": result.calmar_ratio,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_trades": result.profit_trades,
                "final_value": result.final_value,
                "final_cash": result.final_cash,
                "final_shares": result.final_shares,
                "benchmark_return_pct": result.benchmark_return_pct,
                "equity": equity_data,
                "trades": trades_data,
            },
        })

    except Exception as e:
        logger.exception("回测失败")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# API: 参数扫描
# ============================================================
@app.route("/api/scan", methods=["POST"])
def api_scan():
    """参数扫描"""
    try:
        params = request.get_json() or {}

        symbol = params.get("symbol", "002036")
        market = params.get("market", "SZ")
        start_date = params.get("start_date", "2020-01-01")
        end_date = params.get("end_date", "2025-12-31")
        use_ma_gate = bool(params.get("use_ma_gate", False))

        # 参数空间
        grid_steps = [float(s) for s in params.get("grid_steps", [0.005, 0.01, 0.015, 0.02, 0.03])]
        grid_periods = [int(p) for p in params.get("grid_periods", [120, 240, 480])]
        grid_levels_list = [int(l) for l in params.get("grid_levels_list", [7, 11, 15])]

        logger.info("加载 %s.%s 数据用于扫描...", symbol, market)
        df = get_stock_data(symbol, market, start_date, end_date)

        results = []
        total = len(grid_steps) * len(grid_periods) * len(grid_levels_list)
        count = 0

        for step in grid_steps:
            for period in grid_periods:
                for levels in grid_levels_list:
                    count += 1
                    half = (levels - 1) / 2

                    config = GridConfig(
                        symbol=symbol, market=market,
                        start_date=start_date, end_date=end_date,
                        grid_period=period, grid_levels=levels,
                        grid_step=step,
                        grid_top_pct=half * step, grid_bot_pct=-half * step,
                        use_ma_gate=use_ma_gate,
                    )

                    logger.info("[%d/%d] 步长=%.3f 周期=%d 档位=%d", count, total, step, period, levels)

                    try:
                        engine = GridEngine()
                        result = engine.run(df.copy(), config)
                        results.append({
                            "grid_step": step,
                            "grid_period": period,
                            "grid_levels": levels,
                            "annualized_return_pct": result.annualized_return_pct,
                            "sharpe_ratio": result.sharpe_ratio,
                            "max_drawdown_pct": result.max_drawdown_pct,
                            "win_rate": result.win_rate,
                            "total_return_pct": result.total_return_pct,
                        })
                    except Exception as e:
                        logger.warning("组合失败: %s", e)

        results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

        return jsonify({"success": True, "results": results, "total": len(results)})

    except Exception as e:
        logger.exception("参数扫描失败")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# API: 标的池扫描
# ============================================================
@app.route("/api/pool", methods=["POST"])
def api_pool():
    """标的池扫描"""
    try:
        params = request.get_json() or {}

        start_date = params.get("start_date", "2020-01-01")
        end_date = params.get("end_date", "2025-12-31")
        grid_period = int(params.get("grid_period", 240))
        grid_levels = int(params.get("grid_levels", 11))
        grid_step = float(params.get("grid_step", 0.005))
        use_ma_gate = bool(params.get("use_ma_gate", False))
        pool_type = params.get("pool_type", "stock")  # "stock" | "etf"

        pool = STOCK_POOL if pool_type == "stock" else ETF_POOL
        half = (grid_levels - 1) / 2

        results = []
        for code, mkt, name in pool:
            logger.info("扫描 %s (%s.%s)...", name, code, mkt)
            try:
                df = get_stock_data(code, mkt, start_date, end_date)
                config = GridConfig(
                    symbol=code, market=mkt,
                    start_date=start_date, end_date=end_date,
                    grid_period=grid_period, grid_levels=grid_levels,
                    grid_step=grid_step,
                    grid_top_pct=half * grid_step,
                    grid_bot_pct=-half * grid_step,
                    use_ma_gate=use_ma_gate,
                )

                engine = GridEngine()
                result = engine.run(df, config)
                results.append({
                    "symbol": code,
                    "name": name,
                    "market": mkt,
                    "annualized_return_pct": result.annualized_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "total_return_pct": result.total_return_pct,
                    "final_value": result.final_value,
                })
            except Exception as e:
                logger.warning("%s 跳过: %s", name, e)

        results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

        return jsonify({"success": True, "results": results, "total": len(results)})

    except Exception as e:
        logger.exception("标的池扫描失败")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8520
    print("\n" + "=" * 50)
    print(f"  A股网格交易回测 · Web 界面")
    print(f"  浏览器打开 → http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
