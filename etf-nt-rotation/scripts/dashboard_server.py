"""
ETF 国家队轮动策略 — API 服务器
================================

为 dashboard.html 提供数据接口。启动后打开 dashboard.html 即可使用。

用法: python scripts/dashboard_server.py
端口: http://localhost:8765
"""
import sys, os, json, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'etf-nt-rotation' / 'scripts'))

from flask import Flask, jsonify, request

app = Flask(__name__)

# CORS support for standalone HTML
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    return response


@app.route('/')
def index():
    return jsonify({'status': 'ok', 'message': 'ETF NT Rotation API Server', 'docs': 'Open dashboard.html in browser'})


@app.route('/api/ping')
def api_ping():
    return jsonify({'status': 'ok', 'time': str(__import__('datetime').datetime.now())})


@app.route('/api/backtest')
def api_backtest():
    try:
        from backtest_nt_rotation import load_data, run_backtest, Config
        cfg = Config(momentum_weight=0.5, volatility_weight=0.5, mom_bars=14,
                     rank_min=1, rank_max=10, top_n=5, trade_cost=0.001)
        data, active = load_data(cfg)
        result = run_backtest(data, active, cfg, 1.0)
        m = result['metrics']
        annual = []
        if result['monthly_returns'] is not None and len(result['monthly_returns']) > 0:
            for yr, grp in result['monthly_returns'].groupby(result['monthly_returns'].index.year):
                annual.append({'year': int(yr), 'ret': float((1+grp).prod()-1)})
        nav = result['nav']
        bench = result.get('bench_nav')
        step = max(1, len(nav)//200)
        return jsonify({
            'config': str(cfg), 'months': result['schedule_months'], 'trades': m['total_trades'],
            'metrics': {'cagr':m['cagr'],'sharpe':m['sharpe'],'max_drawdown':m['max_drawdown'],
                        'calmar':m['calmar'],'win_rate_monthly':m['win_rate_monthly'],
                        'total_return':m['total_return']},
            'nav_dates': [str(d.date()) for d in nav.index[::step]],
            'nav_values': [float(x) for x in nav.values[::step]],
            'bench_values': [float(x) for x in bench.values[::step]] if bench is not None else [],
            'annual': annual,
        })
    except Exception as e:
        return jsonify({'error': f'{e}\n{traceback.format_exc()}'})


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    try:
        import pandas as pd
        from backtest_nt_rotation import load_data, Config
        from optimize_params import run_single_backtest, EnhancedConfig

        cfg_base = Config(momentum_weight=0.5, volatility_weight=0.5, mom_bars=14,
                          rank_min=1, rank_max=10, top_n=5)
        data, active = load_data(cfg_base)
        results = []
        for w, vb in [(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2)]:
            for mom in [14, 21, 33]:
                for n in [5, 8]:
                    for rmax in [5, 10]:
                        if rmax < n: continue
                        cfg = EnhancedConfig(momentum_weight=w, volatility_weight=vb,
                                            mom_bars=mom, top_n=n, rank_max=rmax)
                        m = run_single_backtest(data, active, cfg)
                        if m['score'] > -99:
                            results.append({'config':f'w={w}/{vb} mom={mom}d n={n} rmax={rmax}',
                                            **{k:v for k,v in m.items()}})

        df = pd.DataFrame(results).sort_values('score', ascending=False)
        return jsonify({'results': df.head(15).to_dict('records')})
    except Exception as e:
        return jsonify({'error': f'{e}\n{traceback.format_exc()}'})


@app.route('/api/live-signal')
def api_live_signal():
    try:
        from live_signal import generate_signal
        return jsonify(generate_signal())
    except Exception as e:
        return jsonify({'error': f'{e}\n{traceback.format_exc()}'})


@app.route('/api/etf-data')
def api_etf_data():
    try:
        refresh = request.args.get('refresh') == '1'
        data_file = PROJECT_ROOT / '.cache' / 'etf_sector' / 'nt_rotation_35.parquet'
        if refresh:
            import subprocess
            subprocess.run([sys.executable,
                str(PROJECT_ROOT/'etf-nt-rotation'/'scripts'/'fetch_etf_data.py'),'--force'],
                capture_output=True, timeout=300, cwd=str(PROJECT_ROOT))
        if data_file.exists():
            import pandas as pd
            data = pd.read_parquet(data_file)
            all_codes = set(data.columns)
            expected = {'510300','510500','510050','159915','588000','159949','512100','510880',
                       '512880','512800','512660','512670','512690','159736','159996','159995',
                       '512480','512760','159869','512980','515050','516510','159865','512010',
                       '512170','159755','515790','561910','159611','515220','516970','512200',
                       '516950','159766','561330'}
            missing = sorted(expected - all_codes)
            return jsonify({'exists':True,'ok':len(all_codes&expected),'total':len(expected),
                           'days':len(data),'date_range':f'{data.index[0].date()}~{data.index[-1].date()}',
                           'missing':missing})
        return jsonify({'exists':False,'error':'数据文件不存在'})
    except Exception as e:
        return jsonify({'error':str(e)})


@app.route('/api/nt-data')
def api_nt_data():
    try:
        refresh = request.args.get('refresh') == '1'
        pos_file = PROJECT_ROOT / '.cache' / 'national_team' / 'predicted_position.json'
        if refresh:
            import subprocess
            subprocess.run([sys.executable,
                str(PROJECT_ROOT/'nt-position-sizer'/'src'/'predictive_engine.py')],
                capture_output=True, timeout=60, cwd=str(PROJECT_ROOT))
        if not pos_file.exists():
            return jsonify({'exists':False,'error':'预测仓位数据不存在'})
        data = json.loads(pos_file.read_text())
        cache_dir = PROJECT_ROOT / '.cache' / 'national_team'
        holdings = list(cache_dir.glob('holdings_*.parquet')) if cache_dir.exists() else []
        return jsonify({'exists':True,**data,'holdings_quarters':len(holdings),
                       'latest_quarter':f'Q{holdings[-1].stem[-2:]}' if holdings else 'N/A'})
    except Exception as e:
        return jsonify({'error':str(e)})


@app.route('/api/spec')
def api_spec():
    try:
        spec_file = PROJECT_ROOT / 'etf-nt-rotation' / 'docs' / 'specs' / 'strategy-design.md'
        if not spec_file.exists():
            return jsonify({'error':'文档不存在'})
        text = spec_file.read_text(encoding='utf-8')
        return jsonify({
            '策略名称':'ETF 国家队轮动策略','版本':'v3.0','核心逻辑':'沪指K线→15维特征→Ridge→预测仓位 + 月度动量Top10→5只',
            '最终参数':'50/50 权重, 14d 动量, Top10→5只, 等权重, 满仓','全期年化':'22.7% (2022-2026)',
            '夏普比率':'0.85','最大回撤':'-30.9%','ETF候选池':'35只 (8宽基 + 27行业)',
            '调仓频率':'月度 (月初买入, 月末卖出)','仓位信号源':'nt-position-sizer K线预测引擎',
            '文档长度':f'{len(text)} 字符',
        })
    except Exception as e:
        return jsonify({'error':str(e)})


if __name__ == '__main__':
    print()
    print('='*55)
    print('  ETF 国家队轮动策略 · API 服务器')
    print('='*55)
    print()
    print(f'  后端已启动: http://localhost:8765')
    print(f'  前端入口:   打开 dashboard.html')
    print(f'  停止:       按 Ctrl+C')
    print()
    import numpy as np
    app.run(host='0.0.0.0', port=8765, debug=False)
