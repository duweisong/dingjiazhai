"""
国家队仓位信号 — 一键数据管线
==============================

整合现有基础设施，产出策略可消费的预测仓位信号。

链路:
  1. national_team_tracker.py  →  拉取季度社保持仓 (akshare)
  2. position_engine.py        →  计算实际信心指数 → conviction.json
  3. predictive_engine.py      →  K线拟合 → 预测仓位 → predicted_position.json
  4. 本脚本                    →  校验 → 输出 → 供回测/实盘消费

用法:
  python fetch_nt_data.py              # 使用缓存，输出当前预测仓位
  python fetch_nt_data.py --update     # 强制拉取最新季度数据
  python fetch_nt_data.py --json       # JSON 输出（供程序消费）
  python fetch_nt_data.py --validate   # 仅校验数据完整性
"""

import sys
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import os, json, argparse
from pathlib import Path
from datetime import datetime, timezone

# 路径解析
PROJECT_ROOT = Path(__file__).parent.parent.parent
NT_CACHE = PROJECT_ROOT / '.cache' / 'national_team'
PREDICTED_FILE = NT_CACHE / 'predicted_position.json'
CONVICTION_FILE = NT_CACHE / 'conviction.json'

# 将 nt-position-sizer/src 加入 sys.path（复用预测引擎）
sys.path.insert(0, str(PROJECT_ROOT / 'nt-position-sizer' / 'src'))


def check_prerequisites() -> dict:
    """检查前置条件，返回状态字典。"""
    status = {
        'cache_dir_exists': NT_CACHE.exists(),
        'has_holdings': False,
        'holdings_count': 0,
        'has_conviction': CONVICTION_FILE.exists(),
        'has_predicted': PREDICTED_FILE.exists(),
    }

    if status['cache_dir_exists']:
        holdings = sorted(NT_CACHE.glob('holdings_*.parquet'))
        status['has_holdings'] = len(holdings) > 0
        status['holdings_count'] = len(holdings)

    return status


def validate_predicted(data: dict) -> list[str]:
    """校验 predicted_position.json 数据完整性。返回问题列表。"""
    issues = []

    # 必需字段
    required = ['date', 'predicted_conviction', 'actual_conviction',
                'predicted_position', 'actual_position', 'signal']
    for field in required:
        if field not in data:
            issues.append(f'缺少字段: {field}')

    if issues:
        return issues

    # 值域校验
    pred_pos = data['predicted_position']
    if not (0.01 <= pred_pos <= 0.99):
        issues.append(f'predicted_position 超出范围: {pred_pos}')

    pred_conv = data['predicted_conviction']
    if not (-1.0 <= pred_conv <= 1.0):
        issues.append(f'predicted_conviction 超出范围: {pred_conv}')

    # 时效性
    try:
        data_date = datetime.fromisoformat(data['date'])
        age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - data_date).days
        if age_days > 7:
            issues.append(f'数据已过期 ({age_days} 天前)，建议 --update')
    except (ValueError, KeyError):
        issues.append('date 字段格式无效')

    # signal 值
    if data['signal'] not in ('BUY', 'SELL', 'HOLD'):
        issues.append(f'未知 signal 值: {data["signal"]}')

    return issues


def run_pipeline(force_update: bool = False) -> dict:
    """运行完整数据管线，返回结果。"""
    import subprocess

    results = {'steps': {}, 'success': True}

    # Step 1: 拉取季度数据（仅在 --update 时）
    if force_update:
        tracker = PROJECT_ROOT / 'national_team_tracker.py'
        if tracker.exists():
            print('[1/3] 拉取国家队季度持仓数据...')
            r = subprocess.run([sys.executable, str(tracker)],
                               capture_output=True, text=True, timeout=600,
                               cwd=str(PROJECT_ROOT))
            results['steps']['tracker'] = {
                'ok': r.returncode == 0,
                'output': r.stdout[-200:] if r.stdout else '',
                'error': r.stderr[-200:] if r.stderr else '',
            }
        else:
            print('[1/3] [WARN]national_team_tracker.py 未找到，跳过拉取')
            results['steps']['tracker'] = {'ok': False, 'error': 'file not found'}
    else:
        print('[1/3] 使用缓存数据 (--update 可强制刷新)')
        results['steps']['tracker'] = {'ok': True, 'cached': True}

    # Step 2: 计算实际仓位
    print('[2/3] 计算实际仓位指数...')
    try:
        from position_engine import load_conviction_history, calculate_position
        df = load_conviction_history()
        if df.empty:
            results['steps']['position'] = {'ok': False, 'error': '季度数据为空'}
        else:
            pos = calculate_position(df)
            results['steps']['position'] = {
                'ok': True,
                'quarters': len(df),
                'actual_position': pos['position'],
                'latest_quarter': pos['latest_quarter'],
                'conviction': pos['conviction'],
            }
    except Exception as e:
        results['steps']['position'] = {'ok': False, 'error': str(e)}

    # Step 3: 预测仓位
    print('[3/3] 运行 K线拟合预测引擎...')
    try:
        from predictive_engine import (
            load_market_data, load_nt_conviction, compute_market_features,
            align_quarterly, train_model, predict_conviction, conviction_to_position,
            FEATURE_COLS,
        )
        market = load_market_data()
        features = compute_market_features(market)
        conviction = load_nt_conviction()
        aligned = align_quarterly(features, conviction)
        model, scaler, importance = train_model(aligned)

        if model is None:
            results['steps']['predictive'] = {'ok': False, 'error': '训练数据不足'}
        else:
            latest_features = features.iloc[-1][FEATURE_COLS].to_dict()
            predicted_conv = predict_conviction(model, scaler, latest_features)
            pred_pos = conviction_to_position(predicted_conv)

            # 保存供外部消费
            NT_CACHE.mkdir(parents=True, exist_ok=True)
            top3 = {str(k): round(float(v), 4)
                    for k, v in list(importance.items())[:3]} if importance is not None else {}
            output = {
                'date': datetime.now().isoformat(),
                'predicted_conviction': round(predicted_conv, 4),
                'actual_conviction': results['steps']['position'].get('conviction', 0),
                'predicted_position': round(pred_pos, 3),
                'actual_position': round(
                    results['steps']['position'].get('actual_position', 0.5), 3),
                'signal': 'BUY' if pred_pos > 0.60 else ('SELL' if pred_pos < 0.40 else 'HOLD'),
                'top_factors': top3,
            }
            with open(PREDICTED_FILE, 'w') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            results['steps']['predictive'] = {
                'ok': True,
                'predicted_conviction': round(predicted_conv, 4),
                'predicted_position': round(pred_pos, 3),
                'top_factors': top3,
            }
    except Exception as e:
        results['steps']['predictive'] = {'ok': False, 'error': str(e)}

    # 汇总
    results['success'] = all(s.get('ok', False) for s in results['steps'].values())
    return results


def main():
    parser = argparse.ArgumentParser(description='国家队仓位信号 — 一键数据管线')
    parser.add_argument('--update', action='store_true', help='强制拉取最新季度数据')
    parser.add_argument('--json', action='store_true', help='JSON 输出（供程序消费）')
    parser.add_argument('--validate', action='store_true', help='仅校验数据完整性')
    args = parser.parse_args()

    if args.validate:
        status = check_prerequisites()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print('=== 数据完整性校验 ===')
            for k, v in status.items():
                mark = '[OK]' if v else ('[FAIL]' if isinstance(v, bool) else '[WARN]')
                print(f'  {mark} {k}: {v}')

        if PREDICTED_FILE.exists():
            data = json.loads(PREDICTED_FILE.read_text())
            issues = validate_predicted(data)
            if issues:
                print(f'\n❌ predicted_position.json 校验失败 ({len(issues)} 个问题):')
                for issue in issues:
                    print(f'  - {issue}')
            else:
                print('\n✅ predicted_position.json 校验通过')
        return 0 if PREDICTED_FILE.exists() else 1

    # 前置检查
    status = check_prerequisites()
    if not args.update and not status['has_holdings']:
        print('❌ 无缓存数据，请先运行: python fetch_nt_data.py --update')
        return 1

    # 运行管线
    results = run_pipeline(force_update=args.update)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if results['success'] else 1

    # 终端报告
    print()
    print('=' * 60)
    print('  国家队仓位信号管线 — 运行报告')
    print('=' * 60)

    for step_name, step in results['steps'].items():
        mark = '[OK]' if step.get('ok') else '[FAIL]'
        label = {'tracker': '季度数据拉取', 'position': '实际仓位计算',
                 'predictive': 'K线预测引擎'}.get(step_name, step_name)
        print(f'  {mark} {label}')
        if step_name == 'predictive' and step.get('ok'):
            print(f'     预测仓位: {step["predicted_position"]:.0%}')
            print(f'     预测信心: {step["predicted_conviction"]:+.4f}')
            if step.get('top_factors'):
                print(f'     Top 因子: {list(step["top_factors"].keys())[:3]}')
        if step_name == 'position' and step.get('ok'):
            print(f'     实际仓位: {step["actual_position"]:.0%} '
                  f'(基于 {step["quarters"]} 个季度, 最新: {step["latest_quarter"]})')
        if step.get('error'):
            print(f'     错误: {step["error"]}')

    print('=' * 60)

    if results['success'] and PREDICTED_FILE.exists():
        data = json.loads(PREDICTED_FILE.read_text())
        issues = validate_predicted(data)
        if not issues:
            print(f'  ✅ 预测仓位已就绪: {PREDICTED_FILE}')
            print(f'  [DATA] 当前信号: {data["signal"]} '
                  f'(预测 {data["predicted_position"]:.0%} vs 实际滞后 {data["actual_position"]:.0%})')
        else:
            print(f'  [WARN]数据校验问题: {issues}')

    print('=' * 60)

    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
