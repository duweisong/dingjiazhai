# 计划（plan）—— 契约

> 经确认后执行。要偏离，**先改这里**再动手。

## 里程碑
- [ ] M1：策略设计定稿 — DESIGN.md 完成（因子定义、选股规则、仓位管理、信号逻辑）
- [ ] M2：数据管线 — 国家队持仓数据 + ETF 行情数据获取脚本
- [ ] M3：核心策略回测 — 完整回测脚本，输出绩效报告
- [ ] M4：参数优化 — 敏感性分析、最优参数搜索
- [ ] M5：实盘信号 — 月度信号生成 + PushPlus 推送

## 任务拆解
| 任务 | 负责角色/工具 | 输入 | 产出（落哪个文件） | 验收标准 |
|------|-------------|------|-------------------|---------|
| T01 | 策略设计 | charter.md | `DESIGN.md`、`docs/specs/strategy-design.md` | 因子公式、选股算法、仓位规则完整且无歧义 |
| T02 | 数据管线-国家队 | efinance/akshare | `scripts/fetch_nt_holdings.py` | 能拉取汇金/证金/社保持仓变化数据 |
| T03 | 数据管线-ETF行情 | efinance/akshare | `scripts/fetch_etf_data.py` | 能拉取 ≥30 只 ETF 日线数据 |
| T04 | 回测引擎 | DESIGN.md + T02+T03 产出 | `scripts/backtest_nt_rotation.py` | 输出年化/夏普/回撤/月胜率，与 V4.2 可比 |
| T05 | 参数优化 | T04 产出 | `scripts/optimize_params.py` | 网格搜索最优排名区间和权重 |
| T06 | 实盘信号 | T05 最优参数 | `scripts/live_signal.py` | 月末运行输出下月持仓，PushPlus 推送成功 |

## 实时进展/交接棒
→ 见 `flow/进展.md` 顶部（每棒收工在那追加一条：做了什么/为什么/产出路径/下一步）。
（plan.md 只管"计划=契约"；"现在到哪了"在进展日志，不在这儿覆盖。）
