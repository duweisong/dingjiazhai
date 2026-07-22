# grid-backtest — A股网格交易回测系统

> 多股票/ETF 网格交易回测引擎，支持步长、持仓、区间等参数优化。

## 项目定位

网格交易（Grid Trading）回测系统：在价格区间内设置 N 档买卖挂单，价格触及网格线自动触发交易，利用震荡获利。通过参数扫描找到最优网格条件以最大化收益。

## 目录地图

| 目录 | 用途 |
|------|------|
| `src/` | 回测引擎源码 |
| `tests/` | 单元测试 |
| `flow/` | 项目推进（章程/计划/进展/决策） |
| `docs/` | 知识交付（设计文档/研究报告） |
| `output/` | 生成图表和报告 |

## 核心模块

```
src/
├── config.py          # 全局配置（标的、资金、交易成本、网格参数）
├── data_loader.py     # 数据加载（efinance + akshare 双源）
├── grid_engine.py     # 网格交易核心引擎
├── grid_strategy.py   # 网格策略参数定义
├── optimizer.py       # 参数扫描优化器
├── metrics.py         # 绩效指标计算
├── visualize.py       # 可视化图表
├── reporter.py        # HTML/Excel 报告生成
└── main.py            # CLI 入口
```

## 开工协议

启动新任务时：
1. 读 `flow/进展.md` 了解当前状态
2. 读相关源码，不凭记忆
3. 确认当前分支干净

## 收工协议

每棒结束 → 在 `flow/进展.md` 顶部追加一条交接记录。

## 验证命令

```bash
cd grid-backtest && python -m py_compile src/*.py && python -m pytest tests/ -v
```

## 约束

- Python 3.12+，遵循 PEP 8
- 数据源：efinance（主）+ akshare（备用）
- A 股规则：T+1、万2.5 佣金、千1 印花税
- 不可变数据模式优先（dataclass / NamedTuple）

---

<!-- project-flow-cy:start -->
## 协作约定

- 审稿模型 ≠ 产出模型
- 产出落文件，先 plan 后 act
- 一会话一焦点
- 从根本解决，不打补丁

详规：`flow/规范/`
<!-- project-flow-cy:end -->
