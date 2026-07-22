# ETF 国家队轮动策略 · 完整方案规格

> 产出日期：2026-06-12 | 版本：v2.1 | 状态：待评审
> v2.0 变更：国家队仓位闸门升级为K线拟合预测模型
> v2.1 变更：新增仓位校准层（§6.4）——非对称缩放 + 收益导向优化
> 本文档是 T01 交付物。任一 Python 开发者拿着本文档就能写出回测脚本。

---

## 1. 策略概述

### 1.1 一句话

沪指 K 线 → 15 维市场特征 → Ridge 回归 → **预测**国家队下一季度信心指数 → 连续仓位建议(0-100%)；
月度「涨幅 + 量比」双因子排名 → 取第 5–20 名中段 5 只 ETF → 月初按预测仓位等权买入，月末清仓。

**核心突破**：不用滞后 1-4 个月的季度披露数据做决策，而是用 K 线拟合模型**提前预判**国家队行为。

### 1.2 与 V4.2 行业轮动的差异

| 维度 | V4.2 行业轮动 | 本策略（国家队轮动） |
|------|-------------|-------------------|
| 宏观闸门 | 沪深300 MA60 双条件 | **国家队 K线拟合预测仓位**（日频更新，领先 1 季） |
| 闸门逻辑 | 价格 < MA60 且 MA60 下行 → 空仓 | 预测仓位 0-100% 连续调节（不禁用轮动，只缩仓位） |
| 排名因子 | 2月动量(60%) + 量比(40%) | 1月动量(60%) + 量比(40%) |
| 选股区间 | 排名 6-20 | **排名 5-20，取中段 5 只** |
| 仓位分配 | 得分²加权 | **等权重**（首次迭代） |
| 行业约束 | 同行业 ≤1 | **无行业约束**（纯排名驱动） |
| 候选池 | 45只 × 11行业 | **35只横跨宽基+行业** |

---

## 2. 数据源规格

### 2.1 ETF 行情数据

```python
import efinance as ef

# 获取 ETF 日线数据（市价 K 线）
df = ef.stock.get_quote_history(code, beg='20190101', end='20260612')
# 返回字段：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 涨跌幅, 换手率

# 批量获取
df = ef.stock.get_quote_history(['510300', '510500', ...], beg='20190101')
```

**数据频率**：日频
**回测区间**：2019-01-01 ~ 2026-06-12（覆盖完整牛熊周期）
**缓存策略**：首次拉取存 parquet，后续读缓存（同 V4.2 模式）

### 2.2 国家队仓位信号 —— 预测模型（核心闸门）

> **设计决策**：不直接使用季度披露数据（滞后 1-4 个月，会误导），
> 而是用 K 线市场特征**拟合国家队行为**，产出**日频更新的预测仓位**。

#### 2.2.1 为什么不用季度数据直接做决策

```
时间线示例（Q4 年报）:
  12/31  Q4 期末（国家队实际已完成调仓）
     ↓  3-4 个月真空期（市场已充分反映，你还在等数据）
  4/30   年报披露（此时数据已是 4 个月前的历史）
     ↓
  5月    你据此调仓 → 国家队可能早已反向操作
```

#### 2.2.2 预测模型架构

```
┌─────────────────────────────────────────────────────┐
│                  预测性仓位引擎                       │
│                                                      │
│  沪指/HS300 日线 K 线                                 │
│      ↓                                                │
│  15 维市场特征提取（每日）                             │
│  - 动量: ret_5d, ret_20d, ret_60d                    │
│  - 均线偏离: dist_ma20/60/120                        │
│  - 均线趋势: ma20_slope, ma60_slope                  │
│  - 相对强弱: RSI_14                                   │
│  - 波动率: vol_20d, vol_60d                          │
│  - 回撤深度: hh_60d, hh_120d, ll_60d                 │
│  - 市场广度: up_days_20                               │
│      ↓                                                │
│  季度末特征 ↔ 下季度国家队信心指数（训练标签）         │
│      ↓                                                │
│  Ridge 回归模型（α=0.5，正则化防过拟合）              │
│      ↓                                                │
│  预测下一季度 NT 信心指数                              │
│      ↓                                                │
│  三因子公式 → 连续仓位 0-100%                          │
│  仓位 = sigmoid(3 × (0.50×信心 + 0.30×趋势 + 0.20×加速度)) │
│      ↓                                                │
│  输出 predicted_position.json（日频更新）              │
│      ↓                                                │
│  ETF 轮动策略消费 → 月度调仓时按预测仓位缩放           │
└─────────────────────────────────────────────────────┘
```

#### 2.2.3 15 维市场特征定义

| # | 特征名 | 计算方式 | 含义 |
|---|--------|---------|------|
| 1 | `ret_5d` | close.pct_change(5) | 5日动量 |
| 2 | `ret_20d` | close.pct_change(20) | 20日/约1月动量 |
| 3 | `ret_60d` | close.pct_change(60) | 60日/约1季动量 |
| 4 | `dist_ma20` | (close - MA20) / MA20 | 距20日均线 |
| 5 | `dist_ma60` | (close - MA60) / MA60 | 距60日均线 |
| 6 | `dist_ma120` | (close - MA120) / MA120 | 距半年线 |
| 7 | `ma20_slope` | MA20.pct_change(20) | MA20自身趋势 |
| 8 | `ma60_slope` | MA60.pct_change(20) | MA60自身趋势 |
| 9 | `rsi_14` | 14日RSI | 超买超卖 |
| 10 | `vol_20d` | ret.rolling(20).std() | 短期波动率 |
| 11 | `vol_60d` | ret.rolling(60).std() | 长期波动率 |
| 12 | `hh_60d` | close / high_60d - 1 | 距60日高点 |
| 13 | `hh_120d` | close / high_120d - 1 | 距120日高点 |
| 14 | `ll_60d` | close / low_60d - 1 | 距60日低点 |
| 15 | `up_days_20` | (ret>0).rolling(20).sum()/20 | 上涨天数占比 |

#### 2.2.4 仓位计算公式

```python
def conviction_to_position(conviction: float, trend: float = 0, acceleration: float = 0) -> float:
    """
    三因子 → 仓位（0.02 ~ 0.98）
    
    因子权重:
      - 当前信心指数: 50%  （模型预测的核心输出）
      - 趋势方向:     30%  （连续2季度变化方向）
      - 加速度:       20%  （变化速度，二阶导）
    """
    conv_norm = clip(conviction / 0.4, -1, 1)
    trend_norm = clip(trend / 0.3, -1, 1)
    accel_norm = clip(acceleration / 0.3, -1, 1)
    
    raw = 0.50 * conv_norm + 0.30 * trend_norm + 0.20 * accel_norm
    
    # Sigmoid 平滑映射到 (0, 1)
    position = 1.0 / (1.0 + exp(-3.0 * raw))
    
    # 季度间最大变化 ±30%（防止剧烈跳变）
    position = clip(position, prev_position - 0.30, prev_position + 0.30)
    
    return clip(position, 0.02, 0.98)
```

#### 2.2.5 季度实际数据的作用

季度披露数据**不作为实时决策信号**，而是用于：
1. **训练标签**：每个季度末的特征 → 下一季度实际信心指数（监督学习目标）
2. **模型校准**：每个新季度数据发布后重新训练，更新特征→信心的映射关系
3. **预测误差监控**：追踪 |预测 - 实际|，若连续 2 季误差 > 0.15，触发模型重建

#### 2.2.6 集成方式

```python
# etf-nt-rotation 策略中获取仓位信号：
from pathlib import Path
import json

def get_predicted_position() -> float:
    """读取 nt-position-sizer 预测引擎输出的仓位"""
    cache = Path('.cache/national_team/predicted_position.json')
    if cache.exists():
        data = json.loads(cache.read_text())
        age = (datetime.now() - datetime.fromisoformat(data['date'])).days
        if age < 7:  # 7天内有效
            return data['predicted_position']
    # 降级：默认满仓
    return 1.0
```

#### 2.2.7 数据流总结

```
┌──────────────────────┐    ┌──────────────────────┐
│  national_team_      │    │  nt-position-sizer/  │
│  tracker.py          │    │  predictive_engine.py│
│  (季度拉取持仓数据)   │    │  (日频K线→预测仓位)   │
├──────────────────────┤    ├──────────────────────┤
│ 输入: akshare API    │    │ 输入: HS300 日线K线   │
│ 输出: .cache/        │    │ 输出: .cache/        │
│   national_team/     │    │   national_team/     │
│   holdings_*.parquet │    │   predicted_position │
│   (训练标签)          │    │   .json (策略消费)    │
└──────┬───────────────┘    └──────┬───────────────┘
       │                            │
       └──────────┬─────────────────┘
                  ↓
┌──────────────────────────────────────────────┐
│         etf-nt-rotation/scripts/             │
│         backtest_nt_rotation.py              │
│                                              │
│  get_predicted_position() → pos ∈ [0.02,0.98]│
│  月度 ETF 轮动分配 = 1/5 × pos               │
│  现金 = 1 - pos                               │
└──────────────────────────────────────────────┘
```

### 2.3 数据降级策略

| 优先级 | 仓位信号源 | 失败时动作 |
|--------|-----------|-----------|
| 1 | `nt-position-sizer` 预测引擎（`predicted_position.json`，7天内有效） | — |
| 2 | `nt-position-sizer` 实际仓位引擎（`position_engine.py`，基于最新季度数据） | 打印警告 |
| 3 | 降级模式 | `position = 1.0`（满仓），策略退化为纯排名轮动 |

---

## 3. ETF 候选池（35 只，已确认代码格式）

### 3.1 宽基 ETF（8 只）

| 代码 | 名称 | 类型 |
|------|------|------|
| 510300 | 华泰柏瑞沪深300ETF | 大盘 |
| 510500 | 南方中证500ETF | 中盘 |
| 510050 | 华夏上证50ETF | 大盘 |
| 159915 | 易方达创业板ETF | 创业板 |
| 588000 | 华夏科创50ETF | 科创板 |
| 159949 | 华安创业板50ETF | 创业板 |
| 512100 | 南方中证1000ETF | 小盘 |
| 510880 | 华泰柏瑞红利ETF | 红利 |

### 3.2 行业/主题 ETF（27 只）

| 代码 | 名称 | 行业标签 |
|------|------|---------|
| 512880 | 国泰证券ETF | 金融 |
| 512800 | 华宝银行ETF | 金融 |
| 512660 | 易方达军工ETF | 军工 |
| 512670 | 鹏华国防ETF | 军工 |
| 512690 | 鹏华酒ETF | 消费 |
| 159736 | 招商食品饮料ETF | 消费 |
| 159996 | 国泰家电ETF | 消费 |
| 159995 | 华夏芯片ETF | 科技 |
| 512480 | 国泰半导体ETF | 科技 |
| 512760 | 国泰半导体50ETF | 科技 |
| 159869 | 华夏游戏ETF | 科技 |
| 512980 | 广发传媒ETF | 传媒 |
| 515050 | 华夏5GETF | 科技 |
| 516510 | 易方达云计算ETF | 科技 |
| 159865 | 华夏人工智能ETF | 科技 |
| 512010 | 华夏医药ETF | 医药 |
| 512170 | 华宝医疗ETF | 医药 |
| 159755 | 华夏新能源车ETF | 新能源 |
| 515790 | 天弘光伏ETF | 新能源 |
| 561910 | 易方达电池ETF | 新能源 |
| 159611 | 广发电力ETF | 公用事业 |
| 515220 | 国泰煤炭ETF | 周期 |
| 516970 | 国泰建材ETF | 周期 |
| 512200 | 华夏房地产ETF | 基建 |
| 516950 | 广发基建ETF | 基建 |
| 159766 | 富国旅游ETF | 消费 |
| 561330 | 国泰矿业ETF | 周期 |

> **注意**：以上代码基于 efinance 已知支持范围。实现阶段需逐只验证数据可用性（最少 100 个交易日），不可用的剔除并记录到 `flow/踩坑记录.md`。

---

## 4. 因子定义（精确数学公式）

### 4.1 因子 1：月度涨幅（Momentum）

```
momentum_t = (close_t - close_{t-21}) / close_{t-21}

其中：
  close_t = ETF 在第 t 个交易日的收盘价
  21 = 约一个月的交易日数（参数 MOM_BARS，可配）
```

**边界处理**：
- 若 `close_{t-21}` 不存在（上市不足 21 日），该日 `momentum = NaN`，不参与排名
- 若 `close_{t-21} == 0`，`momentum = NaN`

### 4.2 因子 2：量比（Volume Ratio）

```
volume_ratio_t = SMA(volume, 5)_t / SMA(volume, 44)_t

其中：
  volume_t = ETF 在第 t 个交易日的成交量
  SMA(volume, 5) = 过去 5 个交易日成交量均值（短期）
  SMA(volume, 44) = 过去 44 个交易日成交量均值（约 2 个月，长期基线）
```

**边界处理**：
- 若 `SMA(volume, 44)_t == 0`，`volume_ratio = 1.0`
- 若成交量数据缺失，当日量比 = NaN

### 4.3 综合得分

```
对每个交易日 t，对每只 ETF i：

  pct_rank_mom_i = percentile_rank(momentum_i, 所有 ETF 的 momentum)  # 0~1，越高越好
  pct_rank_vol_i = percentile_rank(volume_ratio_i, 所有 ETF 的 volume_ratio)  # 0~1，越高越好

  score_i = MOMENTUM_WEIGHT × pct_rank_mom_i + VOLUME_WEIGHT × pct_rank_vol_i

  排名：score 降序（高分 = 动量强 + 放量确认）
```

**默认权重**（经 152 组网格搜索优化）：
- `MOMENTUM_WEIGHT = 0.50`（优化后：50/50 > 60/40）
- `VOLUME_WEIGHT = 0.50`
- `MOM_BARS = 14`（优化后：14d > 21d，更快捕获轮动信号）

**Percentile Rank 定义**：
```python
# 对于值 v 在数组 arr 中的分位数排名
pct_rank = (arr < v).sum() / len(arr)
# 结果范围 (0, 1)，值越大排名越高
```

---

## 5. 选股算法（伪代码）

### 5.1 主流程

```python
def select_etfs(
    scores: pd.Series,       # 所有 ETF 的综合得分
    rank_min: int = 5,       # 选股排名下限（跳过前 4 名）
    rank_max: int = 20,      # 选股排名上限
    top_n: int = 5,          # 最终持仓数量
    nt_cap_factor: float = 1.0,  # 国家队仓位系数
) -> dict[str, float]:
    """
    返回: {etf_code: weight} 仓位映射
    """
    # Step 1: 排名（降序，第一名 = 最高得分）
    ranked = scores.dropna().sort_values(ascending=False)

    # Step 2: 截取第 rank_min ~ rank_max 名
    n = len(ranked)
    if n < rank_max:
        # ETF 总数不足 20，收缩区间
        lo = min(rank_min, n)
        hi = n
    else:
        lo = rank_min
        hi = rank_max
    candidates = ranked.iloc[lo - 1 : hi]  # 0-based index

    # Step 3: 从候选中取中间 top_n 只
    # 候选项按排名排序，取中段位置
    m = len(candidates)
    if m <= top_n:
        selected = list(candidates.index)
    else:
        # 计算中段起始位置（偏上，略优）
        start = (m - top_n) // 2
        selected = list(candidates.index[start : start + top_n])

    # Step 4: 等权重分配（考虑国家队仓位系数）
    weight_per = (1.0 / len(selected)) * nt_cap_factor if selected else 0.0
    cash_pct = 1.0 - nt_cap_factor  # 剩余为现金

    return {code: weight_per for code in selected}
```

### 5.2 选股位置可视化

```
排名  1   2   3   4  | 5  6  7  8  9 [10 11 12 13 14] 15 16 17 18 19 20 | 21 22 ...
      ← 跳过(过热) → | ←————— 候选区(5-20) ———————→ | ← 跳过(动量不足) →
                        ↑ 取中段 10-14 共 5 只 ↑
```

### 5.3 候选不足时的降级策略

```python
if n < top_n:
    # 连 5 只都不够，有多少选多少
    selected = list(ranked.index[:n])
elif n < rank_max:
    # 不足 20 只，缩小区间（保持取中段逻辑）
    lo = 1
    hi = n
    # ...从 1~n 中取中间 top_n 只
```

---

## 6. 仓位管理规则（预测模型驱动）

### 6.1 仓位信号获取

```python
def get_position_signal(date) -> float:
    """
    获取指定日期的仓位信号。
    
    优先使用预测引擎（K线拟合，日频更新，领先1季度），
    不可用时回退到实际仓位引擎（基于最新季度披露）。
    
    返回: 仓位比例 ∈ [0.02, 0.98]
    """
    # 1. 尝试预测仓位
    pred = _load_predicted_position()
    if pred and pred_age_days(pred) < 7:
        return pred['predicted_position']
    
    # 2. 回退到实际仓位
    actual = _load_actual_position()
    if actual:
        return actual['position']
    
    # 3. 降级：满仓（不做仓位择时，纯 ETF 轮动）
    return 1.0
```

### 6.2 仓位生效规则

```
月末计算排名时:
  预测仓位 = get_position_signal(T_sell)  # 日频更新的连续值
  
  每只 ETF 仓位 = (1.0 / 5) × 预测仓位
  现金 = 1.0 - 预测仓位
  
  示例:
    预测仓位 = 0.74 → 每只 ETF 14.8%，现金 26%
    预测仓位 = 0.35 → 每只 ETF 7.0%，现金 65%
    预测仓位 = 1.00 → 每只 ETF 20.0%，现金 0%（满仓轮动）
```

### 6.3 与旧版（离散闸门）的对比

| 维度 | 旧版（季度离散闸门） | 新版（K线预测连续仓位） |
|------|-------------------|---------------------|
| 数据源 | 季度披露（滞后1-4月） | 日频K线（实时） |
| 输出类型 | 离散档位（1.0/0.7/0.4） | 连续值（0.02-0.98） |
| 更新频率 | 每季度一次 | 每日更新 |
| 信号领先性 | 滞后（跟着后视镜） | 领先（预判国家队行为） |
| 极端情况 | 4个月真空期不更新 | 每日刷新，持续有效 |
| 模型复杂度 | 简单 if-else | Ridge回归 + Sigmoid映射 |

### 6.4 仓位→盈利 校准优化（v2.1 新增）

> **核心问题**：预测模型优化目标是「准确预测国家队行为」，而非「最大化投资收益」。
> **解决思路**：在预测仓位基础上，增加一个**可优化的校准层**，通过对历史回测的网格搜索，找到使 ETF 轮动策略收益最大化的映射参数。

#### 6.4.1 校准公式

```
最终仓位 = clip(校准函数(预测仓位), 下限, 上限)

校准函数（非对称S曲线）:
  if 预测仓位 > 0.5:
      最终仓位 = 预测仓位 × alpha  (放大系数，默认 1.0)
  else:
      最终仓位 = 预测仓位 × beta   (缩小系数，默认 1.0)

其中 alpha, beta 为待优化参数。
```

#### 6.4.2 待优化参数

| 参数 | 默认值 | 搜索范围 | 说明 |
|------|--------|---------|------|
| `alpha` | 1.0 | [0.8, 1.8] 步长 0.1 | 高信号放大系数 |
| `beta` | 1.0 | [0.3, 1.0] 步长 0.1 | 低信号缩小系数 |
| `pos_floor` | 0.02 | [0, 0.3] 步长 0.05 | 最低仓位（避免完全空仓错失反弹） |
| `pos_ceil` | 0.98 | [0.7, 1.0] 步长 0.05 | 最高仓位（避免满仓承受黑天鹅） |

#### 6.4.3 优化目标

```python
def calibration_score(params, backtest_results):
    """
    综合评分 = 年化收益 × 0.4 + 夏普 × 0.3 - 最大回撤 × 0.3

    为什么不是纯收益最大化：
      - 加入夏普惩罚高波动策略
      - 加入回撤惩罚极端风险
      - 综合评分更稳健
    """
    alpha, beta, floor, ceil = params
    cagr = results['cagr']
    sharpe = results['sharpe']
    max_dd = abs(results['max_drawdown'])
    return cagr * 0.4 + sharpe * 0.3 - max_dd * 0.3
```

#### 6.4.4 量化分析（2024Q2-2026Q1 样本）

```
仓位信号 vs 下季度 HS300 收益:
  高仓位(>60%): 3个季度, 平均收益 +6.30%, 胜率 67%
  中仓位(50-70%): 3个季度, 平均收益 +0.67%

  预测仓位 vs 下季度收益 相关系数: +0.88 (强正相关)

策略对比（年化）:
  满仓持有:            +18.09%
  预测仓位缩放:        +13.01%  (保守，回撤更小)
  非对称(高1.3x低0.5x): +17.41%  (接近满仓，风险居中)
  阈值过滤(<30%空仓):  +13.01%

结论: 非对称策略在保持接近满仓收益的同时降低了回撤。
      完全按预测仓位缩放过于保守（牺牲了太多收益）。
```

#### 6.4.5 实现计划

校准优化将在 T05（参数优化）中通过网格搜索完成，与 ETF 选股参数（排名区间、权重等）联合优化。目标是在 ETF 轮动回测框架内找到全局最优的 `(alpha, beta, pos_floor, pos_ceil)` 组合。

---

## 7. 调仓节奏

### 7.1 时间线

```
月份 M 的最后一个交易日 (T_sell):
  ├─ 收盘价卖出全部 5 只持仓
  ├─ 计算下月排名（基于 T_sell 日及之前的数据）
  ├─ 应用国家队仓位系数
  └─ 生成 M+1 月持仓信号

月份 M+1 的第一个交易日 (T_buy):
  └─ 开盘价买入目标 5 只 ETF

月份 M+1 的最后一个交易日 (T_sell_next):
  └─ 循环...
```

### 7.2 交易日识别

```python
def get_monthly_schedule(dates: list[pd.Timestamp], buy_day_offset: int = 0):
    """
    dates: 所有可用交易日
    buy_day_offset: 0 = 第一个交易日, 1 = 第二个, 以此类推
    
    返回: [{'month': '2024-01', 'buy': date, 'sell': date}, ...]
    """
    # 按月分组
    # 每月第一个交易日 = min(dates in month)
    # 每月最后一个交易日 = max(dates in month)
```

### 7.3 首次调仓

```
回测开始日 (2019-01-02):
  ├─ 直接买入（视为月初第一个交易日）
  └─ 第一次排名用已有数据（可能不足一个月，用最短可用窗口）
```

---

## 8. 回测框架设计

### 8.1 输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `etf_codes` | list[str] | 35只代码 | ETF 候选池 |
| `start_date` | str | '20190101' | 回测起始日 |
| `end_date` | str | '20260612' | 回测结束日 |
| `initial_capital` | float | 1_000_000 | 初始资金 |
| `momentum_weight` | float | 0.6 | 涨幅权重 |
| `volume_weight` | float | 0.4 | 量比权重 |
| `mom_bars` | int | 21 | 动量回溯天数 |
| `vol_short` | int | 5 | 量比短期窗口 |
| `vol_long` | int | 44 | 量比长期窗口 |
| `rank_min` | int | 5 | 选股起始排名 |
| `rank_max` | int | 20 | 选股结束排名 |
| `top_n` | int | 5 | 持仓数量 |
| `trade_cost` | float | 0.001 | 单边手续费率 (0.1%) |
| `slippage` | float | 0.001 | 滑点 (0.1%) |

### 8.2 输出指标

```python
{
    'total_return': float,       # 总收益率
    'cagr': float,               # 年化收益率
    'annual_volatility': float,  # 年化波动率
    'sharpe_ratio': float,       # 夏普比率（无风险利率=2%）
    'sortino_ratio': float,      # 索提诺比率
    'max_drawdown': float,       # 最大回撤
    'max_drawdown_duration': int,# 最大回撤持续天数
    'calmar_ratio': float,       # 卡玛比率
    'win_rate': float,           # 日胜率
    'monthly_win_rate': float,   # 月胜率
    'total_trades': int,         # 总交易笔数
    'num_months': int,           # 回测月数
    'avg_holdings': float,       # 平均持仓 ETF 数量
    'nt_gate_activations': int,  # 国家队闸门触发次数
    'benchmark_cagr': float,     # 等权基准年化
    'excess_return': float,      # 超额收益
    'equity_curve': pd.Series,   # 净值曲线
    'monthly_returns': pd.Series,# 月度收益
}
```

### 8.3 基准

**等权持有全部候选 ETF**（买入持有，不调仓），作为策略的超额收益比较基准。

---

## 9. 参数表（含优化范围）

| 参数 | 默认值 | 优化范围 | 步长 | 说明 |
|------|--------|---------|------|------|
| `momentum_weight` | **0.50** | [0.3, 0.8] 步长 0.1 | 涨幅因子权重（经网格搜索优化：50/50最优） |
| `volume_weight` | **0.50** | = 1 - momentum_weight | — | 量比因子权重 |
| `mom_bars` | **14** | [10, 44] 步长 5 | 动量回溯天数（经优化：14d > 21d） |
| `vol_short` | 5 | [3, 10] | 1 | 量比短期窗口 |
| `vol_long` | 44 | [21, 66] | 5 | 量比长期窗口 |
| `rank_min` | **1** | [1, 3] | 1 | 选股起始排名（经证伪：跳过头部=放弃收益） |
| `rank_max` | **10** | [5, 15] | 1 | 选股结束排名（扩大到前10，增加候选池） |
| `top_n` | **8** | [3, 10] | 1 | 持仓数量（8只：Sharpe最优，收益/集中度平衡） |
| `nt_threshold_bull` | 0.05 | 已废弃（v2.0 替换为预测模型） | — |
| `nt_threshold_bear` | -0.05 | 已废弃 | — |
| `nt_cap_high` | 1.0 | 已废弃（仓位由预测引擎连续输出） | — |
| `nt_cap_medium` | 0.7 | 已废弃 | — |
| `nt_cap_low` | 0.4 | 已废弃 | — |
| `pred_position_min` | 0.02 | 固定（预测引擎内置下限） | — |
| `pred_position_max` | 0.98 | 固定（预测引擎内置上限） | — |
| `pred_cache_max_age_days` | 7 | [3, 14] | 预测仓位缓存有效期 |
| `pos_alpha` | 1.0 | [0.8, 1.8] 步长 0.1 | 高信号(>0.5)放大系数 |
| `pos_beta` | 1.0 | [0.3, 1.0] 步长 0.1 | 低信号(<0.5)缩小系数 |
| `pos_floor` | 0.02 | [0, 0.3] 步长 0.05 | 最低仓位 |
| `pos_ceil` | 0.98 | [0.7, 1.0] 步长 0.05 | 最高仓位 |

---

## 10. 边界条件 & 异常处理

### 10.1 数据异常

| 场景 | 处理 |
|------|------|
| ETF 退市/暂停交易 | 从候选池移除，记录到 `flow/踩坑记录.md` |
| 交易日不足 21 天 | 该 ETF 当日因子 = NaN，不参与排名 |
| 全市场停牌（如春节） | 沿用上一交易日排名 |
| 数据源返回空 | 自动切换到降级数据源 |
| 网络超时 | 重试 3 次，间隔 2s；全失败则退出并报错 |

### 10.2 选股异常

| 场景 | 处理 |
|------|------|
| 候选池 < top_n 只 | 全选可用 ETF，等权重 |
| 所有 ETF 动量 < 0 | 仍然排名选股（不做绝对动量过滤，首次迭代简化） |
| 排名区间无 ETF | 扩大到全部排名 |

### 10.3 交易异常

| 场景 | 处理 |
|------|------|
| 买入日非交易日 | 顺延到下一交易日 |
| 卖出日非交易日 | 提前到上一交易日 |
| 买入资金不足 | 按比例缩量（实际不会，因为是等权重买整数份额的近似） |

---

## 11. 策略评估标准（来自 charter）

| 指标 | 目标值 | 最低可接受 |
|------|--------|----------|
| 年化收益率 | ≥ 15% | ≥ 10% |
| 夏普比率 | ≥ 1.2 | ≥ 0.8 |
| 最大回撤 | ≤ 20% | ≤ 25% |
| 月胜率 | ≥ 55% | ≥ 50% |
| Calmar比率 | ≥ 0.75 | ≥ 0.4 |

---

## 12. 代码实现蓝图

### 12.1 文件结构（规划）

```
scripts/
├── fetch_etf_data.py          # T03: ETF 行情数据获取 + 缓存
├── fetch_nt_data.py           # T02: 国家队持仓季度数据获取（训练标签用）
├── backtest_nt_rotation.py    # T04: 核心回测脚本（~500行）
├── optimize_params.py         # T05: 参数网格搜索优化
└── live_signal.py             # T06: 月度实盘信号 + PushPlus 推送

依赖外部项目（无需重复开发）:
├── ../national_team_tracker.py           # 季度国家队数据拉取
├── ../national_team_conviction.py        # 季度信心指数计算
├── ../nt-position-sizer/src/
│   ├── position_engine.py                # 实际仓位引擎（回退用）
│   └── predictive_engine.py              # 预测仓位引擎（主力信号源）
└── .cache/national_team/
    ├── holdings_*.parquet                # 季度原始持仓
    ├── conviction.json                   # 实际信心指数
    └── predicted_position.json           # 预测仓位（策略消费）
```

### 12.2 回测脚本核心类

```python
@dataclass
class NTRotationConfig:
    """策略参数配置"""
    etf_codes: list[str]
    # 因子参数
    momentum_weight: float = 0.60
    volume_weight: float = 0.40
    mom_bars: int = 21
    vol_short: int = 5
    vol_long: int = 44
    # 选股参数
    rank_min: int = 5
    rank_max: int = 20
    top_n: int = 5
    # 交易成本
    trade_cost: float = 0.001
    slippage: float = 0.001
    # 仓位信号（从预测引擎获取，不作为参数调优）
    use_predictive_position: bool = True


class NTRotationBacktest:
    """国家队 ETF 轮动回测引擎"""

    def __init__(self, config: NTRotationConfig): ...

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """加载 ETF 行情 + 预测仓位时间序列"""

    def get_position_signal(self, date) -> float:
        """获取当日预测仓位（优先），不可用时回退实际仓位或满仓"""

    def calc_factors(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """计算动量 + 量比因子"""

    def calc_scores(self, momentum: pd.DataFrame, volume_ratio: pd.DataFrame) -> pd.DataFrame:
        """计算综合得分排名"""

    def select_etfs(self, scores_row: pd.Series, position: float) -> dict[str, float]:
        """选股，按预测仓位比例分配资金"""

    def run(self) -> BacktestResult:
        """执行完整回测"""

    def report(self) -> str:
        """生成绩效报告"""
```

---

## 13. 风险提示与已知限制

1. **预测模型非完美替代**：K 线拟合国家队行为基于历史相关性，结构性突变（政策转向、市场机制变更）会导致预测失效
2. **模型训练数据有限**：国家队季度数据仅约 30-40 个样本点（~10年 × 4季），Ridge 回归在样本外表现需持续监控
3. **国家队持仓 ≠ ETF 持仓**：国家队主要持有个股，我们用它预测大盘方向而非 ETF 轮动本身——这个传导链条存在噪音
4. **中段选股逻辑未经检验**：跳过头部的代价可能大于好处（头部动量的持续性在 A 股较强）
5. **无行业中性**：可能在单一行业过度集中
6. **模型漂移**：需每季度新数据发布后重训练，至少每 2 季度检查一次特征重要性变化

---

## 附录 A：国家队列名关键词（用于 efinance 十大流通股东识别）

```python
NATIONAL_TEAM_KEYWORDS = [
    '中央汇金', '汇金资产', '证金公司', '中证金融',
    '社保基金', '全国社保', '基本养老保险', '养老金',
    '国家集成电路', '中国国有企业结构调整基金',
]
```

## 附录 B：参考资源

- [efinance 文档](https://github.com/Micro-sheep/efinance)
- [akshare 文档](https://akshare.akfamily.xyz/)
- V4.2 行业轮动源码：`c:/AI/etf_sector_rotation_v4.2.py`
- backtest 回测引擎：`c:/AI/backtest/engine.py`
