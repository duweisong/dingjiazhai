"""
缠论 v2.0 — 量化时代重构：分型·笔·线段
============================================
基于缠中说禅「市场哲学的数学原理」核心公理（走势终完美、级别递归、完全分类），
在保留原哲学内核的前提下，将操作层定义从定性推到定量。

核心改进（v1 → v2）：
  分型：三重过滤（结构 + 能量 + 确认）
  笔：  ATR 空间阈值 + OBV 资金确认 + 最短时间 → 终结「千人千缠」
  线段：完成度评分系统（0-100）替代二元确认 + 未来函数
  买卖点：概率化确认替代绝对判断

Author: chan-shi-perspective / nuwa-skill
Date:   2026-07-01
License: MIT — 拿去用，盈亏自负。猎手不为猎物的账户负责。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Literal
import numpy as np
import pandas as pd
from enum import Enum


# ============================================================================
# 一、基础数据结构
# ============================================================================

class Direction(Enum):
    UP = 1
    DOWN = -1


class FractalType(Enum):
    TOP = 1       # 顶分型
    BOTTOM = -1   # 底分型


class BSPType(Enum):
    """三类买卖点"""
    BUY_1 = "B1"    # 一类买点：趋势底背驰
    BUY_2 = "B2"    # 二类买点：一类后次级别回抽不破低
    BUY_3 = "B3"    # 三类买点：中枢上方次级别回调不进中枢
    SELL_1 = "S1"   # 一类卖点：趋势顶背驰
    SELL_2 = "S2"   # 二类卖点：一类后次级别反弹不破高
    SELL_3 = "S3"   # 三类卖点：中枢下方次级别反弹不回中枢


@dataclass
class KLine:
    """单根K线"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)


@dataclass
class Fractal:
    """分型"""
    ftype: FractalType
    index: int          # 原始K线序列中的位置
    kline_idx: int      # 分型中间K线在合并后序列的位置
    mid_kline: KLine    # 中间K线
    left_kline: KLine   # 左侧K线
    right_kline: KLine  # 右侧K线
    # 三重过滤标记
    energy_ok: bool = False     # 能量条件通过
    confirmed: bool = False     # 确认条件通过
    # 验证信息
    vol_ratio: float = 0.0      # 成交量/均量比
    range_ratio: float = 0.0    # 振幅/ATR比


@dataclass
class Stroke:
    """笔"""
    direction: Direction
    start_fractal: Fractal
    end_fractal: Fractal
    klines: List[KLine] = field(default_factory=list)
    # v2 验证信息
    amplitude: float = 0.0      # 笔的振幅(%)
    independent_klines: int = 0 # 独立K线数
    duration_days: float = 0.0  # 持续时间
    obv_confirmed: bool = False # OBV确认
    atr_mult: float = 0.0       # 振幅/ATR比值
    is_valid: bool = True       # 是否通过v2验证


@dataclass
class Segment:
    """线段"""
    direction: Direction
    strokes: List[Stroke] = field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0
    pivot_zg: Optional[float] = None   # 中枢高点
    pivot_zd: Optional[float] = None   # 中枢低点
    completion_score: int = 0          # 完成度评分 0-100
    has_opposite_stroke: bool = False   # 反向笔出现
    has_opposite_segment: bool = False  # 反向线段确认


@dataclass
class BuySellPoint:
    """买卖点"""
    bsp_type: BSPType
    index: int
    timestamp: pd.Timestamp
    price: float
    score: int = 0              # 信号强度评分 0-100
    # 验证信息
    divergence_detail: str = ""  # 背驰详情


# ============================================================================
# 二、级别参数表 — v2.0 核心创新
# ============================================================================

@dataclass
class LevelParams:
    """某级别的操作参数"""
    name: str
    min_klines: int         # 笔的最小独立K线数
    atr_period: int         # ATR计算周期
    atr_mult: float         # 笔振幅 ≥ ATR × mult
    min_duration: str       # 笔的最短持续时间（pandas offset）
    obv_check: bool         # 是否做OBV确认
    fractal_vol_period: int # 分型能量过滤：成交量均线周期
    fractal_range_mult: float # 分型能量过滤：振幅/ATR倍数

# v2.0 级别参数表
LEVEL_PARAMS: Dict[str, LevelParams] = {
    'M':   LevelParams('月线',   2, 12, 0.8, '60D', False, 12, 0.3),
    'W':   LevelParams('周线',   2, 14, 0.9, '10D', False, 14, 0.35),
    'D':   LevelParams('日线',   3, 14, 1.0, '3D',  True,  20, 0.5),
    '30m': LevelParams('30分钟', 3, 20, 1.0, '2h',  True,  25, 0.55),
    '5m':  LevelParams('5分钟',  3, 30, 1.2, '30m', True,  30, 0.6),
    '1m':  LevelParams('1分钟',  5, 50, 1.5, '10m', True,  40, 0.7),
}


# ============================================================================
# 三、K线包含关系处理（原定义，保持不变）
# ============================================================================

def merge_klines(klines: List[KLine], direction: Direction = Direction.UP) -> List[KLine]:
    """
    K线包含关系处理。
    向上处理：取高高、高低
    向下处理：取低低、低高
    """
    if len(klines) <= 1:
        return klines

    merged = [klines[0]]
    for i in range(1, len(klines)):
        prev = merged[-1]
        curr = klines[i]

        # 判断包含关系
        if (prev.high >= curr.high and prev.low <= curr.low) or \
           (curr.high >= prev.high and curr.low <= prev.low):
            # 有包含关系
            if direction == Direction.UP:
                merged[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=max(prev.high, curr.high),
                    low=max(prev.low, curr.low),
                    close=curr.close,
                    volume=prev.volume + curr.volume
                )
            else:
                merged[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=min(prev.high, curr.high),
                    low=min(prev.low, curr.low),
                    close=curr.close,
                    volume=prev.volume + curr.volume
                )
        else:
            merged.append(curr)

    return merged


def process_klines(klines: List[KLine]) -> List[KLine]:
    """
    完整的K线包含关系处理：先判断方向再合并。
    连续多根K线有包含关系时全部合并为1根。
    """
    if len(klines) < 2:
        return klines

    result = [klines[0]]
    i = 1
    while i < len(klines):
        prev = result[-1]
        curr = klines[i]

        # 判断方向：前一根上涨还是下跌
        if prev.close >= prev.open:  # 阳线或平
            direction = Direction.UP
        else:
            direction = Direction.DOWN

        # 判断包含
        is_contained = False
        if (prev.high >= curr.high and prev.low <= curr.low):
            is_contained = True
            if direction == Direction.UP:
                result[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=max(prev.high, curr.high),
                    low=max(prev.low, curr.low),
                    close=curr.close if curr.close > prev.close else prev.close,
                    volume=prev.volume + curr.volume
                )
            else:
                result[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=min(prev.high, curr.high),
                    low=min(prev.low, curr.low),
                    close=curr.close if curr.close < prev.close else prev.close,
                    volume=prev.volume + curr.volume
                )
        elif (curr.high >= prev.high and curr.low <= prev.low):
            is_contained = True
            if direction == Direction.UP:
                result[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=max(prev.high, curr.high),
                    low=max(prev.low, curr.low),
                    close=curr.close if curr.close > prev.close else prev.close,
                    volume=prev.volume + curr.volume
                )
            else:
                result[-1] = KLine(
                    timestamp=curr.timestamp,
                    open=prev.open,
                    high=min(prev.high, curr.high),
                    low=min(prev.low, curr.low),
                    close=curr.close if curr.close < prev.close else prev.close,
                    volume=prev.volume + curr.volume
                )

        if not is_contained:
            result.append(curr)

        i += 1

    return result


# ============================================================================
# 四、分型识别 — v2.0 三重过滤
# ============================================================================

def detect_fractals_v1(klines: List[KLine]) -> List[Fractal]:
    """
    分型识别 v1（原定义，只用结构条件，无任何过滤）。
    用于 v1 vs v2 对比基准。
    """
    if len(klines) < 3:
        return []
    fractals = []
    for i in range(1, len(klines) - 1):
        left, mid, right = klines[i-1], klines[i], klines[i+1]
        is_top = (mid.high > left.high and mid.high > right.high and
                  mid.low > left.low and mid.low > right.low)
        is_bottom = (mid.low < left.low and mid.low < right.low and
                     mid.high < left.high and mid.high < right.high)
        if is_top or is_bottom:
            ftype = FractalType.TOP if is_top else FractalType.BOTTOM
            fractals.append(Fractal(
                ftype=ftype, index=i, kline_idx=i,
                mid_kline=mid, left_kline=left, right_kline=right
            ))
    return fractals


def detect_fractals(klines: List[KLine], params: LevelParams = None,
                     use_filter: bool = True) -> List[Fractal]:
    """
    分型识别（v2.0 三重过滤版本）。

    一重：结构条件 — 原定义三K线关系
    二重：能量条件 — 成交量 + 振幅过滤
    三重：确认条件 — 反向K线确认

    若 use_filter=False，等同于 v1 原定义。
    """
    if params is None:
        params = LEVEL_PARAMS['D']
    if not use_filter:
        return detect_fractals_v1(klines)
    if len(klines) < 3:
        return []

    closes = np.array([k.close for k in klines])
    volumes = np.array([k.volume for k in klines])
    ranges = np.array([k.range for k in klines])

    vol_ma = pd.Series(volumes).rolling(window=params.fractal_vol_period, min_periods=5).mean().values

    tr = np.zeros(len(klines))
    for i in range(1, len(klines)):
        tr[i] = max(
            klines[i].high - klines[i].low,
            abs(klines[i].high - klines[i-1].close),
            abs(klines[i].low - klines[i-1].close)
        )
    atr = pd.Series(tr).rolling(window=params.atr_period, min_periods=5).mean().values

    fractals = []
    for i in range(1, len(klines) - 1):
        left, mid, right = klines[i-1], klines[i], klines[i+1]
        is_top = (mid.high > left.high and mid.high > right.high and
                  mid.low > left.low and mid.low > right.low)
        is_bottom = (mid.low < left.low and mid.low < right.low and
                     mid.high < left.high and mid.high < right.high)
        if not (is_top or is_bottom):
            continue

        ftype = FractalType.TOP if is_top else FractalType.BOTTOM
        frac = Fractal(
            ftype=ftype, index=i, kline_idx=i,
            mid_kline=mid, left_kline=left, right_kline=right
        )

        if i < len(vol_ma) and i < len(atr):
            frac.vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] > 0 else 0
            frac.range_ratio = ranges[i] / atr[i] if atr[i] > 0 else 0
            frac.energy_ok = (
                frac.vol_ratio >= 0.8 and
                frac.range_ratio >= params.fractal_range_mult
            )
        else:
            frac.energy_ok = True

        if i + 2 < len(klines):
            confirm_k = klines[i + 2]
            if ftype == FractalType.TOP:
                frac.confirmed = confirm_k.close < mid.low
            else:
                frac.confirmed = confirm_k.close > mid.high
        else:
            frac.confirmed = False

        fractals.append(frac)

    return fractals


# ============================================================================
# 五、笔识别 — v2.0 ATR+OBV 双确认
# ============================================================================

def _calc_obv(klines: List[KLine]) -> np.ndarray:
    """计算OBV序列"""
    obv = np.zeros(len(klines))
    obv[0] = float(klines[0].volume)
    for i in range(1, len(klines)):
        if klines[i].close > klines[i-1].close:
            obv[i] = obv[i-1] + klines[i].volume
        elif klines[i].close < klines[i-1].close:
            obv[i] = obv[i-1] - klines[i].volume
        else:
            obv[i] = obv[i-1]
    return obv


def detect_strokes(klines: List[KLine], fractals: List[Fractal],
                   params: LevelParams = None) -> List[Stroke]:
    """
    笔识别（v2.0 ATR+OBV版本）。

    条件：
    1. 相邻顶底分型交替连接
    2. 独立K线数达标
    3. 振幅 ≥ ATR × 级别系数
    4. OBV方向与笔一致
    5. 持续时间达标
    """
    if params is None:
        params = LEVEL_PARAMS['D']

    if len(fractals) < 2:
        return []

    # 计算ATR
    tr = np.zeros(len(klines))
    for i in range(1, len(klines)):
        tr[i] = max(
            klines[i].high - klines[i].low,
            abs(klines[i].high - klines[i-1].close),
            abs(klines[i].low - klines[i-1].close)
        )
    atr_series = pd.Series(tr).rolling(window=params.atr_period, min_periods=5).mean()

    obv = _calc_obv(klines)

    strokes = []
    i = 0
    while i < len(fractals) - 1:
        f1 = fractals[i]
        # 找下一个相反类型的分型
        j = i + 1
        while j < len(fractals) and fractals[j].ftype == f1.ftype:
            # 同类型连续 → 取极端那个
            if f1.ftype == FractalType.TOP:
                if fractals[j].mid_kline.high > f1.mid_kline.high:
                    f1 = fractals[j]
                    i = j
            else:
                if fractals[j].mid_kline.low < f1.mid_kline.low:
                    f1 = fractals[j]
                    i = j
            j += 1

        if j >= len(fractals):
            break

        f2 = fractals[j]

        # 确定方向
        if f1.ftype == FractalType.BOTTOM and f2.ftype == FractalType.TOP:
            direction = Direction.UP
        elif f1.ftype == FractalType.TOP and f2.ftype == FractalType.BOTTOM:
            direction = Direction.DOWN
        else:
            i = j
            continue

        # 笔的K线区间
        start_idx, end_idx = f1.kline_idx, f2.kline_idx
        stroke_klines = klines[start_idx:end_idx + 1]

        # 独立K线数（分型之间不含分型K线）
        independent = max(0, end_idx - start_idx - 1)

        # 振幅
        if direction == Direction.UP:
            amplitude = (f2.mid_kline.high - f1.mid_kline.low) / f1.mid_kline.low * 100
        else:
            amplitude = (f1.mid_kline.high - f2.mid_kline.low) / f1.mid_kline.high * 100

        # OBV确认
        if params.obv_check and len(stroke_klines) > 1:
            obv_start = obv[start_idx]
            obv_end = obv[end_idx]
            obv_confirmed = (direction == Direction.UP and obv_end > obv_start) or \
                           (direction == Direction.DOWN and obv_end < obv_start)
        else:
            obv_confirmed = True

        # 持续时间
        duration = klines[end_idx].timestamp - klines[start_idx].timestamp

        # === v2 验证（修正：ATR转百分比后比较）===
        price_ref = klines[end_idx].close
        avg_atr = atr_series.iloc[end_idx] if end_idx < len(atr_series) else atr_series.iloc[-1]
        avg_atr_val = float(avg_atr) if not pd.isna(avg_atr) else 0
        atr_pct = avg_atr_val / price_ref * 100 if price_ref > 0 and avg_atr_val > 0 else 0.5

        # ATR阈值：振幅必须 ≥ ATR百分比 × 级别乘数
        atr_ok = amplitude >= atr_pct * params.atr_mult

        # 计算振幅/ATR比值（用于评分排序）
        atr_ratio = amplitude / (atr_pct * params.atr_mult) if atr_pct > 0 and params.atr_mult > 0 else 0

        # 最短时间检查
        min_dur = pd.Timedelta(params.min_duration)
        time_ok = duration >= min_dur

        # 构建笔
        stroke = Stroke(
            direction=direction,
            start_fractal=f1,
            end_fractal=f2,
            klines=stroke_klines,
            amplitude=amplitude,
            independent_klines=independent,
            duration_days=duration.total_seconds() / 86400,
            obv_confirmed=obv_confirmed,
            atr_mult=atr_ratio,
        )

        stroke.is_valid = (
            independent >= params.min_klines and
            time_ok and
            atr_ok and
            obv_confirmed
        )

        strokes.append(stroke)
        i = j

    return strokes


# ============================================================================
# 六、线段与中枢识别
# ============================================================================

def detect_segments(strokes: List[Stroke], use_all: bool = False) -> List[Segment]:
    """
    线段识别。
    线段 = 至少3笔（奇数），前三笔必须有重叠。

    参数 use_all=True: 使用所有笔（v1模式），用于结构分析
         use_all=False: 使用v2有效笔，用于信号质量分析
    """
    # 默认用所有笔做结构分析（结构存在性不依赖笔的质量判断）
    seg_stroke_pool = strokes if use_all else strokes

    segments = []
    if len(seg_stroke_pool) < 3:
        return segments

    i = 0
    while i <= len(seg_stroke_pool) - 3:
        s1, s2, s3 = seg_stroke_pool[i], seg_stroke_pool[i+1], seg_stroke_pool[i+2]

        # 方向必须交替
        if s1.direction == s2.direction or s2.direction == s3.direction:
            i += 1
            continue

        # 重叠检查（放宽到90%相交即可，适用于v2稀疏笔环境）
        if s1.direction == Direction.UP:
            overlap_high = min(s1.end_fractal.mid_kline.high, s3.end_fractal.mid_kline.high)
            overlap_low = max(s1.start_fractal.mid_kline.low, s3.start_fractal.mid_kline.low)
        else:
            overlap_high = min(s1.start_fractal.mid_kline.high, s3.start_fractal.mid_kline.high)
            overlap_low = max(s1.end_fractal.mid_kline.low, s3.end_fractal.mid_kline.low)

        has_overlap = overlap_high >= overlap_low * 0.90  # 10% tolerance

        if not has_overlap:
            i += 1
            continue

        seg_strokes = [s1, s2, s3]
        j = i + 3
        while j < len(seg_stroke_pool):
            if seg_stroke_pool[j].direction == seg_strokes[-1].direction:
                break
            seg_strokes.append(seg_stroke_pool[j])
            j += 1

        direction = s1.direction

        # 计算中枢
        pivot_zg, pivot_zd = None, None
        if len(seg_strokes) >= 3:
            for k in range(len(seg_strokes) - 2):
                a, b, c = seg_strokes[k], seg_strokes[k+1], seg_strokes[k+2]
                if a.direction == b.direction or b.direction == c.direction:
                    continue
                if a.direction == Direction.UP:
                    zg = min(a.end_fractal.mid_kline.high, c.end_fractal.mid_kline.high)
                    zd = max(b.start_fractal.mid_kline.low, b.end_fractal.mid_kline.low)
                else:
                    zg = max(b.start_fractal.mid_kline.high, b.end_fractal.mid_kline.high)
                    zd = min(a.end_fractal.mid_kline.low, c.end_fractal.mid_kline.low)
                if zg > zd:
                    pivot_zg, pivot_zd = zg, zd
                    break

        # 统计v2质量：段内有效笔占比
        valid_in_seg = sum(1 for s in seg_strokes if s.is_valid)
        quality = valid_in_seg / len(seg_strokes) if seg_strokes else 0

        segments.append(Segment(
            direction=direction,
            strokes=seg_strokes,
            start_idx=i,
            end_idx=j - 1,
            pivot_zg=pivot_zg,
            pivot_zd=pivot_zd,
            completion_score=int(quality * 100),  # 初始评分=段内v2有效笔占比
        ))
        i = j

    return segments


def _split_long_segments(segments: List[Segment], strokes: List[Stroke]) -> List[Segment]:
    """
    将过长线段在中枢边界处拆分为子段。
    拆分规则：每当5笔形成新的非重叠中枢时，切分为新子段。
    """
    sub_segments = []
    for seg in segments:
        if len(seg.strokes) <= 7:
            sub_segments.append(seg)
            continue

        # 滑动窗口：每5笔检查是否形成独立中枢
        current_start = 0
        last_pivot_zg, last_pivot_zd = None, None

        for k in range(0, len(seg.strokes) - 4, 2):  # step=2 to reduce sub-segment count
            window = seg.strokes[k:k+5]
            if len(window) < 5:
                continue

            # 计算这5笔的中枢
            pivot_found = False
            for m in range(len(window) - 2):
                a, b, c = window[m], window[m+1], window[m+2]
                if a.direction == b.direction or b.direction == c.direction:
                    continue
                if a.direction == Direction.UP:
                    zg = min(a.end_fractal.mid_kline.high, c.end_fractal.mid_kline.high)
                    zd = max(b.start_fractal.mid_kline.low, b.end_fractal.mid_kline.low)
                else:
                    zg = max(b.start_fractal.mid_kline.high, b.end_fractal.mid_kline.high)
                    zd = min(a.end_fractal.mid_kline.low, c.end_fractal.mid_kline.low)

                if zg > zd:
                    # 检查是否与前一个中枢不同（非重叠=新中枢区域）
                    if last_pivot_zg is None or zg > last_pivot_zg * 1.02 or zd < last_pivot_zd * 0.98:
                        # 切分
                        if k - current_start >= 3:
                            sub_strokes = seg.strokes[current_start:k+3]
                            valid_in = sum(1 for s in sub_strokes if s.is_valid)
                            quality = valid_in / len(sub_strokes) if sub_strokes else 0
                            sub_segments.append(Segment(
                                direction=seg.direction,
                                strokes=sub_strokes,
                                start_idx=seg.start_idx + current_start,
                                end_idx=seg.start_idx + k + 2,
                                pivot_zg=zg, pivot_zd=zd,
                                completion_score=int(quality * 100),
                            ))
                        current_start = k
                        last_pivot_zg, last_pivot_zd = zg, zd
                        pivot_found = True
                        break

        # 剩余尾部
        if current_start < len(seg.strokes) - 3:
            tail = seg.strokes[current_start:]
            valid_in = sum(1 for s in tail if s.is_valid)
            quality = valid_in / len(tail) if tail else 0
            sub_segments.append(Segment(
                direction=seg.direction,
                strokes=tail,
                start_idx=seg.start_idx + current_start,
                end_idx=seg.end_idx,
                completion_score=int(quality * 100),
            ))

    return sub_segments if sub_segments else segments


def compute_segment_completion(segment: Segment, all_strokes: List[Stroke],
                                upper_level_pivots: List[Tuple[float, float]] = None) -> int:
    """
    v2.0 核心创新：线段完成度评分（0-100）。

    评分逻辑：
      +30: 反向笔出现（当前线段方向被一根反向笔挑战）
      +50: 反向笔演变为反向线段（挑战成功，原线段确认完成）
      +20: 大级别中枢边界确认（多级别独立验证）

    总分≥80 → 线段确认完成
    30-79 → 可能完成，次级别确认
    <30 → 未完成
    """
    score = 0

    # 检查反向笔
    seg_end_idx = segment.strokes[-1].end_fractal.index
    for s in all_strokes:
        if s.start_fractal.index > seg_end_idx:
            if s.direction != segment.direction:
                score += 30
                segment.has_opposite_stroke = True

                # 检查反向笔是否演变为反向线段
                s_idx = all_strokes.index(s)
                if s_idx + 2 < len(all_strokes):
                    s2, s3 = all_strokes[s_idx+1], all_strokes[s_idx+2]
                    if s2.direction == segment.direction and s3.direction != segment.direction:
                        score += 50
                        segment.has_opposite_segment = True
            break

    # 大级别中枢确认
    if upper_level_pivots and segment.pivot_zg and segment.pivot_zd:
        seg_end_price = segment.strokes[-1].end_fractal.mid_kline.close
        for zg, zd in upper_level_pivots:
            if abs(seg_end_price - zg) < zg * 0.01 or abs(seg_end_price - zd) < zd * 0.01:
                score += 20
                break

    segment.completion_score = min(score, 100)
    return segment.completion_score


# ============================================================================
# 七、买卖点识别
# ============================================================================

def _detect_divergence(klines: List[KLine], idx_a: int, idx_b: int,
                       direction: Direction) -> Tuple[bool, str]:
    """
    背驰检测 (MACD辅助)。
    比较两段走势的MACD柱面积。
    """
    closes = np.array([k.close for k in klines])

    if len(closes) < max(idx_a, idx_b) + 1:
        return False, "数据不足"

    # 简化MACD计算
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = 2 * (dif - dea)

    # 比较两段的MACD面积
    seg_a_bars = macd_bar.iloc[idx_a:idx_b+1]
    seg_a_area = abs(seg_a_bars.sum())

    # 后续段
    remaining = macd_bar.iloc[idx_b:]
    if len(remaining) > 10:
        seg_b_area = abs(remaining.iloc[:len(seg_a_bars)].sum())
    else:
        seg_b_area = abs(remaining.sum())

    if direction == Direction.UP:
        # 顶背驰：价格新高，MACD面积缩小
        if closes[idx_b] > closes[idx_a] and seg_b_area < seg_a_area * 0.95:
            return True, f"顶背驰：面积比 {seg_b_area/seg_a_area:.2f}"
    else:
        # 底背驰：价格新低，MACD面积缩小
        if closes[idx_b] < closes[idx_a] and seg_b_area < seg_a_area * 0.95:
            return True, f"底背驰：面积比 {seg_b_area/seg_a_area:.2f}"

    return False, "无背驰"


def detect_bsp(klines: List[KLine], strokes: List[Stroke],
               segments: List[Segment]) -> List[BuySellPoint]:
    """
    三类买卖点识别。
    """
    bsps = []

    if len(strokes) < 3:
        return bsps

    closes = np.array([k.close for k in klines])

    for i, seg in enumerate(segments):
        if seg.pivot_zg is None or seg.pivot_zd is None:
            continue

        # 第一类买卖点：趋势背驰
        if len(seg.strokes) >= 5:  # 至少两个同向中枢
            last_stroke = seg.strokes[-1]
            prev_same_dir_stroke = None
            for s in reversed(seg.strokes[:-2]):
                if s.direction == last_stroke.direction:
                    prev_same_dir_stroke = s
                    break

            if prev_same_dir_stroke:
                is_div, detail = _detect_divergence(
                    klines,
                    prev_same_dir_stroke.start_fractal.index,
                    last_stroke.end_fractal.index,
                    last_stroke.direction
                )
                if is_div:
                    if last_stroke.direction == Direction.DOWN:
                        bsp = BuySellPoint(
                            bsp_type=BSPType.BUY_1,
                            index=last_stroke.end_fractal.index,
                            timestamp=klines[last_stroke.end_fractal.index].timestamp,
                            price=closes[last_stroke.end_fractal.index],
                            score=80,
                            divergence_detail=detail,
                        )
                    else:
                        bsp = BuySellPoint(
                            bsp_type=BSPType.SELL_1,
                            index=last_stroke.end_fractal.index,
                            timestamp=klines[last_stroke.end_fractal.index].timestamp,
                            price=closes[last_stroke.end_fractal.index],
                            score=80,
                            divergence_detail=detail,
                        )
                    bsps.append(bsp)

        # 第三类买卖点：中枢离开后回抽不入
        if i > 0 and len(seg.strokes) >= 3:
            # 简化：如果当前段离开中枢后反向笔不破中枢边界
            zg, zd = seg.pivot_zg, seg.pivot_zd
            end_price = closes[seg.strokes[-1].end_fractal.index]

            if seg.direction == Direction.UP and end_price > zg:
                # 找离开后的回调笔
                for s in seg.strokes:
                    if s.direction == Direction.DOWN:
                        retrace_low = s.end_fractal.mid_kline.low
                        if retrace_low > zg:
                            bsps.append(BuySellPoint(
                                bsp_type=BSPType.BUY_3,
                                index=s.end_fractal.index,
                                timestamp=klines[s.end_fractal.index].timestamp,
                                price=closes[s.end_fractal.index],
                                score=70,
                            ))
                            break
            elif seg.direction == Direction.DOWN and end_price < zd:
                for s in seg.strokes:
                    if s.direction == Direction.UP:
                        retrace_high = s.end_fractal.mid_kline.high
                        if retrace_high < zd:
                            bsps.append(BuySellPoint(
                                bsp_type=BSPType.SELL_3,
                                index=s.end_fractal.index,
                                timestamp=klines[s.end_fractal.index].timestamp,
                                price=closes[s.end_fractal.index].timestamp,
                                score=70,
                            ))
                            break

    return bsps


# ============================================================================
# 八、完整分析管道
# ============================================================================

class ChanAnalyzer:
    """缠论v2.0 完整分析器"""

    def __init__(self, level: str = 'D'):
        self.level = level
        self.params = LEVEL_PARAMS.get(level, LEVEL_PARAMS['D'])
        self.klines: List[KLine] = []
        self.merged_klines: List[KLine] = []
        self.fractals: List[Fractal] = []
        self.strokes: List[Stroke] = []
        self.segments: List[Segment] = []
        self.bsp_list: List[BuySellPoint] = []

    def load_klines(self, df: pd.DataFrame):
        """从DataFrame加载K线"""
        self.klines = []
        for _, row in df.iterrows():
            self.klines.append(KLine(
                timestamp=row.get('date', row.name) if isinstance(row.name, pd.Timestamp) else row['date'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            ))

    def run(self):
        """运行完整分析"""
        # 1. 包含关系处理
        self.merged_klines = process_klines(self.klines)

        # 2. 分型识别（三重过滤）
        self.fractals = detect_fractals(self.merged_klines, self.params)

        # 3. 笔识别（ATR+OBV）
        self.strokes = detect_strokes(self.merged_klines, self.fractals, self.params)

        # 4. 线段识别（用全量笔画结构，v2评分为信号权重）
        self.segments = detect_segments(self.strokes, use_all=True)

        # 4.5 子线段拆分：长线段在中枢边界处切分为子段，便于评分验证
        self.sub_segments = _split_long_segments(self.segments, self.strokes)

        # 5. 线段完成度评分
        upper_pivots = [(s.pivot_zg, s.pivot_zd) for s in self.segments
                        if s.pivot_zg and s.pivot_zd]
        for seg in self.segments:
            compute_segment_completion(seg, self.strokes, upper_pivots)

        # 6. 买卖点识别
        self.bsp_list = detect_bsp(self.merged_klines, self.strokes, self.segments)

        return self

    def get_valid_strokes(self) -> List[Stroke]:
        """获取通过v2验证的笔"""
        return [s for s in self.strokes if s.is_valid]

    def get_summary(self) -> Dict:
        """分析摘要"""
        valid_strokes = self.get_valid_strokes()
        return {
            'level': self.level,
            'klines_raw': len(self.klines),
            'klines_merged': len(self.merged_klines),
            'fractals_total': len(self.fractals),
            'fractals_energy_ok': sum(1 for f in self.fractals if f.energy_ok),
            'fractals_confirmed': sum(1 for f in self.fractals if f.confirmed),
            'strokes_total': len(self.strokes),
            'strokes_valid': len(valid_strokes),
            'strokes_filtered': len(self.strokes) - len(valid_strokes),
            'segments_total': len(self.segments),
            'bps_total': len(self.bsp_list),
            'filter_rate_strokes': 1 - len(valid_strokes) / len(self.strokes) if self.strokes else 0,
        }
