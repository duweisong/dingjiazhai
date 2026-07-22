
"""
资金结构分析器 —— 分析个股的参与者结构

徐翔旧逻辑只看流通市值。新框架必须回答: 这只票里的钱是谁的？

五类资金的行为特征:
  北向资金: 中期趋势投资，看ROE/估值 → 观察持续流入信号
  量化资金: T+1反向交易，封板即卖 → 潜伏量化不关注的标的
  游资: 追题材、互接盘 → 龙虎榜数据结合席位分析
  机构: 建仓慢、出货慢 → 机构重仓股不追板
  散户: 追涨杀跌 → 观察融资余额、散户情绪指数
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapitalProfile:
    """个股资金结构画像"""
    symbol: str
    name: str
    # 持股结构
    north_bound_pct: float      # 北向持股比例
    inst_holding_pct: float     # 机构持股比例
    retail_holding_pct: float   # 散户持股比例(估算)
    top10_concentration: float  # 前十大股东集中度
    # 交易特征
    avg_turnover: float         # 日均换手率
    recent_turnover: float      # 近期换手率
    vol_20d: float              # 20日均量
    # 量化渗透估计
    quant_penetration: float    # 量化参与度估计 0-1
    # 分类标签
    dominant_type: str          # "north"|"inst"|"retail"|"quant"|"mixed"
    is_quant_vulnerable: bool   # 是否容易被量化收割
    suitability: str            # 是否适合徐翔体系
    risk_flags: list            # 风险标记


class CapitalStructureAnalyzer:
    """
    资金结构分析器

    从可获取的公开数据推断个股的资金结构，
    判断量化渗透率、机构参与度、散户比例等。
    """

    def __init__(self,
                 north_bound_threshold=0.03,
                 inst_threshold=0.10,
                 quant_turnover_threshold=0.15,
                 retail_threshold=0.50):
        self.north_bound_threshold = north_bound_threshold
        self.inst_threshold = inst_threshold
        self.quant_turnover_threshold = quant_turnover_threshold
        self.retail_threshold = retail_threshold

    def estimate_quant_penetration(self, turnover_rate, mcap, close, volume):
        """
        估计量化资金参与度 (0-1)

        指标:
          - 换手率异常高 (>15%) → 量化参与度高
          - 小市值 (<50亿) → 量化偏好
          - 日内振幅大 + 涨幅小 → 量化T+0痕迹
        """
        score = 0.0
        if turnover_rate > 0.20:
            score += 0.4
        elif turnover_rate > 0.15:
            score += 0.3
        elif turnover_rate > 0.10:
            score += 0.15
        if mcap < 50e8:
            score += 0.3
        elif mcap < 100e8:
            score += 0.15
        return min(score, 1.0)

    def analyze(self, symbol, name, row) -> CapitalProfile:
        """
        分析单只股票的资金结构。

        row可包含以下可选字段:
          north_bound_pct, inst_holding, turnover, mcap, close, volume,
          top10_pct, amplitude, pct_chg, fund_holders
        """
        nb_pct = row.get("north_bound_pct", 0) or 0
        inst_pct = row.get("inst_holding", 0) or row.get("fund_holders", 0) or 0
        turnover = row.get("turnover", 0) or row.get("turnover_rate", 0) or 0.05
        mcap = row.get("mcap", 0) or row.get("circulating_mcap", 0) or 50e8
        close = row.get("close", 0) or 10.0
        volume = row.get("volume", 0) or 1e7
        top10 = row.get("top10_pct", 0) or row.get("top10_concentration", 0) or 0.50
        amplitude = row.get("amplitude", 0) or 0.05
        pct_chg = row.get("pct_chg", 0) or 0

        # 估算散户比例
        known = nb_pct + inst_pct + max(0, top10 - inst_pct)
        retail_pct = max(0, 1.0 - known)

        # 量化渗透估计
        quant_pen = self.estimate_quant_penetration(turnover, mcap, close, volume)

        # 主力类型判定
        risk_flags = []
        if nb_pct > self.north_bound_threshold:
            dominant = "north"
        elif inst_pct > self.inst_threshold:
            dominant = "inst"
            if mcap > 200e8:
                risk_flags.append("大盘机构重仓-不适合短线")
        elif retail_pct > self.retail_threshold:
            dominant = "retail"
        else:
            dominant = "mixed"

        # 量化风险
        is_quant_vul = quant_pen > 0.5 or turnover > self.quant_turnover_threshold
        if is_quant_vul:
            risk_flags.append("量化参与度高-打板易被收割")

        if dominant == "inst" and mcap > 200e8:
            risk_flags.append("机构重仓-不宜追板")

        # 适配性
        if dominant in ("retail", "mixed") and not is_quant_vul and mcap < 100e8:
            suitability = "理想标的: 散户为主+小盘+低量化渗透"
        elif dominant == "north" and nb_pct > 0.05:
            if risk_flags:
                suitability = "可观察: 北向持续流入但不宜追板"
            else:
                suitability = "可参与: 北向趋势+基本面支撑"
        elif is_quant_vul:
            suitability = "回避: 量化主导，短线胜率低"
        elif len(risk_flags) > 0:
            suitability = "谨慎: " + "; ".join(risk_flags)
        else:
            suitability = "中性: 需结合题材判断"

        return CapitalProfile(
            symbol=symbol, name=name,
            north_bound_pct=nb_pct,
            inst_holding_pct=inst_pct,
            retail_holding_pct=retail_pct,
            top10_concentration=top10,
            avg_turnover=turnover,
            recent_turnover=turnover,
            vol_20d=volume,
            quant_penetration=quant_pen,
            dominant_type=dominant,
            is_quant_vulnerable=is_quant_vul,
            suitability=suitability,
            risk_flags=risk_flags)


def quick_profile(row):
    """快速生成资金画像摘要"""
    analyzer = CapitalStructureAnalyzer()
    profile = analyzer.analyze(
        symbol=row.get("symbol", "000000"),
        name=row.get("name", "unknown"),
        row=row)
    return {
        "symbol": profile.symbol,
        "name": profile.name,
        "dominant": profile.dominant_type,
        "quant_pen": f"{profile.quant_penetration:.0%}",
        "quant_vul": profile.is_quant_vulnerable,
        "suitability": profile.suitability,
        "flags": profile.risk_flags,
    }


if __name__ == "__main__":
    analyzer = CapitalStructureAnalyzer()
    test_stocks = [
        {"symbol": "002036", "name": "联创电子",
         "north_bound_pct": 0.02, "inst_holding": 0.08,
         "turnover": 0.08, "mcap": 80e8, "close": 12.5, "volume": 5e7,
         "top10_pct": 0.35},
        {"symbol": "600519", "name": "贵州茅台",
         "north_bound_pct": 0.07, "inst_holding": 0.15,
         "turnover": 0.005, "mcap": 20000e8, "close": 1500, "volume": 2e6,
         "top10_pct": 0.65},
        {"symbol": "300750", "name": "某小票",
         "north_bound_pct": 0.01, "inst_holding": 0.05,
         "turnover": 0.22, "mcap": 35e8, "close": 8.0, "volume": 8e7,
         "top10_pct": 0.25},
    ]
    for ts in test_stocks:
        profile = analyzer.analyze(ts["symbol"], ts["name"], ts)
        print()
        print(f"{profile.name}({profile.symbol}):")
        print(f"  主力: {profile.dominant_type} | 量化: {profile.quant_penetration:.0%}")
        print(f"  适配: {profile.suitability}")
        if profile.risk_flags:
            print(f"  风险: {profile.risk_flags}")
