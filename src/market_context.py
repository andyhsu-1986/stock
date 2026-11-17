# -*- coding: utf-8 -*-
"""
Market context detection for LLM prompts (Taiwan & US Markets Only).

Detects whether a symbol belongs to the Taiwan market (`.TW` / `.TWO`, including
TW stocks and ETFs) or the US market (US stocks and ETFs), and provides concise
Traditional Chinese / English market rules to save LLM tokens.
"""

from __future__ import annotations

import re
from typing import Optional

from src.services.market_symbol_utils import get_suffix_market


def detect_market(stock_code: Optional[str]) -> str:
    """Detect market from stock code ('tw' or 'us')."""
    if not stock_code:
        return "tw"

    code = stock_code.strip().upper()
    suffix_market = get_suffix_market(code)
    if suffix_market == "tw" or code.endswith((".TW", ".TWO")):
        return "tw"

    if re.match(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$", code) or code.startswith("^"):
        return "us"

    return "tw"


_MARKET_ROLES = {
    "tw": {
        "zh": "台股（上市/上櫃個股與 ETF）",
        "en": "Taiwan stock & ETF",
    },
    "us": {
        "zh": "美股（NYSE/NASDAQ 個股與 ETF）",
        "en": "US stock & ETF",
    },
}

_MARKET_GUIDELINES = {
    "tw": {
        "zh": (
            "- 本次分析標的為 **台股**（台灣證券交易所上市 `.TW` 或櫃買中心上櫃 `.TWO`）。\n"
            "- 交易規則與分析重點：計價幣別為**新台幣（TWD）**；交易時段 09:00–13:30，漲跌幅限制為 **±10%**（ETF 無漲跌幅限制或依追蹤標的而定）。\n"
            "- 籌碼與基本面重點：關注**三大法人（外資、投信、自營商）買賣超**、月營收年增率、近 3 年營收 CAGR、毛利率斜率、本益比（PE）5 年位階與現金殖利率；台股 ETF 需特別檢核**即時折溢價率**（溢價 > 1.5% 時嚴禁追高）。"
        ),
        "en": (
            "- This analysis covers a **Taiwan security** (TWSE `.TW` or TPEx `.TWO`, priced in TWD).\n"
            "- Focus on the three institutional investor groups (Foreign, Investment Trust, Dealer), monthly revenue YoY, 3Y revenue CAGR, gross margin slope, 5Y PE band, dividend yield, and TWSE/TPEx ±10% daily price limits (for ETFs, check real-time NAV premium/discount > 1.5%)."
        ),
    },
    "us": {
        "zh": (
            "- 本次分析標的為 **美股**（NYSE / NASDAQ 上市個股或美股 ETF）。\n"
            "- 交易規則與分析重點：計價幣別為**美元（USD）**；無單日漲跌幅限制（設有市場熔斷機制），支援盤前與盤後交易。\n"
            "- 基本面與總經重點：關注聯準會（Fed）利率政策、美債殖利率、美元指數、企業季報財測（Guidance）；美股 ETF 重點評估 **240 日年線乖離率** 與 **歷史殖利率分位數**。"
        ),
        "en": (
            "- This analysis covers a **US security** (NYSE/NASDAQ stock or ETF, priced in USD).\n"
            "- US markets have no daily price limits (circuit breakers apply) and support pre/post-market trading. Consider Fed policy, US Treasury yields, earnings guidance, and for ETFs, 240-day MA bias and historical dividend yield percentile."
        ),
    },
}


def get_market_role(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific role description for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang in ("en", "ko") else "zh"
    return _MARKET_ROLES.get(market, _MARKET_ROLES["tw"])[lang_key]


def get_market_guidelines(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific analysis guidelines for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang in ("en", "ko") else "zh"
    return _MARKET_GUIDELINES.get(market, _MARKET_GUIDELINES["tw"])[lang_key]
