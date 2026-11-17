# -*- coding: utf-8 -*-
"""
===================================
資料源策略層 - 台股與美股專用包初始化
===================================

本包僅服務台灣股市（上市 .TW / 上櫃 .TWO 個股與 ETF）及美國股市（個股與 ETF）：
1. YfinanceFetcher：台美股核心日線、年線與基礎行情來源
2. TwInstitutionalFetcher：台股上市櫃三大法人買賣超資料源
3. FinnhubFetcher / AlphaVantageFetcher / LongbridgeFetcher：美股可選備援資料源
"""

from .base import BaseFetcher, DataFetcherManager
from .yfinance_fetcher import YfinanceFetcher
from .tw_institutional_fetcher import TwInstitutionalFetcher
from .longbridge_fetcher import LongbridgeFetcher
from .finnhub_fetcher import FinnhubFetcher
from .alphavantage_fetcher import AlphaVantageFetcher
from .us_index_mapping import (
    US_INDEX_MAPPING,
    get_us_index_yf_symbol,
    is_us_index_code,
    is_us_stock_code,
)


def is_hk_stock_code(code: str) -> bool:
    """Deprecated stub kept for legacy caller compatibility; non-TW/US markets are disabled."""
    return False


__all__ = [
    "BaseFetcher",
    "DataFetcherManager",
    "YfinanceFetcher",
    "TwInstitutionalFetcher",
    "LongbridgeFetcher",
    "FinnhubFetcher",
    "AlphaVantageFetcher",
    "is_us_index_code",
    "is_us_stock_code",
    "is_hk_stock_code",
    "get_us_index_yf_symbol",
    "US_INDEX_MAPPING",
]
