# -*- coding: utf-8 -*-
"""
===================================
台股全市場量化初篩模組 (TWSE / TPEx Batch Screener)
===================================

職責：
1. 串接台灣證交所（TWSE）與櫃買中心（TPEx）每日全市場收盤行情、三大法人買賣超與基本面估值批次 API。
2. 將每日全市場資料正規化並追加儲存至 `data/tw_daily_history.parquet`（自動滾動保留近 5 年資料）。
3. 支援冷啟動（Cold-Start）批次歷史回補，確保首次執行即可計算近 20 日均線與法人連續買超。
4. 結合「近 20 日均線多頭排列、成交量濾網、三大法人連續買超、基本面體質指標」計算總分並挑出前 20 名標的。
5. 提供與既有設定檔 `STOCK_LIST`（含美股 ETF、台股 ETF 及固定觀察清單）無縫去重合併之輔助介面。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# --- Official TWSE & TPEx Batch Endpoints ---
# 1. Latest-day OpenAPI / RWD batch endpoints (fast single-shot per market)
_TWSE_OPENAPI_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
_TWSE_OPENAPI_BWIBBU_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
_TWSE_OPENAPI_PUNISH = "https://openapi.twse.com.tw/v1/announcement/punish"
_TWSE_OPENAPI_TWT85U = "https://openapi.twse.com.tw/v1/exchangeReport/TWT85U"
_TWSE_RWD_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
_TWSE_RWD_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
_TWSE_RWD_BWIBBU_D = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"

_TPEX_OPENAPI_DAILY_CLOSE = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
_TPEX_OPENAPI_3INSTI = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
_TPEX_OPENAPI_PE = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
_TPEX_OPENAPI_DISPOSAL = "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"
_TPEX_OPENAPI_CMODE = "https://www.tpex.org.tw/openapi/v1/tpex_cmode"
_TPEX_WEB_DAILY_QUOTES = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
_TPEX_WEB_DAILY_TRADE = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_DEFAULT_PARQUET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tw_daily_history.parquet"
)

# Common stocks: 4 digits not starting with 0 (1101..9999)
_COMMON_STOCK_RE = re.compile(r"^[1-9]\d{3}$")
# Pure numeric Taiwan ETFs: 4-6 digits starting with 00 (0050, 0056, 00878, 006208, etc.)
_NUMERIC_ETF_RE = re.compile(r"^00\d{2,4}$")

PARQUET_COLUMNS = [
    "date",
    "code",
    "symbol",
    "name",
    "market",
    "is_etf",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_value",
    "foreign_net",
    "trust_net",
    "dealer_net",
    "inst_total_net",
    "pe_ratio",
    "dividend_yield",
    "pb_ratio",
]


def is_eligible_tw_code(code: Any) -> bool:
    """Return True if ``code`` is a 4-digit TWSE/TPEx common stock or 4-6 digit pure-numeric ETF."""
    text = str(code or "").strip().upper()
    if not text:
        return False
    return bool(_COMMON_STOCK_RE.match(text) or _NUMERIC_ETF_RE.match(text))


def is_tw_etf_code(code: Any) -> bool:
    """Return True if ``code`` is a pure-numeric Taiwan ETF code (e.g. 0050, 00878, 006208)."""
    text = str(code or "").strip().upper()
    return bool(_NUMERIC_ETF_RE.match(text))


def to_yahoo_tw_symbol(code: str, market: str) -> str:
    """Convert bare Taiwan security code and market ('TWSE' or 'TPEX') to Yahoo suffix symbol."""
    clean_code = str(code or "").strip().upper()
    mkt = str(market or "").strip().upper()
    suffix = "TWO" if mkt == "TPEX" else "TW"
    return f"{clean_code}.{suffix}"


def minguo_or_ad_to_iso_date(raw_date: Any) -> Optional[str]:
    """Convert ROC date (``1151116`` or ``115/11/16``) or AD date (``20261116`` or ``2026-11-16``) to ``YYYY-MM-DD``."""
    text = str(raw_date or "").strip()
    if not text:
        return None
    clean = text.replace("-", "").replace("/", "").replace(".", "").strip()
    if not clean.isdigit():
        return None
    if len(clean) == 7:
        year = int(clean[:3]) + 1911
        month = int(clean[3:5])
        day = int(clean[5:7])
        return f"{year:04d}-{month:02d}-{day:02d}"
    if len(clean) == 8:
        year = int(clean[:4])
        month = int(clean[4:6])
        day = int(clean[6:8])
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Parse comma-formatted or string numeric value to float safely."""
    if value is None:
        return default
    try:
        text = str(value).replace(",", "").replace(" ", "").strip()
        if text in ("", "-", "--", "---", "—", "N/A", "NaN", "None"):
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    """Parse comma-formatted or string integer value safely."""
    if value is None:
        return default
    try:
        text = str(value).replace(",", "").replace(" ", "").strip()
        if text in ("", "-", "--", "---", "—", "N/A", "NaN", "None"):
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _canonicalize_symbol(token: str) -> str:
    """Canonicalize symbol using data_provider when available, with safe uppercase fallback."""
    clean = str(token or "").strip()
    if not clean:
        return ""
    try:
        from data_provider.base import canonical_stock_code

        return canonical_stock_code(clean) or clean.upper()
    except Exception:
        return clean.upper()


def merge_watchlist_with_screened(
    fixed_watchlist: Sequence[str],
    screened_symbols: Sequence[str],
) -> List[str]:
    """Merge user's fixed watchlist (US ETFs, TW ETFs, core stocks) with screened TW symbols, preserving order and deduplicating."""
    merged: List[str] = []
    seen_canonical: set[str] = set()

    for raw in list(fixed_watchlist or []) + list(screened_symbols or []):
        token = str(raw or "").strip()
        if not token:
            continue
        canon = _canonicalize_symbol(token)
        if canon not in seen_canonical:
            seen_canonical.add(canon)
            merged.append(canon)
    return merged


class TWMarketScreener:
    """
    Taiwan Full-Market Post-Market Quantitative Screener.

    Fetches full-market daily closing quotes, institutional net buy/sell, and fundamental
    valuation metrics from official TWSE & TPEx batch APIs, caches up to 5 years of history
    in ``data/tw_daily_history.parquet``, and ranks the universe (~2,150 stocks & ETFs)
    to select the top N candidates.
    """

    def __init__(
        self,
        parquet_path: Optional[Path] = None,
        *,
        top_n: int = 20,
        retention_years: int = 5,
        min_history_days: int = 20,
        bootstrap_days: int = 22,
        min_avg_volume_5d_lots: int = 1000,
        min_close_price: float = 10.0,
        restricted_codes: Optional[Iterable[str]] = None,
        request_timeout: int = 15,
        min_request_interval: float = 1.2,
    ) -> None:
        self.parquet_path = Path(parquet_path) if parquet_path else _DEFAULT_PARQUET_PATH
        self.top_n = max(1, int(top_n))
        self.retention_years = max(1, int(retention_years))
        self.min_history_days = max(5, int(min_history_days))
        self.bootstrap_days = max(self.min_history_days, int(bootstrap_days))
        self.min_avg_volume_5d_lots = max(1, int(min_avg_volume_5d_lots))
        self.min_avg_volume_5d_shares = self.min_avg_volume_5d_lots * 1000
        self.min_close_price = float(min_close_price)
        self.restricted_codes: set[str] = {
            str(c).strip().upper().split(".")[0]
            for c in (restricted_codes or [])
            if str(c).strip()
        }
        self.request_timeout = max(5, int(request_timeout))
        self.min_request_interval = max(0.0, float(min_request_interval))
        self._throttle_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/json, text/plain, */*",
            }
        )

    # ------------------------------------------------------------------ HTTP with Retry & Timeout
    def _throttle(self) -> None:
        if self.min_request_interval <= 0:
            return
        with self._throttle_lock:
            elapsed = time.time() - self._last_request_ts
            wait_sec = self.min_request_interval - elapsed
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_request_ts = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
        reraise=True,
    )
    def _request_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform HTTP GET with timeout, polite throttling, and exponential backoff retry."""
        self._throttle()
        response = self._session.get(url, params=params, timeout=self.request_timeout)
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(f"Upstream HTTP {response.status_code} from {url}")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ Parquet Storage (5-Year Rolling)
    def _load_history(self) -> pd.DataFrame:
        if not self.parquet_path.exists():
            return pd.DataFrame(columns=PARQUET_COLUMNS)
        try:
            df = pd.read_parquet(self.parquet_path)
            for col in PARQUET_COLUMNS:
                if col not in df.columns:
                    df[col] = 0 if col in ("volume", "foreign_net", "trust_net", "dealer_net", "inst_total_net") else ""
            return df[PARQUET_COLUMNS]
        except Exception as exc:
            logger.warning("[TWMarketScreener] 讀取本地 Parquet 快取失敗，將重新建立: %s", exc)
            return pd.DataFrame(columns=PARQUET_COLUMNS)

    def _merge_and_prune_history(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        reference_date: Optional[date] = None,
    ) -> pd.DataFrame:
        ref = reference_date or date.today()
        cutoff_str = (ref - timedelta(days=365 * self.retention_years)).strftime("%Y-%m-%d")

        frames = [df for df in (existing_df, new_df) if df is not None and not df.empty]
        if not frames:
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = combined["date"].astype(str)
        combined["symbol"] = combined["symbol"].astype(str)
        combined = combined[combined["date"] >= cutoff_str]
        combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
        combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        return combined[PARQUET_COLUMNS]

    def _save_history(self, df: pd.DataFrame, reference_date: Optional[date] = None) -> None:
        pruned = self._merge_and_prune_history(
            pd.DataFrame(columns=PARQUET_COLUMNS), df, reference_date=reference_date
        )
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.parquet_path.with_suffix(".parquet.tmp")
        pruned.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, self.parquet_path)

    # ------------------------------------------------------------------ Latest-Day Batch Fetchers
    def fetch_restricted_codes(self) -> set[str]:
        """
        Fetch current Taiwan disposition stocks (處置股) and full-cash delivery / altered trading
        stocks (全額交割股 / 變更交易方法) from official TWSE & TPEx OpenAPI endpoints.
        """
        codes: set[str] = set(self.restricted_codes)
        endpoints = (
            ("TWSE 處置股", _TWSE_OPENAPI_PUNISH, "Code"),
            ("TWSE 全額交割/變更交易股", _TWSE_OPENAPI_TWT85U, "Code"),
            ("TPEx 處置股", _TPEX_OPENAPI_DISPOSAL, "SecuritiesCompanyCode"),
            ("TPEx 全額交割/變更交易股", _TPEX_OPENAPI_CMODE, "SecuritiesCompanyCode"),
        )
        for label, url, key in endpoints:
            try:
                payload = self._request_json(url)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            c = str(item.get(key) or "").strip().upper()
                            if c:
                                codes.add(c)
            except Exception as exc:
                logger.warning("[TWMarketScreener] %s 名單抓取失敗 (Fail-Open): %s", label, exc)
        self.restricted_codes = codes
        return codes

    def fetch_latest_market_snapshot(self) -> pd.DataFrame:
        """Fetch latest full-market closing quotes, 3-institutional net buy/sell, and PE/PB/Yield for TWSE + TPEx."""
        self.fetch_restricted_codes()
        twse_df = self._fetch_latest_twse_batch()
        tpex_df = self._fetch_latest_tpex_batch()
        frames = [df for df in (twse_df, tpex_df) if not df.empty]
        if not frames:
            return pd.DataFrame(columns=PARQUET_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)[PARQUET_COLUMNS]
        if self.restricted_codes:
            combined = combined[~combined["code"].astype(str).str.upper().isin(self.restricted_codes)].reset_index(drop=True)
        return combined

    def _fetch_latest_twse_batch(self) -> pd.DataFrame:
        quotes_raw = self._request_json(_TWSE_OPENAPI_STOCK_DAY_ALL)
        if not isinstance(quotes_raw, list) or not quotes_raw:
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        # Parse valuation map {code: (pe, dy, pb)}
        val_map: Dict[str, Tuple[float, float, float]] = {}
        try:
            bwibbu_raw = self._request_json(_TWSE_OPENAPI_BWIBBU_ALL)
            if isinstance(bwibbu_raw, list):
                for item in bwibbu_raw:
                    c = str(item.get("Code", "")).strip()
                    if c:
                        val_map[c] = (
                            _parse_float(item.get("PEratio")),
                            _parse_float(item.get("DividendYield")),
                            _parse_float(item.get("PBratio")),
                        )
        except Exception as exc:
            logger.warning("[TWMarketScreener] TWSE BWIBBU_ALL 估值抓取失敗 (Fail-Open): %s", exc)

        #Detect trading date from quotes
        sample_date_iso = minguo_or_ad_to_iso_date(quotes_raw[0].get("Date"))
        ad_date_compact = sample_date_iso.replace("-", "") if sample_date_iso else None

        # Parse institutional map {code: (foreign, trust, dealer, total)}
        inst_map = self._fetch_twse_t86_map(ad_date_compact)

        rows: List[Dict[str, Any]] = []
        for item in quotes_raw:
            code = str(item.get("Code", "")).strip().upper()
            if not is_eligible_tw_code(code):
                continue
            row_date = minguo_or_ad_to_iso_date(item.get("Date")) or sample_date_iso
            if not row_date:
                continue
            close_price = _parse_float(item.get("ClosingPrice"))
            if close_price <= 0:
                continue
            open_price = _parse_float(item.get("OpeningPrice"), default=close_price)
            high_price = _parse_float(item.get("HighestPrice"), default=close_price)
            low_price = _parse_float(item.get("LowestPrice"), default=close_price)
            volume = _parse_int(item.get("TradeVolume"))
            turnover = _parse_float(item.get("TradeValue"))
            foreign, trust, dealer, inst_total = inst_map.get(code, (0, 0, 0, 0))
            pe, dy, pb = val_map.get(code, (0.0, 0.0, 0.0))

            rows.append(
                {
                    "date": row_date,
                    "code": code,
                    "symbol": to_yahoo_tw_symbol(code, "TWSE"),
                    "name": str(item.get("Name", "")).strip(),
                    "market": "TWSE",
                    "is_etf": is_tw_etf_code(code),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "turnover_value": turnover,
                    "foreign_net": foreign,
                    "trust_net": trust,
                    "dealer_net": dealer,
                    "inst_total_net": inst_total,
                    "pe_ratio": pe,
                    "dividend_yield": dy,
                    "pb_ratio": pb,
                }
            )
        return pd.DataFrame(rows, columns=PARQUET_COLUMNS) if rows else pd.DataFrame(columns=PARQUET_COLUMNS)

    def _fetch_twse_t86_map(self, ad_date_compact: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]:
        """Fetch TWSE T86 institutional net buy/sell map {code: (foreign, trust, dealer, total)}."""
        inst_map: Dict[str, Tuple[int, int, int, int]] = {}
        try:
            params: Dict[str, str] = {"response": "json", "selectType": "ALLBUT0999"}
            if ad_date_compact:
                params["date"] = ad_date_compact
            payload = self._request_json(_TWSE_RWD_T86, params=params)
            if not isinstance(payload, dict) or payload.get("stat") != "OK":
                return inst_map
            fields = payload.get("fields") or []
            data_rows = payload.get("data") or []
            if not isinstance(fields, list) or not isinstance(data_rows, list):
                return inst_map

            def _find_col(keyword: str, fallback: int) -> int:
                for idx, f in enumerate(fields):
                    if keyword in str(f):
                        return idx
                return fallback

            idx_code = _find_col("證券代號", 0)
            idx_foreign = _find_col("外陸資買賣超股數(不含外資自營商)", 4)
            idx_trust = _find_col("投信買賣超股數", 10)
            idx_dealer = _find_col("自營商買賣超股數", 11)
            idx_total = _find_col("三大法人買賣超股數", len(fields) - 1)

            for r in data_rows:
                if not isinstance(r, (list, tuple)) or len(r) <= max(idx_code, idx_foreign, idx_trust, idx_dealer, idx_total):
                    continue
                code = str(r[idx_code]).strip().upper()
                if not is_eligible_tw_code(code):
                    continue
                f_net = _parse_int(r[idx_foreign])
                t_net = _parse_int(r[idx_trust])
                d_net = _parse_int(r[idx_dealer])
                tot_net = _parse_int(r[idx_total], default=f_net + t_net + d_net)
                inst_map[code] = (f_net, t_net, d_net, tot_net)
        except Exception as exc:
            logger.warning("[TWMarketScreener] TWSE T86 法人資料抓取失敗 (Fail-Open): %s", exc)
        return inst_map

    def _fetch_latest_tpex_batch(self) -> pd.DataFrame:
        quotes_raw = self._request_json(_TPEX_OPENAPI_DAILY_CLOSE)
        if not isinstance(quotes_raw, list) or not quotes_raw:
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        val_map: Dict[str, Tuple[float, float, float]] = {}
        try:
            pe_raw = self._request_json(_TPEX_OPENAPI_PE)
            if isinstance(pe_raw, list):
                for item in pe_raw:
                    c = str(item.get("SecuritiesCompanyCode", "")).strip().upper()
                    if c:
                        val_map[c] = (
                            _parse_float(item.get("PriceEarningRatio")),
                            _parse_float(item.get("YieldRatio")),
                            _parse_float(item.get("PriceBookRatio")),
                        )
        except Exception as exc:
            logger.warning("[TWMarketScreener] TPEx 本益比估值抓取失敗 (Fail-Open): %s", exc)

        inst_map: Dict[str, Tuple[int, int, int, int]] = {}
        try:
            inst_raw = self._request_json(_TPEX_OPENAPI_3INSTI)
            if isinstance(inst_raw, list):
                for item in inst_raw:
                    c = str(item.get("SecuritiesCompanyCode", "")).strip().upper()
                    if not is_eligible_tw_code(c):
                        continue
                    f_net = _parse_int(
                        item.get(
                            "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Difference"
                        )
                    )
                    t_net = _parse_int(item.get("SecuritiesInvestmentTrustCompanies-Difference"))
                    d_net = _parse_int(item.get("Dealers-Difference"))
                    tot_net = _parse_int(item.get("TotalDifference"), default=f_net + t_net + d_net)
                    inst_map[c] = (f_net, t_net, d_net, tot_net)
        except Exception as exc:
            logger.warning("[TWMarketScreener] TPEx 三大法人抓取失敗 (Fail-Open): %s", exc)

        sample_date_iso = minguo_or_ad_to_iso_date(quotes_raw[0].get("Date"))
        rows: List[Dict[str, Any]] = []
        for item in quotes_raw:
            code = str(item.get("SecuritiesCompanyCode", "")).strip().upper()
            if not is_eligible_tw_code(code):
                continue
            row_date = minguo_or_ad_to_iso_date(item.get("Date")) or sample_date_iso
            if not row_date:
                continue
            close_price = _parse_float(item.get("Close"))
            if close_price <= 0:
                continue
            open_price = _parse_float(item.get("Open"), default=close_price)
            high_price = _parse_float(item.get("High"), default=close_price)
            low_price = _parse_float(item.get("Low"), default=close_price)
            volume = _parse_int(item.get("TradingShares"))
            turnover = _parse_float(item.get("TransactionAmount"))
            foreign, trust, dealer, inst_total = inst_map.get(code, (0, 0, 0, 0))
            pe, dy, pb = val_map.get(code, (0.0, 0.0, 0.0))

            rows.append(
                {
                    "date": row_date,
                    "code": code,
                    "symbol": to_yahoo_tw_symbol(code, "TPEX"),
                    "name": str(item.get("CompanyName", "")).strip(),
                    "market": "TPEX",
                    "is_etf": is_tw_etf_code(code),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "turnover_value": turnover,
                    "foreign_net": foreign,
                    "trust_net": trust,
                    "dealer_net": dealer,
                    "inst_total_net": inst_total,
                    "pe_ratio": pe,
                    "dividend_yield": dy,
                    "pb_ratio": pb,
                }
            )
        return pd.DataFrame(rows, columns=PARQUET_COLUMNS) if rows else pd.DataFrame(columns=PARQUET_COLUMNS)

    # ------------------------------------------------------------------ Historical Single-Date Batch Fetchers (Cold-Start Backfill)
    def fetch_historical_date_snapshot(self, target_date: date) -> pd.DataFrame:
        """Fetch TWSE + TPEx full-market closing quotes & institutional flows for a specific historical date."""
        twse_df = self._fetch_twse_historical_date(target_date)
        tpex_df = self._fetch_tpex_historical_date(target_date)
        frames = [df for df in (twse_df, tpex_df) if not df.empty]
        if not frames:
            return pd.DataFrame(columns=PARQUET_COLUMNS)
        return pd.concat(frames, ignore_index=True)[PARQUET_COLUMNS]

    def _fetch_twse_historical_date(self, target_date: date) -> pd.DataFrame:
        ad_compact = target_date.strftime("%Y%m%d")
        iso_date = target_date.strftime("%Y-%m-%d")
        try:
            payload = self._request_json(
                _TWSE_RWD_MI_INDEX,
                params={"response": "json", "date": ad_compact, "type": "ALLBUT0999"},
            )
        except Exception as exc:
            logger.warning("[TWMarketScreener] TWSE MI_INDEX (%s) 抓取失敗: %s", iso_date, exc)
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        if not isinstance(payload, dict) or payload.get("stat") != "OK":
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        quote_table = None
        for table in payload.get("tables") or []:
            title = str(table.get("title") or "")
            if "每日收盤行情" in title and table.get("data"):
                quote_table = table
                break
        if quote_table is None:
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        inst_map = self._fetch_twse_t86_map(ad_compact)
        rows: List[Dict[str, Any]] = []
        for r in quote_table.get("data") or []:
            if not isinstance(r, (list, tuple)) or len(r) < 16:
                continue
            code = str(r[0]).strip().upper()
            if not is_eligible_tw_code(code):
                continue
            close_price = _parse_float(r[8])
            if close_price <= 0:
                continue
            open_price = _parse_float(r[5], default=close_price)
            high_price = _parse_float(r[6], default=close_price)
            low_price = _parse_float(r[7], default=close_price)
            volume = _parse_int(r[2])
            turnover = _parse_float(r[4])
            pe = _parse_float(r[15])
            foreign, trust, dealer, inst_total = inst_map.get(code, (0, 0, 0, 0))

            rows.append(
                {
                    "date": iso_date,
                    "code": code,
                    "symbol": to_yahoo_tw_symbol(code, "TWSE"),
                    "name": str(r[1]).strip(),
                    "market": "TWSE",
                    "is_etf": is_tw_etf_code(code),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "turnover_value": turnover,
                    "foreign_net": foreign,
                    "trust_net": trust,
                    "dealer_net": dealer,
                    "inst_total_net": inst_total,
                    "pe_ratio": pe,
                    "dividend_yield": 0.0,
                    "pb_ratio": 0.0,
                }
            )
        return pd.DataFrame(rows, columns=PARQUET_COLUMNS) if rows else pd.DataFrame(columns=PARQUET_COLUMNS)

    def _fetch_tpex_historical_date(self, target_date: date) -> pd.DataFrame:
        slash_date = target_date.strftime("%Y/%m/%d")
        iso_date = target_date.strftime("%Y-%m-%d")
        try:
            quote_payload = self._request_json(
                _TPEX_WEB_DAILY_QUOTES,
                params={"date": slash_date, "id": "", "response": "json"},
            )
        except Exception as exc:
            logger.warning("[TWMarketScreener] TPEx dailyQuotes (%s) 抓取失敗: %s", iso_date, exc)
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        tables = (quote_payload or {}).get("tables") if isinstance(quote_payload, dict) else None
        if not tables or not isinstance(tables, list) or not tables[0].get("data"):
            return pd.DataFrame(columns=PARQUET_COLUMNS)

        inst_map: Dict[str, Tuple[int, int, int, int]] = {}
        try:
            inst_payload = self._request_json(
                _TPEX_WEB_DAILY_TRADE,
                params={"type": "Daily", "sect": "AL", "date": slash_date, "id": "", "response": "json"},
            )
            inst_tables = (inst_payload or {}).get("tables") if isinstance(inst_payload, dict) else None
            if inst_tables and isinstance(inst_tables, list) and inst_tables[0].get("data"):
                for r in inst_tables[0]["data"]:
                    if not isinstance(r, (list, tuple)) or len(r) < 24:
                        continue
                    c = str(r[0]).strip().upper()
                    if not is_eligible_tw_code(c):
                        continue
                    f_net = _parse_int(r[4])
                    t_net = _parse_int(r[13])
                    d_net = _parse_int(r[22])
                    tot_net = _parse_int(r[23], default=f_net + t_net + d_net)
                    inst_map[c] = (f_net, t_net, d_net, tot_net)
        except Exception as exc:
            logger.warning("[TWMarketScreener] TPEx dailyTrade (%s) 抓取失敗 (Fail-Open): %s", iso_date, exc)

        rows: List[Dict[str, Any]] = []
        for r in tables[0].get("data") or []:
            if not isinstance(r, (list, tuple)) or len(r) < 10:
                continue
            code = str(r[0]).strip().upper()
            if not is_eligible_tw_code(code):
                continue
            close_price = _parse_float(r[2])
            if close_price <= 0:
                continue
            open_price = _parse_float(r[4], default=close_price)
            high_price = _parse_float(r[5], default=close_price)
            low_price = _parse_float(r[6], default=close_price)
            volume = _parse_int(r[8])
            turnover = _parse_float(r[9])
            foreign, trust, dealer, inst_total = inst_map.get(code, (0, 0, 0, 0))

            rows.append(
                {
                    "date": iso_date,
                    "code": code,
                    "symbol": to_yahoo_tw_symbol(code, "TPEX"),
                    "name": str(r[1]).strip(),
                    "market": "TPEX",
                    "is_etf": is_tw_etf_code(code),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "turnover_value": turnover,
                    "foreign_net": foreign,
                    "trust_net": trust,
                    "dealer_net": dealer,
                    "inst_total_net": inst_total,
                    "pe_ratio": 0.0,
                    "dividend_yield": 0.0,
                    "pb_ratio": 0.0,
                }
            )
        return pd.DataFrame(rows, columns=PARQUET_COLUMNS) if rows else pd.DataFrame(columns=PARQUET_COLUMNS)

    def _get_recent_tw_trading_dates(self, end_date: date, count: int) -> List[date]:
        """Return up to ``count`` recent XTAI trading dates strictly before ``end_date``."""
        try:
            import exchange_calendars as xcals

            cal = xcals.get_calendar("XTAI")
            start_candidate = end_date - timedelta(days=max(count * 2 + 15, 45))
            sessions = cal.sessions_in_range(
                pd.Timestamp(start_candidate),
                pd.Timestamp(end_date - timedelta(days=1)),
            )
            dates = [ts.date() for ts in sessions]
            return dates[-count:]
        except Exception:
            # Fallback to weekdays if exchange_calendars fails or is out of range
            dates = []
            cur = end_date - timedelta(days=1)
            while len(dates) < count:
                if cur.weekday() < 5:
                    dates.append(cur)
                cur -= timedelta(days=1)
            dates.reverse()
            return dates

    # ------------------------------------------------------------------ Update History & Cold-Start Bootstrap
    def update_daily_history(self) -> pd.DataFrame:
        """
        Update ``data/tw_daily_history.parquet`` with the latest TWSE + TPEx batch data.
        If local history has fewer than ``min_history_days`` distinct dates, automatically
        backfills recent trading days so MA20 and consecutive institutional buy/sell are accurate.
        """
        history_df = self._load_history()
        latest_df = self.fetch_latest_market_snapshot()
        if not latest_df.empty:
            history_df = self._merge_and_prune_history(history_df, latest_df)
            self._save_history(history_df)
            latest_date_str = str(latest_df["date"].max())
            logger.info(
                "[TWMarketScreener] 已更新最新交易日 (%s) 全市場行情與法人資料，共 %d 檔標的",
                latest_date_str,
                len(latest_df),
            )
        else:
            logger.warning("[TWMarketScreener] 最新交易日批次行情抓取為空，將沿用本地 Parquet 快取")

        existing_dates = sorted(history_df["date"].dropna().unique().tolist()) if not history_df.empty else []
        if len(existing_dates) < self.min_history_days and existing_dates:
            latest_dt = datetime.strptime(existing_dates[-1], "%Y-%m-%d").date()
            needed = self.bootstrap_days - len(existing_dates)
            candidate_dates = self._get_recent_tw_trading_dates(
                latest_dt, count=self.bootstrap_days + 5
            )
            missing_dates = [
                d for d in candidate_dates if d.strftime("%Y-%m-%d") not in set(existing_dates)
            ][-max(needed, 0):]

            if missing_dates:
                logger.info(
                    "[TWMarketScreener] 本地歷史交易日僅 %d 天（不足 %d 天），啟動冷啟動批次回補最近 %d 個交易日...",
                    len(existing_dates),
                    self.min_history_days,
                    len(missing_dates),
                )
                backfill_frames: List[pd.DataFrame] = []
                for idx, d in enumerate(missing_dates, start=1):
                    snap = self.fetch_historical_date_snapshot(d)
                    if not snap.empty:
                        backfill_frames.append(snap)
                        logger.info(
                            "[TWMarketScreener] 冷啟動回補進度 [%d/%d]: %s (%d 檔)",
                            idx,
                            len(missing_dates),
                            d.strftime("%Y-%m-%d"),
                            len(snap),
                        )
                if backfill_frames:
                    backfill_all = pd.concat(backfill_frames, ignore_index=True)
                    # Note: keep latest_df after backfill so latest day's BWIBBU PE/Yield/PB is preserved
                    history_df = self._merge_and_prune_history(backfill_all, history_df)
                    self._save_history(history_df)

        return history_df

    # ------------------------------------------------------------------ 4-Dimension Quantitative Scoring Engine
    @staticmethod
    def _consecutive_positive_days(values: np.ndarray) -> int:
        count = 0
        for val in reversed(values):
            if val > 0:
                count += 1
            else:
                break
        return count

    def score_universe(
        self,
        history_df: pd.DataFrame,
        restricted_codes: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute 4-dimension quantitative scores across the entire Taiwan universe and return the top N DataFrame.

        Beginner Safety & Liquidity Filters (Hard Exclusions):
        - Close price < 10.0 TWD (排除股價低於 10 元雞蛋水餃股)
        - 5-day average volume < 1,000 lots / 1,000,000 shares (排除近 5 日均量小於 1,000 張)
        - Disposition stocks (處置股) & Full-cash delivery stocks (全額交割股 / 變更交易方法)

        Dimensions (Total 100 pts):
        1. Trend & Moving Average Alignment (MA5 > MA10 > MA20, MA20 slope, bias control): 30 pts
        2. Volume & Liquidity Structure (Volume filter >= 1,000 lots, vol_ratio_20 expansion): 20 pts
        3. Institutional Consecutive Buying (Consecutive net buy days, 5d net buy ratio, trust/foreign sync): 30 pts
        4. Fundamental Quality (PE/Yield/PB for stocks; Yield/Liquidity/Stability for ETFs): 20 pts
        """
        if history_df is None or history_df.empty:
            return pd.DataFrame()

        active_restricted: set[str] = set(self.restricted_codes)
        if restricted_codes is not None:
            active_restricted.update(
                str(c).strip().upper().split(".")[0]
                for c in restricted_codes
                if str(c).strip()
            )

        df = history_df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)
        df["turnover_value"] = pd.to_numeric(df["turnover_value"], errors="coerce").fillna(0.0)
        df["foreign_net"] = pd.to_numeric(df["foreign_net"], errors="coerce").fillna(0).astype(np.int64)
        df["trust_net"] = pd.to_numeric(df["trust_net"], errors="coerce").fillna(0).astype(np.int64)
        df["dealer_net"] = pd.to_numeric(df["dealer_net"], errors="coerce").fillna(0).astype(np.int64)
        df["inst_total_net"] = pd.to_numeric(df["inst_total_net"], errors="coerce").fillna(0).astype(np.int64)
        df["pe_ratio"] = pd.to_numeric(df["pe_ratio"], errors="coerce").fillna(0.0)
        df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce").fillna(0.0)
        df["pb_ratio"] = pd.to_numeric(df["pb_ratio"], errors="coerce").fillna(0.0)

        df = df[df["code"].map(is_eligible_tw_code)]
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        latest_market_date = str(df["date"].max())

        records: List[Dict[str, Any]] = []
        for symbol, grp in df.groupby("symbol", sort=False):
            tail = grp.tail(25)
            last_row = tail.iloc[-1]
            # Must be active on the latest market date
            if str(last_row["date"]) != latest_market_date:
                continue

            code_str = str(last_row["code"]).strip().upper()
            name_str = str(last_row.get("name") or "").strip()

            # Hard Safety Filter 1: Exclude Disposition (處置股) & Full-Cash Delivery (全額交割股)
            if code_str in active_restricted or str(symbol).strip().upper() in active_restricted:
                continue
            is_flagged_restricted = False
            for flag_col in ("is_disposition", "is_full_cash_delivery", "is_restricted"):
                if flag_col in tail.columns:
                    val = last_row.get(flag_col)
                    if bool(val) and str(val).strip().lower() not in ("false", "0", "none", "nan", ""):
                        is_flagged_restricted = True
                        break
            if is_flagged_restricted or any( kw in name_str for kw in ("處置", "全額交割", "變更交易")):
                continue

            # Hard Safety Filter 2: Exclude penny stocks with close price < 10.0 TWD
            close = float(last_row["close"])
            if close < self.min_close_price:
                continue

            closes = tail["close"].to_numpy(dtype=np.float64)
            vols = tail["volume"].to_numpy(dtype=np.float64)
            inst_totals = tail["inst_total_net"].to_numpy(dtype=np.float64)
            trust_nets = tail["trust_net"].to_numpy(dtype=np.float64)
            foreign_nets = tail["foreign_net"].to_numpy(dtype=np.float64)

            n_bars = len(closes)
            ma5 = float(np.mean(closes[-min(5, n_bars):]))
            ma10 = float(np.mean(closes[-min(10, n_bars):]))
            ma20 = float(np.mean(closes[-min(20, n_bars):]))
            prev_ma20 = (
                float(np.mean(closes[-min(21, n_bars):-1]))
                if n_bars >= 2
                else ma20
            )
            vol_ma5 = float(np.mean(vols[-min(5, n_bars):]))
            vol_ma20 = float(np.mean(vols[-min(20, n_bars):]))

            # Hard Liquidity Filter 3: 5-day avg volume >= 1,000 lots (1,000,000 shares) & turnover >= 10M TWD
            latest_vol = float(last_row["volume"])
            latest_turnover = float(last_row["turnover_value"])
            if latest_turnover <= 0:
                latest_turnover = close * latest_vol
            if vol_ma5 < self.min_avg_volume_5d_shares or latest_turnover < 10_000_000:
                continue

            # Moving Average & Trend metrics
            is_bullish_aligned = bool(close > ma20 and ma5 > ma10 and ma10 > ma20)
            ma20_rising = bool(ma20 >= prev_ma20)
            bias_20 = ((close - ma20) / ma20 * 100.0) if ma20 > 0 else 0.0
            base_20_close = float(closes[-min(20, n_bars)])
            ret_20 = ((close / base_20_close - 1.0) * 100.0) if base_20_close > 0 else 0.0

            # Exclude stocks below MA20 or over-extended (> 18% above MA20)
            if close <= ma20 or bias_20 > 18.0:
                continue

            # 1. Trend Score (0 - 30)
            trend_score = 0.0
            if is_bullish_aligned:
                trend_score += 15.0
            elif close > ma5 and ma5 > ma20:
                trend_score += 9.0
            else:
                trend_score += 4.0

            if ma20_rising:
                trend_score += 5.0
            if 0.0 < ret_20 <= 25.0:
                trend_score += min(5.0, (ret_20 / 15.0) * 5.0)
            elif ret_20 > 25.0:
                trend_score += 3.0
            if 0.5 <= bias_20 <= 8.0:
                trend_score += 5.0
            elif 8.0 < bias_20 <= 15.0:
                trend_score += 2.5
            else:
                trend_score += 1.0

            # 2. Volume Score (0 - 20)
            vol_ratio_20 = (latest_vol / vol_ma20) if vol_ma20 > 0 else 1.0
            volume_score = 0.0
            if 1.1 <= vol_ratio_20 <= 3.5:
                volume_score += 12.0
            elif 0.8 <= vol_ratio_20 < 1.1:
                volume_score += 7.0
            elif vol_ratio_20 > 3.5:
                volume_score += 5.0
            if vol_ma5 >= vol_ma20:
                volume_score += 5.0
            if close >= float(last_row["open"]):
                volume_score += 3.0

            # 3. Institutional Score (0 - 30)
            inst_consec = self._consecutive_positive_days(inst_totals)
            trust_consec = self._consecutive_positive_days(trust_nets)
            foreign_consec = self._consecutive_positive_days(foreign_nets)
            inst_5d_net = float(np.sum(inst_totals[-min(5, n_bars):]))
            denom_5d_vol = max(vol_ma20 * min(5, n_bars), 1.0)
            inst_5d_ratio = inst_5d_net / denom_5d_vol

            inst_score = 0.0
            inst_score += min(inst_consec, 5) / 5.0 * 14.0
            if inst_5d_ratio > 0:
                inst_score += min(10.0, (inst_5d_ratio / 0.15) * 10.0)
            if trust_consec >= 2 and foreign_consec >= 1:
                inst_score += 6.0
            elif trust_consec >= 1 or foreign_consec >= 2:
                inst_score += 3.5

            # 4. Fundamental Score (0 - 20)
            is_etf = bool(last_row["is_etf"])
            pe = float(last_row["pe_ratio"])
            dy = float(last_row["dividend_yield"])
            pb = float(last_row["pb_ratio"])

            fundamental_score = 0.0
            if is_etf:
                fundamental_score += 10.0
                if dy >= 3.5:
                    fundamental_score += 6.0
                elif dy > 0:
                    fundamental_score += 3.0
                if latest_turnover >= 100_000_000:
                    fundamental_score += 4.0
                elif latest_turnover >= 30_000_000:
                    fundamental_score += 2.0
            else:
                if 0.0 < pe <= 20.0:
                    fundamental_score += 9.0
                elif 20.0 < pe <= 35.0:
                    fundamental_score += 6.5
                elif 35.0 < pe <= 60.0:
                    fundamental_score += 3.5

                if dy >= 4.0:
                    fundamental_score += 7.0
                elif dy >= 2.0:
                    fundamental_score += 4.5
                elif dy > 0:
                    fundamental_score += 2.0

                if 0.0 < pb <= 3.5:
                    fundamental_score += 4.0
                elif 3.5 < pb <= 6.0:
                    fundamental_score += 2.0

            total_score = round(trend_score + volume_score + inst_score + fundamental_score, 2)
            strict_pass = bool(
                is_bullish_aligned
                and (inst_consec >= 1 or inst_5d_net > 0)
                and (is_etf or pe > 0 or dy >= 2.0)
            )

            records.append(
                {
                    "date": latest_market_date,
                    "symbol": str(symbol),
                    "code": str(last_row["code"]),
                    "name": str(last_row["name"]),
                    "market": str(last_row["market"]),
                    "is_etf": is_etf,
                    "close": close,
                    "ma5": round(ma5, 2),
                    "ma10": round(ma10, 2),
                    "ma20": round(ma20, 2),
                    "bias_20": round(bias_20, 2),
                    "vol_ratio_20": round(vol_ratio_20, 2),
                    "inst_consec_days": inst_consec,
                    "trust_consec_days": trust_consec,
                    "inst_5d_lots": int(round(inst_5d_net / 1000.0)),
                    "pe_ratio": pe,
                    "dividend_yield": dy,
                    "pb_ratio": pb,
                    "strict_pass": strict_pass,
                    "trend_score": round(trend_score, 2),
                    "volume_score": round(volume_score, 2),
                    "inst_score": round(inst_score, 2),
                    "fundamental_score": round(fundamental_score, 2),
                    "total_score": total_score,
                }
            )

        if not records:
            return pd.DataFrame()

        scored_df = pd.DataFrame(records)
        strict_df = scored_df[scored_df["strict_pass"]].sort_values(
            ["total_score", "inst_consec_days", "vol_ratio_20"],
            ascending=[False, False, False],
        )

        if len(strict_df) >= self.top_n:
            return strict_df.head(self.top_n).reset_index(drop=True)

        # Fallback: fill remaining slots with non-strict liquid uptrend candidates ranked by total_score
        rest_df = scored_df[~scored_df["strict_pass"]].sort_values(
            ["total_score", "inst_consec_days", "vol_ratio_20"],
            ascending=[False, False, False],
        )
        combined = pd.concat([strict_df, rest_df], ignore_index=True)
        return combined.head(self.top_n).reset_index(drop=True)

    @staticmethod
    def _register_tw_stock_names(history_df: pd.DataFrame) -> None:
        """Populate global STOCK_NAME_MAP with Taiwan stock names from history_df."""
        if history_df is None or history_df.empty:
            return
        try:
            from src.data.stock_mapping import STOCK_NAME_MAP

            latest_names = (
                history_df[["symbol", "name"]]
                .dropna()
                .drop_duplicates(subset=["symbol"], keep="last")
            )
            for row in latest_names.itertuples(index=False):
                sym = str(row.symbol).strip().upper()
                name = str(row.name).strip()
                if sym and name:
                    STOCK_NAME_MAP[sym] = name
        except Exception:
            pass

    def run_screening(self, *, update_history: bool = True) -> Tuple[List[str], pd.DataFrame]:
        """
        Execute the full daily update + quantitative screening pipeline.

        Returns:
            ``(top_symbols, ranked_df)`` where ``top_symbols`` is a list of up to ``top_n``
            Yahoo Finance Taiwan symbols (e.g. ``['2330.TW', '6488.TWO', ...]``).
        """
        history_df = self.update_daily_history() if update_history else self._load_history()
        ranked_df = self.score_universe(history_df)
        if ranked_df.empty:
            logger.warning("[TWMarketScreener] 初篩結果為空，未選出任何台股標的")
            return [], ranked_df

        top_symbols = ranked_df["symbol"].astype(str).tolist()
        self._register_tw_stock_names(history_df)
        logger.info(
            "[TWMarketScreener] 台股全市場量化初篩完成，選出 Top %d 標的: %s",
            len(top_symbols),
            ", ".join(
                f"{row.symbol}({row.name}:{row.total_score:.1f}分)"
                for row in ranked_df.itertuples(index=False)
            ),
        )
        return top_symbols, ranked_df


def get_top_tw_screened_stocks(
    *,
    top_n: int = 20,
    bootstrap_days: int = 22,
    parquet_path: Optional[Path] = None,
    update_history: bool = True,
) -> List[str]:
    """Convenience entry point for ``main.py`` to fetch the daily Top N screened Taiwan stocks."""
    screener = TWMarketScreener(
        parquet_path=parquet_path,
        top_n=top_n,
        bootstrap_days=bootstrap_days,
    )
    symbols, _ = screener.run_screening(update_history=update_history)
    return symbols
