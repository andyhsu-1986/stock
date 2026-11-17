# -*- coding: utf-8 -*-
"""
===================================
台美股雙軌量化算價與財報趨勢引擎 (engine/quant_evaluator.py)
===================================

職責：
1. 台股個股管線：
   - 串接 FinMind 抓取近 3～5 年季報（損益表、現金流量表）、月營收與歷史本益比，快取於 `data/tw_fundamentals.parquet`（每季檢查更新即可）。
   - 計算近 3 年營收 CAGR、近 4 季毛利率變化斜率（OLS）、營業現金流比率（OCF Ratio）。
   - 結合「5 年本益比區間」與「20日 / 60日均線」算出第一買點（順勢拉回）與第二買點（價值支撐）。
2. 台美股 ETF 管線：
   - 台股 ETF 檢核證交所即時折溢價率（溢價超過 1.5% 時標記風險並強制降級建議）。
   - 台股與美股 ETF 統一以「240日年線乖離率」與「歷史殖利率分位數」計算建議買入區間（第一買點、第二買點）。
3. 共用風控算價：
   - 統一使用 14 日 ATR 計算動態移動停損價（近 20 日高點減去 2.5 倍 ATR）與波段停利參考價。
4. 四維度信心分數合成：
   - 依據「35% 財報基本面 + 25% 技術位階 + 20% 籌碼總經 + 20% AI 新聞一致性」產出 0～100 最終信心分數與「買入 / 持有 / 賣出」繁體中文建議。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


def _taipei_today() -> date:
    """Return current calendar date in Asia/Taipei regardless of host OS timezone."""
    return datetime.now(ZoneInfo("Asia/Taipei")).date()

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

_FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
_TWSE_ALL_ETF_NAV_URL = "https://mis.twse.com.tw/stock/data/all_etf.txt"
_DEFAULT_FUNDAMENTALS_PARQUET = (
    Path(__file__).resolve().parents[1] / "data" / "tw_fundamentals.parquet"
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Common US ETFs and heuristic detection
_KNOWN_US_ETFS = {
    "VOO", "SPY", "IVV", "QQQ", "QQQM", "SMH", "SOXX", "VTI", "VT", "VXUS",
    "SCHD", "VIG", "VYM", "JEPI", "JEPQ", "TLT", "IEF", "SHY", "BND", "AGG",
    "GLD", "IAU", "SLV", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP",
    "XLU", "XLB", "XLRE", "XLC", "ARKK", "IWM", "DIA", "EEM", "VWO", "IBIT",
}
_TW_ETF_CODE_RE = re.compile(r"^00\d{2,4}[A-Z]?$")


@dataclass(frozen=True)
class QuantWeightConfig:
    """四維度最終信心分數權重配置（總和 = 1.0）"""

    fundamental_weight: float = 0.35  # 35% 財報基本面 / ETF 估值位階
    technical_weight: float = 0.25    # 25% 技術位階
    chip_macro_weight: float = 0.20   # 20% 籌碼與總經
    ai_news_weight: float = 0.20      # 20% AI 新聞一致性

    # 風控與估值參數常數
    atr_period: int = 14
    stop_loss_high_lookback: int = 20
    atr_stop_multiplier: float = 2.5
    atr_take_profit_multiplier: float = 3.0
    max_single_trade_loss_pct: float = 0.08  # 單筆最大硬性停損上限 8%（買入參考價往下 8%）
    etf_premium_warning_threshold_pct: float = 1.5
    fundamentals_cache_ttl_days: int = 90  # 每季（90天）檢查更新一次
    default_staged_entry_plan: str = (
        "首波試單 30% 資金（於第一買點順勢拉回建立基本部位）➔ "
        "第二波加碼 40% 資金（待 5 日均線走平翻揚且近 3 日收盤未破底確認止跌後投入）➔ "
        "保留 30% 機動現金（突破確認加碼或極端行情備用），單筆最大虧損嚴格限制在 8% 以內"
    )


DEFAULT_QUANT_WEIGHTS = QuantWeightConfig()


@dataclass
class QuantEvaluationResult:
    """台美股雙軌量化算價與財報趨勢評估結果"""

    symbol: str
    name: str
    market: str              # "tw" or "us"
    asset_track: str         # "tw_stock" | "tw_etf" | "us_etf" | "us_stock"
    is_etf: bool
    currency: str            # "TWD" or "USD"
    as_of_date: str

    # 價格與均線位階
    current_price: float = 0.0
    ma5: float = 0.0
    prev_ma5: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    ma240: float = 0.0
    high_20d: float = 0.0
    atr_14: float = 0.0

    # 雙軌算價與風控點位
    buy_point_1: float = 0.0          # 第一買點（順勢拉回）
    buy_point_2: float = 0.0          # 第二買點（價值支撐）
    atr_stop_loss: float = 0.0        # ATR 計算停損價（近 20 日高點 - 2.5 * ATR14）
    hard_cap_stop_loss: float = 0.0   # 硬性停損上限價（買入參考價往下 8%）
    dynamic_stop_loss: float = 0.0    # 最終移動停損價 = max(近20日高點-2.5*ATR14, 買入參考價往下8%)
    swing_take_profit: float = 0.0    # 波段停利參考價
    buy_zone_desc: str = ""           # 買入區間文字說明

    # 第二買點止跌確認機制（5日均線走平翻揚 + 近3日收盤未再破底）
    ma5_stabilized: bool = False
    no_new_low_3d: bool = False
    bottom_stabilized: bool = False
    buy_point_2_status: str = ""

    # 新手友善分批資金配置與白話指標說明
    position_sizing_advice: str = ""
    plain_indicator_explanations: List[str] = field(default_factory=list)

    # 台股個股專屬財報斜率與本益比指標
    revenue_cagr_3y_pct: Optional[float] = None
    gross_margin_slope_4q: Optional[float] = None
    latest_gross_margin_pct: Optional[float] = None
    operating_cash_flow_ratio_pct: Optional[float] = None
    eps_ttm: Optional[float] = None
    current_pe: Optional[float] = None
    pe_5y_low: Optional[float] = None
    pe_5y_mean: Optional[float] = None
    pe_5y_high: Optional[float] = None

    # 台美股 ETF 專屬估值與折溢價指標
    bias_ma240_pct: Optional[float] = None
    current_dividend_yield_pct: Optional[float] = None
    dividend_yield_percentile: Optional[float] = None
    etf_nav: Optional[float] = None
    etf_premium_discount_pct: Optional[float] = None
    etf_premium_risk_flag: bool = False

    # 四維度客觀分項得分 (0 ~ 100)
    fundamental_score: float = 50.0   # 權重 35%
    technical_score: float = 50.0     # 權重 25%
    chip_macro_score: float = 50.0    # 權重 20%
    ai_news_score_default: float = 50.0  # 權重 20%（供 LLM 依近 3 日新聞更新）
    base_confidence_score: int = 50   # 預設合成信心分數 (0 ~ 100)
    quant_action_advice: str = "持有"  # "買入" | "持有" | "賣出"
    risk_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure Modular Mathematical Functions
# ---------------------------------------------------------------------------
def check_bottom_stabilization(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify stop-falling confirmation (止跌確認機制) before catching a falling knife at Second Buy Point:
      1. 5-day moving average has flattened or turned up (`ma5_curr >= ma5_prev`).
      2. Closing prices over the last 3 trading days have not made a new low (`no_new_low_3d`):
         neither of the last 2 days broke below day -3's close (`closes[-1] >= closes[-3]` and
         `closes[-2] >= closes[-3]`), and the recent 2-day close has not broken below the prior
         swing low (`min(closes[-2:]) >= min(closes[-min(20, n):-2])`).
    """
    if df is None or df.empty or "close" not in df.columns:
        return {
            "ma5": 0.0,
            "prev_ma5": 0.0,
            "ma5_stabilized": False,
            "no_new_low_3d": False,
            "bottom_stabilized": False,
        }

    closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    n = len(closes)
    if n == 0:
        return {
            "ma5": 0.0,
            "prev_ma5": 0.0,
            "ma5_stabilized": False,
            "no_new_low_3d": False,
            "bottom_stabilized": False,
        }

    ma5_curr = round(float(np.mean(closes[-min(5, n):])), 2)
    ma5_prev = round(float(np.mean(closes[-min(6, n):-1])), 2) if n >= 2 else ma5_curr
    ma5_stabilized = bool(ma5_curr >= ma5_prev)

    if n >= 3:
        c_m3, c_m2, c_m1 = float(closes[-3]), float(closes[-2]), float(closes[-1])
        not_breaking_3d_base = bool(c_m2 >= c_m3 and c_m1 >= c_m3)
        if n >= 4:
            prior_window = closes[-min(20, n):-2]
            not_breaking_prior_low = bool(min(c_m2, c_m1) >= float(np.min(prior_window)))
        else:
            not_breaking_prior_low = True
        no_new_low_3d = bool(not_breaking_3d_base and not_breaking_prior_low)
    else:
        no_new_low_3d = bool(closes[-1] >= closes[0])

    return {
        "ma5": ma5_curr,
        "prev_ma5": ma5_prev,
        "ma5_stabilized": ma5_stabilized,
        "no_new_low_3d": no_new_low_3d,
        "bottom_stabilized": bool(ma5_stabilized and no_new_low_3d),
    }


def compute_atr_14(
    df: pd.DataFrame,
    period: int = 14,
    high_lookback: int = 20,
    stop_multiplier: float = 2.5,
    take_profit_multiplier: float = 3.0,
    buy_reference_price: Optional[float] = None,
    max_stop_loss_pct: float = 0.08,
) -> Tuple[float, float, float, float]:
    """
    Calculate 14-day ATR, 20-day High, Dynamic Trailing Stop Loss (with 8% hard cap), and Swing Take Profit.

    Formula:
      - TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
      - ATR_14 = mean(TR, 14)
      - ATR Stop Loss = High_20D - 2.5 * ATR_14
      - Hard Cap Stop Loss = Buy_Reference_Price * (1 - 0.08)
      - Dynamic Stop Loss = max(ATR Stop Loss, Hard Cap Stop Loss)
      - Swing Take Profit = max(High_20D + 1.5 * ATR_14, Close + 3.0 * ATR_14)

    Returns:
        (atr_14, high_20d, dynamic_stop_loss, swing_take_profit)
    """
    if df is None or df.empty or "close" not in df.columns:
        return 0.0, 0.0, 0.0, 0.0

    closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(closes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    highs = (
        pd.to_numeric(df["high"], errors="coerce").fillna(df["close"]).to_numpy(dtype=np.float64)
        if "high" in df.columns
        else closes.copy()
    )
    lows = (
        pd.to_numeric(df["low"], errors="coerce").fillna(df["close"]).to_numpy(dtype=np.float64)
        if "low" in df.columns
        else closes.copy()
    )

    n = min(len(closes), len(highs), len(lows))
    closes, highs, lows = closes[-n:], highs[-n:], lows[-n:]
    current_close = float(closes[-1])

    if n >= 2:
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
        )
    else:
        tr = np.maximum(highs - lows, current_close * 0.015)

    atr_window = tr[-min(period, len(tr)):]
    atr_val = float(np.mean(atr_window)) if len(atr_window) > 0 else current_close * 0.015
    if atr_val <= 0:
        atr_val = current_close * 0.015

    atr_val = round(atr_val, 2)
    high_20d = round(float(np.max(highs[-min(high_lookback, len(highs)):])), 2)
    atr_stop_loss = max(0.01, round(high_20d - stop_multiplier * atr_val, 2))
    ref_price = (
        float(buy_reference_price)
        if (buy_reference_price is not None and float(buy_reference_price) > 0)
        else current_close
    )
    hard_cap_stop_loss = round(ref_price * (1.0 - float(max_stop_loss_pct)), 2)
    dynamic_stop_loss = max(atr_stop_loss, hard_cap_stop_loss)
    swing_take_profit = round(
        max(high_20d + 1.5 * atr_val, current_close + take_profit_multiplier * atr_val),
        2,
    )
    return atr_val, high_20d, dynamic_stop_loss, swing_take_profit


def compute_revenue_cagr_3y(monthly_revenues: Sequence[Tuple[str, float]]) -> Optional[float]:
    """
    Compute 3-year Revenue CAGR (%) from a sequence of (date_str, revenue) pairs.

    Uses trailing 12-month (TTM) revenue vs. the 12-month period 3 years (36 months) prior,
    or falls back to earliest available 12-month window if between 24 and 47 months exist.
    """
    if not monthly_revenues:
        return None
    sorted_items = sorted(
        [(str(d), float(v)) for d, v in monthly_revenues if v is not None and float(v) > 0],
        key=lambda x: x[0],
    )
    if len(sorted_items) < 24:
        return None

    recent_12m = sum(v for _, v in sorted_items[-12:])
    if len(sorted_items) >= 48:
        base_12m = sum(v for _, v in sorted_items[-48:-36])
        years = 3.0
    elif len(sorted_items) >= 36:
        base_12m = sum(v for _, v in sorted_items[-36:-24])
        years = 2.0
    else:
        base_12m = sum(v for _, v in sorted_items[:12])
        years = max(1.0, (len(sorted_items) - 12) / 12.0)

    if base_12m <= 0 or recent_12m <= 0:
        return None
    cagr = ((recent_12m / base_12m) ** (1.0 / years) - 1.0) * 100.0
    return round(float(cagr), 2)


def compute_gross_margin_slope_4q(quarterly_margins_pct: Sequence[float]) -> Optional[float]:
    """
    Compute OLS linear regression slope of gross margin (%) over the most recent 4 quarters.

    Positive slope indicates expanding gross margin (percentage points per quarter).
    """
    clean = [float(x) for x in quarterly_margins_pct if x is not None and not np.isnan(float(x))]
    if len(clean) < 2:
        return None
    window = np.array(clean[-4:], dtype=np.float64)
    x = np.arange(len(window), dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(window))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return 0.0
    slope = float(np.sum((x - x_mean) * (window - y_mean)) / denom)
    return round(slope, 2)


def compute_weighted_confidence_score(
    fundamental_score: float,
    technical_score: float,
    chip_macro_score: float,
    ai_news_score: float = 50.0,
    *,
    weights: QuantWeightConfig = DEFAULT_QUANT_WEIGHTS,
    etf_premium_risk_flag: bool = False,
) -> Tuple[int, str]:
    """
    Compute 0-100 final confidence score and Traditional Chinese recommendation ("買入" / "持有" / "賣出").

    Formula:
      Final Score = 35% * Fundamental + 25% * Technical + 20% * Chip/Macro + 20% * AI News Consistency
    """
    f_score = float(np.clip(fundamental_score, 0.0, 100.0))
    t_score = float(np.clip(technical_score, 0.0, 100.0))
    c_score = float(np.clip(chip_macro_score, 0.0, 100.0))
    n_score = float(np.clip(ai_news_score, 0.0, 100.0))

    raw_total = (
        weights.fundamental_weight * f_score
        + weights.technical_weight * t_score
        + weights.chip_macro_weight * c_score
        + weights.ai_news_weight * n_score
    )
    final_score = int(round(float(np.clip(raw_total, 0.0, 100.0))))

    if final_score >= 60:
        advice = "買入"
    elif final_score >= 40:
        advice = "持有"
    else:
        advice = "賣出"

    # Guardrail: Taiwan ETF with real-time premium > 1.5% must be downgraded from Buy to Hold
    if etf_premium_risk_flag and advice == "買入":
        advice = "持有"
        final_score = min(final_score, 58)

    return final_score, advice


# ---------------------------------------------------------------------------
# Main QuantEvaluator Engine
# ---------------------------------------------------------------------------
class QuantEvaluator:
    """
    Dual-Track Quantitative Pricing & Financial Trend Evaluator for Taiwan & US Markets.
    """

    _etf_nav_cache: Dict[str, Tuple[float, float, float]] = {}
    _etf_nav_cache_ts: float = 0.0
    _etf_nav_lock = threading.Lock()

    def __init__(
        self,
        *,
        fundamentals_parquet_path: Optional[Path] = None,
        weights: QuantWeightConfig = DEFAULT_QUANT_WEIGHTS,
        finmind_token: Optional[str] = None,
        request_timeout: int = 15,
    ) -> None:
        self.fundamentals_parquet_path = (
            Path(fundamentals_parquet_path)
            if fundamentals_parquet_path
            else _DEFAULT_FUNDAMENTALS_PARQUET
        )
        self.weights = weights
        self.finmind_token = (
            finmind_token
            if finmind_token is not None
            else (os.getenv("FINMIND_API_TOKEN") or "").strip()
        )
        self.request_timeout = max(5, int(request_timeout))
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    # ------------------------------------------------------------------ HTTP with Exponential Backoff
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
        reraise=True,
    )
    def _request_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self._session.get(url, params=params, timeout=self.request_timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RuntimeError(f"HTTP {resp.status_code} from {url}")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ Asset Classification
    @staticmethod
    def classify_asset(symbol: str, quote_info: Optional[Dict[str, Any]] = None) -> Tuple[str, str, bool]:
        """Return ``(market, asset_track, is_etf)`` for a Taiwan or US symbol."""
        clean = str(symbol or "").strip().upper()
        if clean.endswith(".TW") or clean.endswith(".TWO"):
            bare = clean.rsplit(".", 1)[0]
            is_etf = bool(_TW_ETF_CODE_RE.match(bare))
            return "tw", ("tw_etf" if is_etf else "tw_stock"), is_etf

        # US symbol
        bare = clean.split(".", 1)[0]
        is_etf = bare in _KNOWN_US_ETFS
        if not is_etf and isinstance(quote_info, dict):
            qtype = str(quote_info.get("quoteType") or "").upper()
            if qtype == "ETF":
                is_etf = True
        return "us", ("us_etf" if is_etf else "us_stock"), is_etf

    # ------------------------------------------------------------------ FinMind Quarterly Cache (data/tw_fundamentals.parquet)
    def _load_fundamentals_cache(self) -> pd.DataFrame:
        if not self.fundamentals_parquet_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.fundamentals_parquet_path)
        except Exception as exc:
            logger.warning("[QuantEvaluator] 讀取 tw_fundamentals.parquet 失敗: %s", exc)
            return pd.DataFrame()

    def _save_fundamentals_cache(self, df: pd.DataFrame) -> None:
        try:
            self.fundamentals_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.fundamentals_parquet_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, self.fundamentals_parquet_path)
        except Exception as exc:
            logger.warning("[QuantEvaluator] 寫入 tw_fundamentals.parquet 失敗: %s", exc)

    def get_or_fetch_tw_stock_fundamentals(self, bare_code: str) -> Dict[str, Any]:
        """
        Retrieve Taiwan stock 3-5Y quarterly & monthly fundamentals from `data/tw_fundamentals.parquet`.
        Only queries FinMind API if cache is missing or older than 90 days (quarterly check).
        """
        cache_df = self._load_fundamentals_cache()
        today_tw = _taipei_today()
        today_str = today_tw.strftime("%Y-%m-%d")
        cutoff_date = (
            today_tw - timedelta(days=self.weights.fundamentals_cache_ttl_days)
        ).strftime("%Y-%m-%d")

        if not cache_df.empty and "code" in cache_df.columns:
            matched = cache_df[cache_df["code"].astype(str) == str(bare_code)]
            if not matched.empty:
                row = matched.iloc[-1]
                updated_at = str(row.get("updated_at") or "")
                if updated_at >= cutoff_date:
                    return row.to_dict()

        # Fetch from FinMind API (3-5 years)
        start_5y = (today_tw - timedelta(days=365 * 5 + 30)).strftime("%Y-%m-%d")
        start_4y = (today_tw - timedelta(days=365 * 4 + 30)).strftime("%Y-%m-%d")

        fin_metrics = self._fetch_finmind_stock_metrics(bare_code, start_4y=start_4y, start_5y=start_5y)
        fin_metrics["code"] = str(bare_code)
        fin_metrics["updated_at"] = today_str

        # Upsert into data/tw_fundamentals.parquet
        new_row_df = pd.DataFrame([fin_metrics])
        if not cache_df.empty and "code" in cache_df.columns:
            rem = cache_df[cache_df["code"].astype(str) != str(bare_code)]
            updated_df = pd.concat([rem, new_row_df], ignore_index=True)
        else:
            updated_df = new_row_df
        self._save_fundamentals_cache(updated_df)
        return fin_metrics

    def _fetch_finmind_dataset(self, dataset: str, data_id: str, start_date: str) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
        }
        if self.finmind_token:
            params["token"] = self.finmind_token
        try:
            payload = self._request_json(_FINMIND_API_URL, params=params)
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return payload["data"]
        except Exception as exc:
            logger.warning("[QuantEvaluator] FinMind %s (%s) 抓取失敗: %s", dataset, data_id, exc)
        return []

    def _fetch_finmind_stock_metrics(
        self, bare_code: str, *, start_4y: str, start_5y: str
    ) -> Dict[str, Any]:
        # 1. Monthly Revenue -> 3Y CAGR
        rev_rows = self._fetch_finmind_dataset("TaiwanStockMonthRevenue", bare_code, start_4y)
        monthly_pairs: List[Tuple[str, float]] = []
        for r in rev_rows:
            d = str(r.get("date") or "")
            val = float(r.get("revenue") or 0.0)
            if d and val > 0:
                monthly_pairs.append((d, val))
        revenue_cagr_3y = compute_revenue_cagr_3y(monthly_pairs)

        # 2. Quarterly Income Statement -> 4Q Gross Margin Slope, Operating Income, EPS TTM
        stmt_rows = self._fetch_finmind_dataset("TaiwanStockFinancialStatements", bare_code, start_4y)
        by_quarter: Dict[str, Dict[str, float]] = {}
        for r in stmt_rows:
            q_date = str(r.get("date") or "")
            metric_type = str(r.get("type") or "")
            val = r.get("value")
            if not q_date or val is None:
                continue
            by_quarter.setdefault(q_date, {})[metric_type] = float(val)

        sorted_quarters = sorted(by_quarter.keys())
        gm_series: List[float] = []
        op_inc_series: List[float] = []
        rev_q_series: List[float] = []
        eps_series: List[float] = []

        for q in sorted_quarters:
            q_map = by_quarter[q]
            rev_q = q_map.get("Revenue", 0.0)
            gp_q = q_map.get("GrossProfit")
            op_q = q_map.get("OperatingIncome", 0.0)
            eps_q = q_map.get("EPS")
            if rev_q and rev_q > 0 and gp_q is not None:
                gm_series.append(gp_q / rev_q * 100.0)
            if rev_q and rev_q > 0:
                rev_q_series.append(rev_q)
            if op_q:
                op_inc_series.append(op_q)
            if eps_q is not None:
                eps_series.append(eps_q)

        gm_slope_4q = compute_gross_margin_slope_4q(gm_series)
        latest_gm = round(gm_series[-1], 2) if gm_series else None
        eps_ttm = round(float(sum(eps_series[-4:])), 2) if len(eps_series) >= 4 else (
            round(float(sum(eps_series)), 2) if eps_series else None
        )

        # 3. Quarterly Cash Flow Statement -> Operating Cash Flow Ratio
        cf_rows = self._fetch_finmind_dataset("TaiwanStockCashFlowsStatement", bare_code, start_4y)
        ocf_by_q: Dict[str, float] = {}
        for r in cf_rows:
            q_date = str(r.get("date") or "")
            m_type = str(r.get("type") or "")
            val = r.get("value")
            if q_date and val is not None and m_type in (
                "CashFlowsFromOperatingActivities",
                "NetCashInflowFromOperatingActivities",
            ):
                ocf_by_q[q_date] = float(val)

        ocf_4q_vals = [ocf_by_q[q] for q in sorted(ocf_by_q.keys())[-4:]]
        ocf_ratio_pct: Optional[float] = None
        if ocf_4q_vals:
            sum_ocf = float(sum(ocf_4q_vals))
            denom_op = float(sum(op_inc_series[-4:])) if op_inc_series else 0.0
            denom_rev = float(sum(rev_q_series[-4:])) if rev_q_series else 0.0
            if denom_op > 0:
                ocf_ratio_pct = round(sum_ocf / denom_op * 100.0, 2)
            elif denom_rev > 0:
                ocf_ratio_pct = round(sum_ocf / denom_rev * 100.0, 2)

        # 4. 5-Year Historical PER & Dividend Yield
        per_rows = self._fetch_finmind_dataset("TaiwanStockPER", bare_code, start_5y)
        pe_vals = [
            float(r["PER"])
            for r in per_rows
            if r.get("PER") is not None and 0.0 < float(r["PER"]) < 150.0
        ]
        dy_vals = [
            float(r["dividend_yield"])
            for r in per_rows
            if r.get("dividend_yield") is not None and float(r["dividend_yield"]) >= 0.0
        ]

        pe_5y_low = round(float(np.percentile(pe_vals, 25)), 2) if pe_vals else None
        pe_5y_mean = round(float(np.median(pe_vals)), 2) if pe_vals else None
        pe_5y_high = round(float(np.percentile(pe_vals, 75)), 2) if pe_vals else None
        current_pe = round(float(pe_vals[-1]), 2) if pe_vals else None
        current_dy = round(float(dy_vals[-1]), 2) if dy_vals else None

        return {
            "revenue_cagr_3y_pct": revenue_cagr_3y,
            "gross_margin_slope_4q": gm_slope_4q,
            "latest_gross_margin_pct": latest_gm,
            "operating_cash_flow_ratio_pct": ocf_ratio_pct,
            "eps_ttm": eps_ttm,
            "current_pe": current_pe,
            "pe_5y_low": pe_5y_low,
            "pe_5y_mean": pe_5y_mean,
            "pe_5y_high": pe_5y_high,
            "current_dividend_yield_pct": current_dy,
        }

    # ------------------------------------------------------------------ TWSE Real-Time ETF NAV & Premium/Discount
    def get_tw_etf_realtime_premium(self, bare_code: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Return ``(estimated_nav, premium_discount_pct)`` for a Taiwan ETF from TWSE MIS `all_etf.txt`.
        Cached in memory for 5 minutes to avoid redundant requests.
        """
        clean_code = str(bare_code or "").strip().upper()
        with self._etf_nav_lock:
            now = time.time()
            if self._etf_nav_cache and (now - self._etf_nav_cache_ts) < 300:
                hit = self._etf_nav_cache.get(clean_code)
                return (hit[1], hit[2]) if hit else (None, None)

            try:
                payload = self._request_json(_TWSE_ALL_ETF_NAV_URL)
                nav_map: Dict[str, Tuple[float, float, float]] = {}
                for group in (payload or {}).get("a1") or []:
                    for item in (group or {}).get("msgArray") or []:
                        code = str(item.get("a") or "").strip().upper()
                        if not code:
                            continue
                        price = float(item.get("e") or 0.0)
                        nav = float(item.get("f") or 0.0)
                        prem = float(item.get("g") or 0.0)
                        nav_map[code] = (price, nav, prem)
                if nav_map:
                    self._etf_nav_cache = nav_map
                    self._etf_nav_cache_ts = now
            except Exception as exc:
                logger.warning("[QuantEvaluator] 抓取台股 ETF 即時折溢價失敗 (Fail-Open): %s", exc)

            hit = self._etf_nav_cache.get(clean_code)
            return (hit[1], hit[2]) if hit else (None, None)

    # ------------------------------------------------------------------ Daily Bars & Dividend History Loader
    @staticmethod
    def _ensure_long_history_bars(symbol: str, daily_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Ensure at least 240 trading bars when possible by querying yfinance if input df is short."""
        if daily_df is not None and len(daily_df) >= 240:
            return daily_df
        try:
            import yfinance as yf

            hist = yf.Ticker(symbol).history(period="2y", auto_adjust=False)
            if hist is not None and not hist.empty:
                df = hist.reset_index()
                col_map = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
                df = df.rename(columns=col_map)
                return df
        except Exception as exc:
            logger.debug("[QuantEvaluator] yfinance 2Y history fallback skipped for %s: %s", symbol, exc)
        return daily_df if daily_df is not None else pd.DataFrame()

    @staticmethod
    def _compute_yfinance_yield_distribution(symbol: str, current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Compute current TTM dividend yield (%) and 5-year historical yield percentile (0-100) via yfinance."""
        if current_price <= 0:
            return None, None
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5y", auto_adjust=False)
            if hist is None or hist.empty or "Close" not in hist.columns:
                return None, None

            div_col = hist["Dividends"] if "Dividends" in hist.columns else pd.Series(0.0, index=hist.index)
            rolling_1y_div = div_col.rolling(window=252, min_periods=60).sum()
            yield_series = (rolling_1y_div / hist["Close"] * 100.0).dropna()
            yield_series = yield_series[yield_series > 0]
            if yield_series.empty:
                return None, None

            current_yield = round(float(yield_series.iloc[-1]), 2)
            percentile = round(
                float(np.mean(yield_series.to_numpy(dtype=np.float64) <= current_yield) * 100.0),
                1,
            )
            return current_yield, percentile
        except Exception as exc:
            logger.debug("[QuantEvaluator] yfinance yield distribution skipped for %s: %s", symbol, exc)
            return None, None

    # ------------------------------------------------------------------ Main Evaluation Entrypoint
    def evaluate(
        self,
        symbol: str,
        *,
        name: str = "",
        daily_df: Optional[pd.DataFrame] = None,
        institutional_net: Optional[Dict[str, Any]] = None,
    ) -> QuantEvaluationResult:
        """
        Execute dual-track quantitative valuation and financial trend evaluation for a TW/US stock or ETF.
        """
        clean_symbol = str(symbol or "").strip().upper()
        market, asset_track, is_etf = self.classify_asset(clean_symbol)
        currency = "TWD" if market == "tw" else "USD"

        df = (
            self._ensure_long_history_bars(clean_symbol, daily_df)
            if is_etf or daily_df is None or len(daily_df) < 60
            else daily_df.copy()
        )

        closes = (
            pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if df is not None and not df.empty and "close" in df.columns
            else np.array([], dtype=np.float64)
        )
        current_price = float(closes[-1]) if len(closes) > 0 else 0.0
        ma20 = round(float(np.mean(closes[-min(20, len(closes)):])), 2) if len(closes) > 0 else 0.0
        ma60 = round(float(np.mean(closes[-min(60, len(closes)):])), 2) if len(closes) > 0 else 0.0
        ma240 = round(float(np.mean(closes[-min(240, len(closes)):])), 2) if len(closes) > 0 else 0.0

        atr_14, high_20d, dynamic_stop_loss, swing_take_profit = compute_atr_14(
            df,
            period=self.weights.atr_period,
            high_lookback=self.weights.stop_loss_high_lookback,
            stop_multiplier=self.weights.atr_stop_multiplier,
            take_profit_multiplier=self.weights.atr_take_profit_multiplier,
            buy_reference_price=current_price,
            max_stop_loss_pct=self.weights.max_single_trade_loss_pct,
        )
        stab = check_bottom_stabilization(df)

        result = QuantEvaluationResult(
            symbol=clean_symbol,
            name=name or clean_symbol,
            market=market,
            asset_track=asset_track,
            is_etf=is_etf,
            currency=currency,
            as_of_date=_taipei_today().strftime("%Y-%m-%d"),
            current_price=round(current_price, 2),
            ma5=stab["ma5"],
            prev_ma5=stab["prev_ma5"],
            ma20=ma20,
            ma60=ma60,
            ma240=ma240,
            high_20d=high_20d,
            atr_14=atr_14,
            dynamic_stop_loss=dynamic_stop_loss,
            swing_take_profit=swing_take_profit,
            ma5_stabilized=stab["ma5_stabilized"],
            no_new_low_3d=stab["no_new_low_3d"],
            bottom_stabilized=stab["bottom_stabilized"],
            position_sizing_advice=self.weights.default_staged_entry_plan,
        )

        if is_etf:
            self._populate_etf_track(result)
        elif market == "tw":
            self._populate_tw_stock_track(result)
        else:
            self._populate_us_stock_track(result)

        # Enforce Hard 8% Stop-Loss Cap anchored on Buy Reference Price (buy_point_1):
        # Dynamic Stop Loss = max(High_20D - 2.5 * ATR_14, Buy_Ref * (1 - 8%))
        buy_ref = result.buy_point_1 if result.buy_point_1 > 0 else result.current_price
        result.atr_stop_loss = max(
            0.01, round(result.high_20d - self.weights.atr_stop_multiplier * result.atr_14, 2)
        )
        result.hard_cap_stop_loss = round(
            buy_ref * (1.0 - self.weights.max_single_trade_loss_pct), 2
        )
        result.dynamic_stop_loss = max(result.atr_stop_loss, result.hard_cap_stop_loss)

        # Apply Second Buy Point (價值支撐) Stop-Falling Confirmation (止跌確認機制)
        self._apply_second_buy_point_stabilization_guardrail(result)

        # Populate Beginner-Friendly Plain-Language Indicator Explanations
        self._populate_plain_indicator_explanations(result)

        # Compute technical position score (0-100, 25% weight)
        result.technical_score = self._score_technical_position(
            current_price=current_price,
            ma20=ma20,
            ma60=ma60,
            ma240=ma240,
            dynamic_stop_loss=result.dynamic_stop_loss,
        )
        if not result.bottom_stabilized and current_price < ma20:
            result.technical_score = max(0.0, round(result.technical_score - 10.0, 1))

        # Compute chip & macro score (0-100, 20% weight)
        result.chip_macro_score = self._score_chip_and_macro(
            market=market,
            institutional_net=institutional_net,
            current_price=current_price,
            ma60=ma60,
        )

        # Synthesize default confidence score & recommendation
        base_score, advice = compute_weighted_confidence_score(
            fundamental_score=result.fundamental_score,
            technical_score=result.technical_score,
            chip_macro_score=result.chip_macro_score,
            ai_news_score=result.ai_news_score_default,
            weights=self.weights,
            etf_premium_risk_flag=result.etf_premium_risk_flag,
        )
        result.base_confidence_score = base_score
        result.quant_action_advice = advice
        return result

    def _apply_second_buy_point_stabilization_guardrail(self, result: QuantEvaluationResult) -> None:
        """
        Confirm whether 5-day MA has flattened/turned up AND 3-day closes have stopped making new lows
        before activating Second Buy Point (價值支撐).
        """
        if result.bottom_stabilized:
            result.buy_point_2_status = (
                f"✅ 止跌確認通過（5日均線 {result.prev_ma5}➔{result.ma5} 已走平或翻揚，且近 3 日收盤未再破底），"
                f"可於第二買點 {result.buy_point_2} {result.currency} 依計晝分批布局"
            )
        else:
            unmet: List[str] = []
            if not result.ma5_stabilized:
                unmet.append(f"5日均線仍下彎 ({result.prev_ma5}➔{result.ma5})")
            if not result.no_new_low_3d:
                unmet.append("近3日收盤價仍在破底")
            reason_str = "、".join(unmet) if unmet else "短線尚未止跌"
            result.buy_point_2_status = (
                f"⚠️ 尚未完成止跌確認（{reason_str}），第二買點 {result.buy_point_2} {result.currency} "
                "暫勿過早接刀，須待 5 日均線走平翻揚且近 3 日收盤未破底後再行承接"
            )
            result.risk_notes.append(result.buy_point_2_status)

    @staticmethod
    def _populate_plain_indicator_explanations(result: QuantEvaluationResult) -> None:
        """Generate beginner-friendly plain-language explanations of technical & valuation indicators."""
        currency = result.currency
        explanations = [
            f"📌 **均線 (MA5/MA20/MA60)**：代表過去 5日（週線 {result.ma5}）、20日（月線 {result.ma20}）、60日（季線 {result.ma60}）市場投資人的平均買進成本。股價站上均線代表多數人處於獲利狀態、下檔買盤有支撐。",
            f"🛡️ **止跌確認機制（不接掉落的刀子）**：第二買點（價值支撐 {result.buy_point_2} {currency}）除了看估值便宜外，系統強制檢查「5日均線走平或翻揚」加上「近3日收盤未再破底」（目前狀態：{result.buy_point_2_status}），避免在急跌趨勢中過早接刀。",
            f"🛑 **ATR 波動度與 8% 硬性停損上限**：14日 ATR ({result.atr_14} {currency}) 是該標的每天平均上下震盪幅度。移動停損價取「20日高點減 2.5 倍 ATR ({result.atr_stop_loss} {currency})」與「買入參考價往下 8% ({result.hard_cap_stop_loss} {currency})」兩者較高值（即 **{result.dynamic_stop_loss} {currency}**），確保單筆最大虧損不超過 8%。",
            f"💰 **保守分批進場資金建議**：{result.position_sizing_advice}",
        ]
        if result.is_etf:
            explanations.append(
                f"📊 **ETF 年線乖離與殖利率分位**：240日年線 ({result.ma240} {currency}) 代表長線一年的平均成本，乖離率 ({result.bias_ma240_pct}%) 越低代表價格越接近長線成本；殖利率分位數 ({result.dividend_yield_percentile}%) 越高代表配息保護力越強。"
            )
        else:
            explanations.append(
                f"📊 **財報趨勢與本益比 (PE)**：本益比 ({result.current_pe}) 代表買進後約幾年能靠公司獲利回本（對照 5 年平均本益比 {result.pe_5y_mean}）；搭配近 3 年營收複合年均成長率 ({result.revenue_cagr_3y_pct}%) 與營業現金流比率 ({result.operating_cash_flow_ratio_pct}%)，確認公司是真的有現金進帳的好公司。"
            )
        result.plain_indicator_explanations = explanations

    # ------------------------------------------------------------------ Track 1: Taiwan Stock Pipeline
    def _populate_tw_stock_track(self, result: QuantEvaluationResult) -> None:
        bare_code = result.symbol.rsplit(".", 1)[0]
        metrics = self.get_or_fetch_tw_stock_fundamentals(bare_code)

        result.revenue_cagr_3y_pct = metrics.get("revenue_cagr_3y_pct")
        result.gross_margin_slope_4q = metrics.get("gross_margin_slope_4q")
        result.latest_gross_margin_pct = metrics.get("latest_gross_margin_pct")
        result.operating_cash_flow_ratio_pct = metrics.get("operating_cash_flow_ratio_pct")
        result.eps_ttm = metrics.get("eps_ttm")
        result.current_pe = metrics.get("current_pe")
        result.pe_5y_low = metrics.get("pe_5y_low")
        result.pe_5y_mean = metrics.get("pe_5y_mean")
        result.pe_5y_high = metrics.get("pe_5y_high")
        result.current_dividend_yield_pct = metrics.get("current_dividend_yield_pct")

        # Calculate 5Y PE Fair Value & Value Support prices
        eps_ttm = result.eps_ttm or 0.0
        pe_mean = result.pe_5y_mean or 0.0
        pe_low = result.pe_5y_low or 0.0

        pe_mean_price = (eps_ttm * pe_mean) if (eps_ttm > 0 and pe_mean > 0) else 0.0
        pe_low_price = (eps_ttm * pe_low) if (eps_ttm > 0 and pe_low > 0) else 0.0

        ma20 = result.ma20 or result.current_price
        ma60 = result.ma60 or (ma20 * 0.95)

        if pe_mean_price > 0:
            bp1_raw = 0.70 * ma20 + 0.30 * min(ma20, pe_mean_price)
        else:
            bp1_raw = ma20 * 0.985
        result.buy_point_1 = round(min(bp1_raw, result.current_price * 0.995) if result.current_price > 0 else bp1_raw, 2)

        if pe_low_price > 0:
            bp2_raw = 0.50 * ma60 + 0.50 * pe_low_price
        else:
            bp2_raw = ma60 * 0.96
        result.buy_point_2 = round(min(bp2_raw, result.buy_point_1 * 0.96), 2)
        result.buy_zone_desc = (
            f"第一買點（順勢拉回 MA20/合理PE）：{result.buy_point_1} {result.currency} ｜ "
            f"第二買點（價值支撐 MA60/低位PE）：{result.buy_point_2} {result.currency}"
        )

        # Fundamental score (0-100, 35% weight)
        f_score = 50.0
        cagr = result.revenue_cagr_3y_pct
        if cagr is not None:
            if cagr >= 15.0:
                f_score += 18.0
            elif cagr >= 5.0:
                f_score += 10.0
            elif cagr < -5.0:
                f_score -= 10.0

        gm_slope = result.gross_margin_slope_4q
        if gm_slope is not None:
            if gm_slope >= 0.5:
                f_score += 14.0
            elif gm_slope >= 0.0:
                f_score += 7.0
            elif gm_slope < -1.0:
                f_score -= 10.0
                result.risk_notes.append(f"近 4 季毛利率變化斜率為負 ({gm_slope:+.2f}%/季)，需留意毛利壓縮")

        ocf_ratio = result.operating_cash_flow_ratio_pct
        if ocf_ratio is not None:
            if ocf_ratio >= 80.0:
                f_score += 10.0
            elif ocf_ratio >= 50.0:
                f_score += 5.0
            elif ocf_ratio < 20.0:
                f_score -= 8.0

        if result.current_pe and result.pe_5y_mean and result.pe_5y_high:
            if result.current_pe <= result.pe_5y_mean:
                f_score += 8.0
            elif result.current_pe > result.pe_5y_high * 1.15:
                f_score -= 8.0
                result.risk_notes.append(
                    f"目前本益比 ({result.current_pe}) 高於 5 年高位區間 ({result.pe_5y_high})，追價風險較高"
                )

        result.fundamental_score = round(float(np.clip(f_score, 0.0, 100.0)), 1)

    # ------------------------------------------------------------------ Track 2: TW & US ETF Pipeline
    def _populate_etf_track(self, result: QuantEvaluationResult) -> None:
        cp = result.current_price
        ma20 = result.ma20 or cp
        ma60 = result.ma60 or cp
        ma240 = result.ma240 or ma60

        bias_240 = ((cp - ma240) / ma240 * 100.0) if ma240 > 0 else 0.0
        result.bias_ma240_pct = round(bias_240, 2)

        # Dividend yield & 5Y percentile
        curr_yield, yield_pctile = self._compute_yfinance_yield_distribution(result.symbol, cp)
        result.current_dividend_yield_pct = curr_yield
        result.dividend_yield_percentile = yield_pctile

        # Taiwan ETF real-time premium/discount check
        if result.market == "tw":
            bare_code = result.symbol.rsplit(".", 1)[0]
            nav, prem_pct = self.get_tw_etf_realtime_premium(bare_code)
            result.etf_nav = nav
            result.etf_premium_discount_pct = prem_pct
            if prem_pct is not None and prem_pct > self.weights.etf_premium_warning_threshold_pct:
                result.etf_premium_risk_flag = True
                result.risk_notes.append(
                    f"⚠️ 台股 ETF 即時溢價率達 {prem_pct:+.2f}%（超過 {self.weights.etf_premium_warning_threshold_pct}% 警戒線），存在溢價收斂風險，強制降級買入建議"
                )

        # ETF Buy Zone Calculation based on 240-day MA bias & yield percentile
        yield_bonus = 0.0
        if yield_pctile is not None:
            if yield_pctile >= 70.0:
                yield_bonus = 0.01  # High historical yield percentile -> stronger valuation support
            elif yield_pctile <= 30.0:
                yield_bonus = -0.015  # Low yield percentile -> demand deeper pullback

        bp1_anchor = min(ma20, ma60 * 1.02) if bias_240 > 12.0 else ma20
        result.buy_point_1 = round(bp1_anchor * (1.0 + yield_bonus), 2)
        if cp > 0 and result.buy_point_1 >= cp:
            result.buy_point_1 = round(cp * 0.985, 2)

        bp2_anchor = max(ma240, ma60 * 0.95) if ma240 > 0 else ma60 * 0.95
        result.buy_point_2 = round(min(bp2_anchor, result.buy_point_1 * 0.96), 2)
        result.buy_zone_desc = (
            f"建議買入區間：{result.buy_point_2} ～ {result.buy_point_1} {result.currency} "
            f"（依據 240 日年線 {ma240} 與歷史殖利率分位定錨）"
        )

        # ETF Valuation Score (0-100, 35% weight)
        f_score = 60.0
        if -5.0 <= bias_240 <= 10.0:
            f_score += 15.0
        elif 10.0 < bias_240 <= 20.0:
            f_score += 5.0
        elif bias_240 > 25.0:
            f_score -= 12.0
            result.risk_notes.append(f"240 日年線乖離率達 {bias_240:+.2f}%，短線位階偏高宜分批待拉回")

        if yield_pctile is not None:
            if yield_pctile >= 70.0:
                f_score += 15.0
            elif yield_pctile >= 40.0:
                f_score += 8.0
            elif yield_pctile < 20.0:
                f_score -= 5.0

        if result.etf_premium_risk_flag:
            f_score -= 20.0

        result.fundamental_score = round(float(np.clip(f_score, 0.0, 100.0)), 1)

    def _populate_us_stock_track(self, result: QuantEvaluationResult) -> None:
        cp = result.current_price
        ma20 = result.ma20 or cp
        ma60 = result.ma60 or (ma20 * 0.95)
        result.buy_point_1 = round(min(ma20, cp * 0.99), 2)
        result.buy_point_2 = round(min(ma60, result.buy_point_1 * 0.95), 2)
        result.buy_zone_desc = (
            f"第一買點（順勢拉回 MA20）：{result.buy_point_1} USD ｜ "
            f"第二買點（價值支撐 MA60）：{result.buy_point_2} USD"
        )
        result.fundamental_score = 65.0

    # ------------------------------------------------------------------ Sub-Scorers (Technical 25%, Chip/Macro 20%)
    @staticmethod
    def _score_technical_position(
        *,
        current_price: float,
        ma20: float,
        ma60: float,
        ma240: float,
        dynamic_stop_loss: float,
    ) -> float:
        if current_price <= 0:
            return 50.0
        score = 50.0
        if ma20 > 0 and current_price >= ma20:
            score += 15.0
        if ma60 > 0 and current_price >= ma60:
            score += 15.0
        if ma20 > 0 and ma60 > 0 and ma20 >= ma60:
            score += 10.0
        if ma240 > 0 and current_price >= ma240:
            score += 10.0
        if dynamic_stop_loss > 0 and current_price < dynamic_stop_loss:
            score -= 25.0
        return round(float(np.clip(score, 0.0, 100.0)), 1)

    @staticmethod
    def _score_chip_and_macro(
        *,
        market: str,
        institutional_net: Optional[Dict[str, Any]],
        current_price: float,
        ma60: float,
    ) -> float:
        score = 55.0
        if market == "tw" and isinstance(institutional_net, dict):
            total_net = float(institutional_net.get("total_net") or 0.0)
            trust_net = float(institutional_net.get("trust_net") or 0.0)
            foreign_net = float(institutional_net.get("foreign_net") or 0.0)
            if total_net > 0:
                score += 18.0
            elif total_net < 0:
                score -= 12.0
            if trust_net > 0 and foreign_net > 0:
                score += 15.0
            elif trust_net > 0 or foreign_net > 0:
                score += 7.0
        else:
            if current_price > 0 and ma60 > 0 and current_price >= ma60:
                score += 15.0
        return round(float(np.clip(score, 0.0, 100.0)), 1)


def format_quant_evaluation_prompt_block(eval_data: Optional[Dict[str, Any]]) -> str:
    """Format `QuantEvaluationResult` dict into a Traditional Chinese Markdown section for the LLM prompt."""
    if not isinstance(eval_data, dict) or not eval_data.get("symbol"):
        return ""

    symbol = eval_data.get("symbol", "")
    track = eval_data.get("asset_track", "")
    currency = eval_data.get("currency", "TWD")
    is_etf = bool(eval_data.get("is_etf"))

    lines = [
        "",
        "## 📐 雙軌量化算價與財報趨勢引擎（客觀定錨數據，請直接引用）",
        "",
        "### 1. 客觀買賣點位、止跌確認與 8% 硬性停損上限風控",
        "| 項目 | 量化計算價位 | 算價依據與新手穩健風控規則 |",
        "|------|------|------|",
        f"| 現價 / 幣別 | **{eval_data.get('current_price', 'N/A')} {currency}** | 5日均線：{eval_data.get('ma5', 'N/A')} ｜ 20日高點：{eval_data.get('high_20d', 'N/A')} ｜ 14日 ATR：{eval_data.get('atr_14', 'N/A')} |",
        f"| 🎯 **第一買點（順勢拉回）** | **{eval_data.get('buy_point_1', 'N/A')} {currency}** | 結合 MA20 ({eval_data.get('ma20', 'N/A')}) 與合理估值/年線乖離定錨（建議首波試單 30% 資金） |",
        f"| 🔵 **第二買點（價值支撐）** | **{eval_data.get('buy_point_2', 'N/A')} {currency}** | 結合 MA60 ({eval_data.get('ma60', 'N/A')}) 與低位估值定錨 ｜ {eval_data.get('buy_point_2_status', '')} |",
        f"| 🛑 **動態移動停損價** | **{eval_data.get('dynamic_stop_loss', 'N/A')} {currency}** | 取「20日高點-2.5×ATR ({eval_data.get('atr_stop_loss', 'N/A')})」與「買入參考價往下 8% ({eval_data.get('hard_cap_stop_loss', 'N/A')})」較高者（最大單筆風險 ≤ 8%） |",
        f"| 🎊 **波段停利參考價** | **{eval_data.get('swing_take_profit', 'N/A')} {currency}** | 依風險報酬比與波段 ATR 擴展目標計算 |",
        f"| 💰 **分批進場資金比例建議** | **30% / 40% / 30%** | {eval_data.get('position_sizing_advice', DEFAULT_QUANT_WEIGHTS.default_staged_entry_plan)} |",
        "",
    ]

    if is_etf:
        prem = eval_data.get("etf_premium_discount_pct")
        prem_str = f"{prem:+.2f}%" if isinstance(prem, (int, float)) else "N/A（美股ETF或盤後無即時折溢價）"
        risk_flag_str = "⚠️ **溢價過高風險 (>1.5%，強制降級買入建議)**" if eval_data.get("etf_premium_risk_flag") else "✅ 正常（無異常溢價）"
        lines.extend([
            "### 2. 台美股 ETF 專屬估值位階與折溢價檢核",
            "| 指標 | 數值 | 量化解讀 |",
            "|------|------|------|",
            f"| 240日年線 (MA240) / 年線乖離率 | {eval_data.get('ma240', 'N/A')} / **{eval_data.get('bias_ma240_pct', 'N/A')}%** | 評估長線位階是否過熱或具價值支撐 |",
            f"| 目前殖利率 / 歷史殖利率分位數 | {eval_data.get('current_dividend_yield_pct', 'N/A')}% / **{eval_data.get('dividend_yield_percentile', 'N/A')}% 分位** | 分位數越高代表配息保護越強 |",
            f"| 台股 ETF 即時預估淨值 / 折溢價率 | 淨值 {eval_data.get('etf_nav', 'N/A')} / **{prem_str}** | {risk_flag_str} |",
            "",
        ])
    else:
        lines.extend([
            f"### 2. 個股多年財報趨勢斜率與 5 年本益比位階 ({track})",
            "| 指標 | 數值 | 量化解讀 |",
            "|------|------|------|",
            f"| 近 3 年營收 CAGR | **{eval_data.get('revenue_cagr_3y_pct', 'N/A')}%** | 評估中長期營收複合成長動能 |",
            f"| 近 4 季毛利率變化斜率 / 最新毛利率 | **{eval_data.get('gross_margin_slope_4q', 'N/A')}%/季**（最新 {eval_data.get('latest_gross_margin_pct', 'N/A')}%） | 正斜率代表產品組合或定價能力提升 |",
            f"| 營業現金流比率 (OCF Ratio) | **{eval_data.get('operating_cash_flow_ratio_pct', 'N/A')}%** | 檢驗帳面獲利轉化為實際現金流之含金量 |",
            f"| 近 4 季累積 EPS / 目前本益比 | EPS {eval_data.get('eps_ttm', 'N/A')} / **PE {eval_data.get('current_pe', 'N/A')}** | 5年本益比區間：低位 {eval_data.get('pe_5y_low', 'N/A')} ～ 平均 {eval_data.get('pe_5y_mean', 'N/A')} ～ 高位 {eval_data.get('pe_5y_high', 'N/A')} |",
            "",
        ])

    lines.extend([
        "### 3. 四維度加權信心分數定錨（請結合近 3 日重大新聞完成最終合成）",
        "- **35% 財報基本面 / ETF 估值位階得分**："
        f"**{eval_data.get('fundamental_score', 50)} / 100**",
        "- **25% 技術位階得分**："
        f"**{eval_data.get('technical_score', 50)} / 100**",
        "- **20% 籌碼與總經得分**："
        f"**{eval_data.get('chip_macro_score', 50)} / 100**",
        "- **20% AI 新聞一致性得分**：請依據近 3 日重大新聞與上述基本面/技術面的一致性評定 (0～100)，並依公式 `0.35*基本面 + 0.25*技術面 + 0.20*籌碼總經 + 0.20*AI新聞一致性` 輸出最終 `sentiment_score` (0～100) 與繁體中文「**買入 / 持有 / 賣出**」建議。",
    ])
    risk_notes = eval_data.get("risk_notes") or []
    if risk_notes:
        lines.append("- **量化引擎風險提示**：" + "；".join(str(x) for x in risk_notes))

    plain_explanations = eval_data.get("plain_indicator_explanations") or []
    if plain_explanations:
        lines.append("")
        lines.append("### 4. 新手白話指標解讀與保守分批進場資金建議")
        for item in plain_explanations:
            lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
