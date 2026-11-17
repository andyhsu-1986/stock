# -*- coding: utf-8 -*-
"""
===================================
持倉管理與 GitHub Pages 戰情室資料匯出模組 (utils/portfolio.py)
===================================

職責：
1. 讀寫 `data/portfolio.json`：管理使用者實際買進持倉（代號、買進成本價、股數、自訂停損/停利）與固定觀察名單。
2. 計算持倉損益與個人化停損／接近停利上限預警：
   - 停損價：預設取 `max(20日高點 - 2.5*ATR14, 買進成本價 * 0.92)`（支援自訂停損）。
   - 接近停利上限：現價距波段停利價 <= 2.0%，或帳面報酬率 >= +15.0%。
   - 接近停損／跌破逃命：現價觸及或距停損 <= 1.5%，或個股連 3 日跌破 20 日均線。
3. 計算 K 線圖與 XQ 指標序列（K棒、20MA、布林上下軌、MACD 柱狀體），匯出至 `docs/dashboard_data.json`，
   供 GitHub Pages 互動網頁視覺化展示與一鍵同步持倉回 GitHub。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PORTFOLIO_PATH = _REPO_ROOT / "data" / "portfolio.json"
_DEFAULT_DASHBOARD_JSON_PATH = _REPO_ROOT / "docs" / "dashboard_data.json"


def normalize_portfolio_symbol(raw: Any) -> str:
    """Normalize stock/ETF ticker to uppercase Yahoo/canonical form."""
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if text.isdigit():
        return f"{text}.TW"
    return text


def _infer_market_tag(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return "tw" if (s.endswith(".TW") or s.endswith(".TWO") or s.isdigit()) else "us"


@dataclass
class PortfolioPosition:
    symbol: str
    name: str = ""
    buy_price: float = 0.0
    shares: float = 0.0
    buy_date: str = ""
    custom_stop_loss: Optional[float] = None
    custom_take_profit: Optional[float] = None
    notes: str = ""
    currency: str = ""

    def __post_init__(self) -> None:
        self.symbol = normalize_portfolio_symbol(self.symbol)
        if not self.currency:
            self.currency = "TWD" if _infer_market_tag(self.symbol) == "tw" else "USD"
        try:
            self.buy_price = round(float(self.buy_price or 0.0), 4)
        except (TypeError, ValueError):
            self.buy_price = 0.0
        try:
            self.shares = float(self.shares or 0.0)
        except (TypeError, ValueError):
            self.shares = 0.0
        for attr in ("custom_stop_loss", "custom_take_profit"):
            val = getattr(self, attr)
            if val in (None, "", 0, "0"):
                setattr(self, attr, None)
            else:
                try:
                    num = float(val)
                    setattr(self, attr, round(num, 2) if num > 0 else None)
                except (TypeError, ValueError):
                    setattr(self, attr, None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_position_risk_metrics(
    position: PortfolioPosition,
    *,
    current_price: float,
    dynamic_stop_loss: float,
    swing_take_profit: float,
    broken_ma20_3d: bool,
    is_etf: bool,
) -> Dict[str, Any]:
    """Compute unrealized P&L, personalized SL/TP, and near-limit exit flags for a held position."""
    buy_price = float(position.buy_price or 0.0)
    shares = float(position.shares or 0.0)
    cp = float(current_price or 0.0)

    # Effective stop-loss respects user override or enforces max 8% loss below user's buy price
    base_cost_cap = round(buy_price * 0.92, 2) if buy_price > 0 else 0.0
    if position.custom_stop_loss and position.custom_stop_loss > 0:
        effective_sl = round(float(position.custom_stop_loss), 2)
    else:
        candidates = [v for v in (float(dynamic_stop_loss or 0.0), base_cost_cap) if v > 0]
        effective_sl = round(max(candidates), 2) if candidates else 0.0

    # Effective take-profit respects user override or uses quantitative swing target
    if position.custom_take_profit and position.custom_take_profit > 0:
        effective_tp = round(float(position.custom_take_profit), 2)
    else:
        effective_tp = round(float(swing_take_profit or 0.0), 2)

    pnl_pct: Optional[float] = None
    pnl_amount: Optional[float] = None
    if buy_price > 0 and cp > 0:
        pnl_pct = round((cp - buy_price) / buy_price * 100.0, 2)
        if shares > 0:
            pnl_amount = round((cp - buy_price) * shares, 2)

    dist_to_stop_pct: Optional[float] = None
    if cp > 0 and effective_sl > 0:
        dist_to_stop_pct = round((cp - effective_sl) / effective_sl * 100.0, 2)

    dist_to_tp_pct: Optional[float] = None
    if cp > 0 and effective_tp > 0:
        dist_to_tp_pct = round((effective_tp - cp) / cp * 100.0, 2)

    hit_stop_loss = bool(cp > 0 and effective_sl > 0 and cp <= effective_sl)
    near_stop_loss = bool(
        not hit_stop_loss and dist_to_stop_pct is not None and 0.0 < dist_to_stop_pct <= 1.5
    )
    hit_take_profit = bool(cp > 0 and effective_tp > 0 and cp >= effective_tp)
    near_take_profit = bool(
        not hit_take_profit
        and (
            (dist_to_tp_pct is not None and 0.0 <= dist_to_tp_pct <= 2.0)
            or (pnl_pct is not None and pnl_pct >= 15.0)
        )
    )
    broken_ma20_exit = bool(broken_ma20_3d and not is_etf)

    return {
        "is_held": True,
        "symbol": position.symbol,
        "buy_price": round(buy_price, 2),
        "shares": shares,
        "buy_date": position.buy_date,
        "notes": position.notes,
        "cost_basis": round(buy_price * shares, 2) if buy_price > 0 and shares > 0 else 0.0,
        "market_value": round(cp * shares, 2) if cp > 0 and shares > 0 else 0.0,
        "unrealized_pnl": pnl_amount,
        "unrealized_pnl_pct": pnl_pct,
        "effective_stop_loss": effective_sl,
        "effective_take_profit": effective_tp,
        "dist_to_stop_pct": dist_to_stop_pct,
        "dist_to_tp_pct": dist_to_tp_pct,
        "hit_stop_loss": hit_stop_loss,
        "near_stop_loss": near_stop_loss,
        "hit_take_profit": hit_take_profit,
        "near_take_profit": near_take_profit,
        "broken_ma20_exit": broken_ma20_exit,
    }


class PortfolioManager:
    """Persistent manager for user positions & watchlist in `data/portfolio.json`."""

    def __init__(self, portfolio_path: Optional[Path] = None) -> None:
        self.portfolio_path = Path(portfolio_path) if portfolio_path else _DEFAULT_PORTFOLIO_PATH
        self._data: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        default_payload: Dict[str, Any] = {
            "version": 1,
            "updated_at": "",
            "github_repo": "andyhsu-1986/stock",
            "watchlist": ["VOO", "QQQ", "0050.TW", "00878.TW"],
            "positions": [],
        }
        if not self.portfolio_path.exists():
            return default_payload
        try:
            payload = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return default_payload
            payload.setdefault("version", 1)
            payload.setdefault("updated_at", "")
            payload.setdefault("github_repo", "andyhsu-1986/stock")
            payload.setdefault("watchlist", [])
            payload.setdefault("positions", [])
            return payload
        except Exception as exc:
            logger.warning("[Portfolio] 讀取 portfolio.json 失敗 (%s): %s", self.portfolio_path, exc)
            return default_payload

    def save(self) -> None:
        self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
        tmp_path = self.portfolio_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.portfolio_path)

    def get_positions(self) -> List[PortfolioPosition]:
        out: List[PortfolioPosition] = []
        for raw in self._data.get("positions") or []:
            if not isinstance(raw, dict):
                continue
            sym = normalize_portfolio_symbol(raw.get("symbol"))
            if not sym:
                continue
            out.append(
                PortfolioPosition(
                    symbol=sym,
                    name=str(raw.get("name") or ""),
                    buy_price=raw.get("buy_price") or 0.0,
                    shares=raw.get("shares") or 0.0,
                    buy_date=str(raw.get("buy_date") or ""),
                    custom_stop_loss=raw.get("custom_stop_loss"),
                    custom_take_profit=raw.get("custom_take_profit"),
                    notes=str(raw.get("notes") or ""),
                    currency=str(raw.get("currency") or ""),
                )
            )
        return out

    def get_positions_map(self) -> Dict[str, PortfolioPosition]:
        """Return map indexed by normalized symbol and bare Taiwan numeric code for safe matching."""
        mapping: Dict[str, PortfolioPosition] = {}
        for pos in self.get_positions():
            mapping[pos.symbol] = pos
            bare = pos.symbol.split(".")[0]
            if bare and bare not in mapping:
                mapping[bare] = pos
        return mapping

    def get_held_symbols(self) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for pos in self.get_positions():
            if pos.symbol and pos.symbol not in seen:
                seen.add(pos.symbol)
                result.append(pos.symbol)
        return result

    def get_watchlist_symbols(self) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for raw in self._data.get("watchlist") or []:
            sym = normalize_portfolio_symbol(raw)
            if sym and sym not in seen:
                seen.add(sym)
                result.append(sym)
        return result

    def upsert_position(self, pos: PortfolioPosition) -> None:
        positions = [p for p in self.get_positions() if p.symbol != pos.symbol]
        positions.append(pos)
        self._data["positions"] = [p.to_dict() for p in positions]
        self.save()

    def raw_data(self) -> Dict[str, Any]:
        return dict(self._data)


def compute_chart_series_from_df(
    df: Optional[pd.DataFrame],
    *,
    max_bars: int = 90,
) -> Dict[str, Any]:
    """
    Compute ECharts-ready OHLCV + MA20 + Bollinger(20, 2) + WeightedClose MACD(12, 26, 9)
    arrays from daily bars DataFrame.
    """
    empty_chart: Dict[str, Any] = {
        "dates": [],
        "ohlc": [],
        "volumes": [],
        "ma20": [],
        "bb_upper": [],
        "bb_lower": [],
        "macd_dif": [],
        "macd_dea": [],
        "macd_osc": [],
    }
    if df is None or df.empty or "close" not in df.columns:
        return empty_chart

    work = df.copy()
    if "date" in work.columns:
        work["date_str"] = work["date"].astype(str).str.slice(0, 10)
        work = work.sort_values("date_str")
    else:
        work["date_str"] = [str(i) for i in range(len(work))]

    for col in ("open", "high", "low", "close", "volume"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["close"])
    work = work[work["close"] > 0]
    if work.empty:
        return empty_chart

    closes = work["close"].to_numpy(dtype=np.float64)
    opens = (
        work["open"].fillna(work["close"]).to_numpy(dtype=np.float64)
        if "open" in work.columns
        else closes.copy()
    )
    highs = (
        work["high"].fillna(work["close"]).to_numpy(dtype=np.float64)
        if "high" in work.columns
        else closes.copy()
    )
    lows = (
        work["low"].fillna(work["close"]).to_numpy(dtype=np.float64)
        if "low" in work.columns
        else closes.copy()
    )
    vols = (
        work["volume"].fillna(0.0).to_numpy(dtype=np.float64)
        if "volume" in work.columns
        else np.zeros(len(closes), dtype=np.float64)
    )

    n = len(closes)
    close_s = pd.Series(closes)
    win = min(20, n)
    ma20_arr = close_s.rolling(window=win, min_periods=1).mean().to_numpy(dtype=np.float64)
    std_arr = close_s.rolling(window=win, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)
    bb_up_arr = ma20_arr + 2.0 * std_arr
    bb_dn_arr = ma20_arr - 2.0 * std_arr

    wc = (highs + lows + 2.0 * closes) / 4.0
    wc_s = pd.Series(wc)
    dif_arr = (
        wc_s.ewm(span=12, adjust=False).mean() - wc_s.ewm(span=26, adjust=False).mean()
    ).to_numpy(dtype=np.float64)
    dea_arr = pd.Series(dif_arr).ewm(span=9, adjust=False).mean().to_numpy(dtype=np.float64)
    osc_arr = dif_arr - dea_arr

    sl = slice(-max_bars, None) if n > max_bars else slice(None)
    dates = work["date_str"].tolist()[sl]

    def _r2(arr: np.ndarray) -> List[Optional[float]]:
        return [None if np.isnan(v) else round(float(v), 2) for v in arr[sl]]

    def _r3(arr: np.ndarray) -> List[Optional[float]]:
        return [None if np.isnan(v) else round(float(v), 3) for v in arr[sl]]

    ohlc = [
        [round(float(o), 2), round(float(c), 2), round(float(l), 2), round(float(h), 2)]
        for o, c, l, h in zip(opens[sl], closes[sl], lows[sl], highs[sl])
    ]
    volumes = [int(round(float(v))) for v in vols[sl]]

    return {
        "dates": dates,
        "ohlc": ohlc,
        "volumes": volumes,
        "ma20": _r2(ma20_arr),
        "bb_upper": _r2(bb_up_arr),
        "bb_lower": _r2(bb_dn_arr),
        "macd_dif": _r3(dif_arr),
        "macd_dea": _r3(dea_arr),
        "macd_osc": _r3(osc_arr),
    }


def _build_stock_dashboard_card(
    result: Any,
    *,
    decision: Any = None,
    chart_series: Optional[Dict[str, Any]] = None,
    screener_info: Optional[Dict[str, Any]] = None,
    held_position: Optional[PortfolioPosition] = None,
    in_watchlist: bool = False,
    analyzed_at: str = "",
) -> Dict[str, Any]:
    """Build detailed dashboard JSON record with 4-dimension selection rationale & chart series."""
    symbol = str(getattr(result, "code", "") or "").strip().upper()
    name = str(getattr(result, "name", "") or symbol).strip()
    dash = getattr(result, "dashboard", None)
    dash = dash if isinstance(dash, dict) else {}
    qe = dash.get("quant_evaluation")
    qe = qe if isinstance(qe, dict) else {}

    market = str(qe.get("market") or _infer_market_tag(symbol))
    currency = str(qe.get("currency") or ("TWD" if market == "tw" else "USD"))
    is_etf = bool(qe.get("is_etf", False))
    current_price = round(float(getattr(result, "current_price", None) or qe.get("current_price") or 0.0), 2)
    advice = str(getattr(result, "operation_advice", None) or qe.get("quant_action_advice") or "持有")
    score = int(round(float(getattr(result, "sentiment_score", None) or qe.get("base_confidence_score") or 50)))

    bp1 = round(float(qe.get("buy_point_1") or 0.0), 2)
    bp2 = round(float(qe.get("buy_point_2") or 0.0), 2)
    sl = round(float(qe.get("dynamic_stop_loss") or 0.0), 2)
    tp = round(float(qe.get("swing_take_profit") or 0.0), 2)
    three_cond = bool(qe.get("three_condition_resonance", False))
    broken_3d = bool(qe.get("broken_ma20_3d", False))

    held_metrics: Optional[Dict[str, Any]] = None
    if held_position is not None:
        held_metrics = compute_position_risk_metrics(
            held_position,
            current_price=current_price,
            dynamic_stop_loss=sl,
            swing_take_profit=tp,
            broken_ma20_3d=broken_3d,
            is_etf=is_etf,
        )

    # Selection badges & human-readable reasons ("讓使用者一眼看懂為什麼選中這檔")
    badges: List[str] = []
    reasons: List[str] = []

    if held_position is not None:
        badges.append("💼 我的持股")
        pnl_str = (
            f"{held_metrics['unrealized_pnl_pct']:+.2f}%"
            if held_metrics and held_metrics.get("unrealized_pnl_pct") is not None
            else "計算中"
        )
        reasons.append(
            f"您實際持有本標的（買進均價 {held_position.buy_price} {currency}，目前損益 {pnl_str}），系統持續追蹤您的個人停利與停損線。"
        )
    if in_watchlist:
        badges.append("⭐ 固定觀察名單")
        reasons.append("納入您的長期固定觀察名單，每日盤後自動追蹤估值位階與買點到價狀態。")

    if isinstance(screener_info, dict) and screener_info:
        rank = screener_info.get("rank")
        scr_score = screener_info.get("total_score")
        rank_txt = f"全市場初篩第 {rank} 名" if rank else "入選全市場初篩 Top 20"
        score_txt = f"（初篩綜合分 {scr_score}）" if scr_score is not None else ""
        reasons.append(f"由台股全市場約 2,150 檔個股中脫穎而出：{rank_txt}{score_txt}，通過均量 ≥ 1,000 張、股價 ≥ 10 元、非處置/全額交割安全濾網。")

    f_score = float(qe.get("fundamental_score") or 50.0)
    t_score = float(qe.get("technical_score") or 50.0)
    c_score = float(qe.get("chip_macro_score") or 50.0)
    sig_attr = dash.get("signal_attribution") if isinstance(dash.get("signal_attribution"), dict) else {}
    ai_news_score = float(sig_attr.get("news_sentiment") or qe.get("ai_news_score_default") or 50.0)
    eps_ttm = qe.get("eps_ttm")

    if three_cond:
        if not is_etf and (f_score < 55.0 or (isinstance(eps_ttm, (int, float)) and eps_ttm <= 0)):
            badges.append("⚠️ 三條件假突破警戒")
            reasons.append(
                f"技術面雖觸發「MACD翻正＋布林開口＋站上上揚20MA」三條件共振，但多年財報基本面僅 {f_score:.0f} 分或獲利能力偏弱，系統依紀律不主動推薦買進。"
            )
        else:
            badges.append("⚡ 三條件共振黃金起漲")
            reasons.append(
                f"技術面同時滿足「WeightedClose MACD 翻正 ＋ 布林通道雙向開口 (帶寬 {qe.get('bb_bandwidth_pct')}%) ＋ 站上3日上揚月線」，且通過多年財報與籌碼檢驗！"
            )

    trig = getattr(decision, "trigger_type", None) if decision is not None else None
    if trig == "buy_zone_touched":
        badges.append("🎯 跌入建議買點")
        reasons.append("現價已回測至量化引擎計算的建議買入區間，無須追高即可分批布局。")
    elif trig == "high_confidence_buy":
        badges.append("🔥 高信心買入 ≥82分")
    elif trig == "portfolio_take_profit":
        badges.append("🎊 持股接近停利上限")
    elif trig == "portfolio_stop_loss":
        badges.append("🚨 持股停損逃命警戒")
    elif broken_3d and not is_etf:
        badges.append("📉 連3日破20MA轉弱")

    if qe.get("etf_premium_risk_flag"):
        badges.append("⚠️ ETF溢價>1.5%")

    # Extract AI Intelligence details
    intel = dash.get("intelligence") if isinstance(dash.get("intelligence"), dict) else {}
    core = dash.get("core_conclusion") if isinstance(dash.get("core_conclusion"), dict) else {}
    model_used = str(getattr(result, "model_used", None) or "")
    ai_analyzed = bool(model_used and model_used != "quant-engine-v2")

    return {
        "symbol": symbol,
        "name": name,
        "market": market,
        "asset_track": str(qe.get("asset_track") or ("tw_stock" if market == "tw" else "us_stock")),
        "is_etf": is_etf,
        "currency": currency,
        "analyzed_at": analyzed_at or datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M"),
        "ai_analyzed": ai_analyzed,
        "model_used": model_used or "quant-engine-v2",
        "advice": advice,
        "score": score,
        "trend_prediction": str(getattr(result, "trend_prediction", None) or "震盪"),
        "current_price": current_price,
        "change_pct": getattr(result, "change_pct", None),
        "buy_point_1": bp1,
        "buy_point_2": bp2,
        "dynamic_stop_loss": sl,
        "atr_stop_loss": round(float(qe.get("atr_stop_loss") or sl), 2),
        "hard_cap_stop_loss": round(float(qe.get("hard_cap_stop_loss") or 0.0), 2),
        "swing_take_profit": tp,
        "buy_zone_desc": str(qe.get("buy_zone_desc") or ""),
        "position_sizing_advice": str(qe.get("position_sizing_advice") or ""),
        "selection_badges": badges,
        "selection_reasons": reasons,
        "screener": screener_info or {},
        "held_position": held_metrics,
        "in_watchlist": bool(in_watchlist),
        "alert_decision": {
            "should_push": bool(getattr(decision, "should_push", False)) if decision else False,
            "trigger_type": trig,
            "state_label": str(getattr(decision, "state_label", advice)) if decision else advice,
            "skip_reason_desc": str(getattr(decision, "skip_reason_desc", "")) if decision else "",
            "reasons": list(getattr(decision, "reasons", []) or []) if decision else [],
        },
        "dimensions": {
            "fundamental": {
                "weight": 35,
                "score": round(f_score, 1),
                "revenue_cagr_3y_pct": qe.get("revenue_cagr_3y_pct"),
                "gross_margin_slope_4q": qe.get("gross_margin_slope_4q"),
                "latest_gross_margin_pct": qe.get("latest_gross_margin_pct"),
                "operating_cash_flow_ratio_pct": qe.get("operating_cash_flow_ratio_pct"),
                "eps_ttm": eps_ttm,
                "current_pe": qe.get("current_pe"),
                "pe_5y_low": qe.get("pe_5y_low"),
                "pe_5y_mean": qe.get("pe_5y_mean"),
                "pe_5y_high": qe.get("pe_5y_high"),
                "bias_ma240_pct": qe.get("bias_ma240_pct"),
                "current_dividend_yield_pct": qe.get("current_dividend_yield_pct"),
                "dividend_yield_percentile": qe.get("dividend_yield_percentile"),
                "etf_nav": qe.get("etf_nav"),
                "etf_premium_discount_pct": qe.get("etf_premium_discount_pct"),
                "etf_premium_risk_flag": bool(qe.get("etf_premium_risk_flag", False)),
            },
            "technical": {
                "weight": 25,
                "score": round(t_score, 1),
                "ma5": qe.get("ma5"),
                "prev_ma5": qe.get("prev_ma5"),
                "ma20": qe.get("ma20"),
                "ma60": qe.get("ma60"),
                "ma240": qe.get("ma240"),
                "high_20d": qe.get("high_20d"),
                "atr_14": qe.get("atr_14"),
                "bottom_stabilized": bool(qe.get("bottom_stabilized", False)),
                "buy_point_2_status": str(qe.get("buy_point_2_status") or ""),
                "macd_osc": qe.get("macd_osc"),
                "prev_macd_osc": qe.get("prev_macd_osc"),
                "macd_flipped_positive": bool(qe.get("macd_flipped_positive", False)),
                "bb_upper": qe.get("bb_upper"),
                "bb_lower": qe.get("bb_lower"),
                "bb_bandwidth_pct": qe.get("bb_bandwidth_pct"),
                "bb_expanding": bool(qe.get("bb_expanding", False)),
                "ma20_rising": bool(qe.get("ma20_rising", False)),
                "above_rising_ma20": bool(qe.get("above_rising_ma20", False)),
                "three_condition_resonance": three_cond,
                "three_condition_status": str(qe.get("three_condition_status") or ""),
                "broken_ma20_3d": broken_3d,
                "exit_signal_desc": str(qe.get("exit_signal_desc") or ""),
            },
            "chip_macro": {
                "weight": 20,
                "score": round(c_score, 1),
                "foreign_net": (screener_info or {}).get("foreign_net"),
                "trust_net": (screener_info or {}).get("trust_net"),
                "inst_consec_days": (screener_info or {}).get("inst_consec_days"),
                "inst_5d_lots": (screener_info or {}).get("inst_5d_lots"),
            },
            "ai_news": {
                "weight": 20,
                "score": round(ai_news_score, 1),
                "one_sentence": str(core.get("one_sentence") or getattr(result, "analysis_summary", "") or ""),
                "sentiment_summary": str(intel.get("sentiment_summary") or ""),
                "earnings_outlook": str(intel.get("earnings_outlook") or ""),
                "risk_alerts": list(intel.get("risk_alerts") or [])[:5],
                "positive_catalysts": list(intel.get("positive_catalysts") or [])[:5],
                "latest_news": str(intel.get("latest_news") or ""),
                "risk_notes": list(qe.get("risk_notes") or []),
                "plain_indicator_explanations": list(qe.get("plain_indicator_explanations") or []),
            },
        },
        "chart": chart_series or {},
    }


def export_dashboard_data(
    results: Sequence[Any],
    *,
    decisions: Optional[Sequence[Any]] = None,
    chart_map: Optional[Dict[str, Dict[str, Any]]] = None,
    screener_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    portfolio_manager: Optional[PortfolioManager] = None,
    watchlist_codes: Optional[Sequence[str]] = None,
    market_scope: str = "all",
    dashboard_path: Optional[Path] = None,
) -> Path:
    """
    Merge current analysis results into `docs/dashboard_data.json`, preserving the opposite
    market's latest snapshot (so morning US runs keep afternoon TW data and vice versa).
    """
    out_path = Path(dashboard_path) if dashboard_path else _DEFAULT_DASHBOARD_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pm = portfolio_manager or PortfolioManager()
    positions_map = pm.get_positions_map()
    watch_set = {normalize_portfolio_symbol(w) for w in (watchlist_codes or []) if w}
    watch_set.update(pm.get_watchlist_symbols())

    existing_stocks: Dict[str, Dict[str, Any]] = {}
    existing_runs: Dict[str, str] = {}
    if out_path.exists():
        try:
            old_payload = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(old_payload, dict):
                existing_runs = dict(old_payload.get("last_runs") or {})
                for item in old_payload.get("stocks") or []:
                    if isinstance(item, dict) and item.get("symbol"):
                        existing_stocks[str(item["symbol"]).upper()] = item
        except Exception:
            pass

    now_str = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    scope = (market_scope or "all").strip().lower()
    if scope in ("tw", "us"):
        existing_runs[scope] = now_str
    else:
        existing_runs["tw"] = now_str
        existing_runs["us"] = now_str

    # Drop stale entries belonging strictly to the current refreshed scope
    if scope in ("tw", "us"):
        existing_stocks = {
            sym: item
            for sym, item in existing_stocks.items()
            if str(item.get("market") or _infer_market_tag(sym)) != scope
        }
    else:
        existing_stocks = {}

    dec_map = {str(getattr(d, "symbol", "") or "").upper(): d for d in (decisions or [])}
    c_map = {str(k).upper(): v for k, v in (chart_map or {}).items()}
    s_map = {str(k).upper(): v for k, v in (screener_meta or {}).items()}

    for res in results or []:
        sym = str(getattr(res, "code", "") or "").strip().upper()
        if not sym:
            continue
        bare = sym.split(".")[0]
        held_pos = positions_map.get(sym) or positions_map.get(bare)
        in_watch = (sym in watch_set) or (bare in watch_set)
        card = _build_stock_dashboard_card(
            res,
            decision=dec_map.get(sym),
            chart_series=c_map.get(sym) or c_map.get(bare),
            screener_info=s_map.get(sym) or s_map.get(bare),
            held_position=held_pos,
            in_watchlist=in_watch,
            analyzed_at=now_str,
        )
        existing_stocks[sym] = card

    def _sort_key(item: Dict[str, Any]) -> tuple:
        is_held = 1 if item.get("held_position") else 0
        three = 1 if (item.get("dimensions", {}).get("technical", {}).get("three_condition_resonance")) else 0
        trig = (item.get("alert_decision") or {}).get("trigger_type") or ""
        is_alert = 1 if trig else 0
        return (-is_held, -three, -is_alert, -int(item.get("score") or 0), str(item.get("symbol") or ""))

    stocks_list = sorted(existing_stocks.values(), key=_sort_key)

    payload = {
        "version": 2,
        "generated_at": now_str,
        "github_repo": pm.raw_data().get("github_repo", "andyhsu-1986/stock"),
        "last_runs": existing_runs,
        "portfolio": pm.raw_data(),
        "summary": {
            "total_stocks": len(stocks_list),
            "held_count": sum(1 for x in stocks_list if x.get("held_position")),
            "three_cond_count": sum(
                1 for x in stocks_list if x.get("dimensions", {}).get("technical", {}).get("three_condition_resonance")
            ),
            "buy_zone_count": sum(
                1 for x in stocks_list if (x.get("alert_decision") or {}).get("trigger_type") == "buy_zone_touched"
            ),
            "alert_count": sum(
                1 for x in stocks_list if (x.get("alert_decision") or {}).get("should_push")
            ),
        },
        "stocks": stocks_list,
    }

    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)
    logger.info("[Dashboard] 已匯出戰情室網頁資料至 %s（共 %d 檔標的）", out_path, len(stocks_list))
    return out_path
