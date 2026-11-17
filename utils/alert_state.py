# -*- coding: utf-8 -*-
"""
===================================
新手友善高門檻防洗版警報狀態機與每週紀律週報 (utils/alert_state.py)
===================================

職責：
1. 本地 JSON 狀態機讀寫 (`data/alert_state.json`)：
   - 記錄每檔標的最後一次發送警報的日期 (`last_alert_date`)、建議狀態 (`last_advice` / `last_state_label`) 與參考價 (`last_reference_price`)。
   - 同一標的在 5 個交易日內，若「狀態未改變」且「價格波動幅度 <= 3%」，僅寫入每日 Markdown 總表而不重複推播。
2. 高門檻精選過濾 (High-Threshold Curated Filter)：
   - 即時通訊（Telegram / Discord）僅在「建議為買入且最終信心分數 >= 82 分」或「現價觸及停利 / 停損線」時觸發。
   - 單日買入推播上限為信心分數最高的前 3 檔；觸及停利/停損線之風控警報不受前 3 檔買入上限排擠。
3. 推播卡片繁體中文規範：
   - 每則推播固定包含：標的名稱與代號、信心分數、白話買賣理由（3 點以內）、建議買入區間、停損價、停利目標、新手分批進場資金比例建議。
4. 每週投資紀律週報 (`generate_weekly_discipline_report`)：
   - 週五盤後彙整本週觀察名單之週漲跌幅、信心分數變化、均線位階趨勢與風險提醒。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.sniper_points import parse_sniper_value

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "alert_state.json"
)
_DEFAULT_TW_PARQUET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tw_daily_history.parquet"
)

_MARKET_CALENDAR_MAP = {
    "tw": "XTAI",
    "us": "XNYS",
}


@dataclass(frozen=True)
class AlertFilterConfig:
    """高門檻防洗版與新手風控過濾參數"""

    min_buy_score: int = 82
    max_daily_buy_alerts: int = 3
    cooldown_trading_days: int = 5
    price_move_threshold_pct: float = 3.0
    weekly_retention_days: int = 14
    default_position_advice: str = (
        "首波試單 30% 資金（於第一買點順勢拉回建立基本部位）➔ "
        "第二波加碼 40% 資金（待 5 日均線走平翻揚且近 3 日收盤未破底確認止跌後投入）➔ "
        "保留 30% 機動現金（突破加碼或極端行情備用），單筆最大虧損嚴格限制在 8% 以內"
    )


DEFAULT_ALERT_FILTER_CONFIG = AlertFilterConfig()


@dataclass
class AlertDecision:
    """單一標的的高門檻過濾與防洗版評估結果"""

    symbol: str
    name: str
    market: str
    currency: str
    advice: str                      # "買入" | "持有" | "賣出"
    score: int
    current_price: float
    buy_point_1: float
    buy_point_2: float
    dynamic_stop_loss: float
    swing_take_profit: float
    atr_stop_loss: float = 0.0
    hard_cap_stop_loss: float = 0.0
    buy_point_2_status: str = ""
    position_sizing_advice: str = ""
    reasons: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)

    trigger_type: Optional[str] = None   # "high_confidence_buy" | "stop_loss" | "take_profit" | None
    state_label: str = "持有"            # "買入" | "觸及停損" | "觸及停利" | "持有" | "賣出"
    should_push: bool = False
    skip_reason_code: str = ""           # "" | "below_threshold" | "antispam_cooldown" | "daily_buy_cap"
    skip_reason_desc: str = ""
    price_diff_from_last_pct: Optional[float] = None
    trading_days_since_last: Optional[int] = None


def _infer_market(symbol: str) -> str:
    clean = str(symbol or "").strip().upper()
    return "tw" if clean.endswith((".TW", ".TWO")) else "us"


def _canonical_advice(raw_advice: Any, decision_type: Any = None) -> str:
    text = str(raw_advice or "").strip().lower()
    dt = str(decision_type or "").strip().lower()
    if any(k in text for k in ("買入", "买入", "加碼", "加仓", "strong buy", "buy", "add")):
        return "買入"
    if any(k in text for k in ("賣出", "卖出", "減碼", "减仓", "避開", "strong sell", "sell", "reduce")):
        return "賣出"
    if dt == "buy":
        return "買入"
    if dt == "sell":
        return "賣出"
    return "持有"


def count_trading_days_between(
    start_date: date | str,
    end_date: date | str,
    market: str = "tw",
) -> int:
    """
    Count actual trading sessions strictly after `start_date` up to & including `end_date`.
    Returns 0 if `end_date <= start_date`.
    Uses exchange_calendars (XTAI / XNYS) with Mon-Fri weekday fallback.
    """
    start_dt = date.fromisoformat(str(start_date)[:10]) if isinstance(start_date, str) else start_date
    end_dt = date.fromisoformat(str(end_date)[:10]) if isinstance(end_date, str) else end_date
    if end_dt <= start_dt:
        return 0

    ex_code = _MARKET_CALENDAR_MAP.get(market, "XTAI")
    try:
        import exchange_calendars as xcals

        cal = xcals.get_calendar(ex_code)
        sessions = cal.sessions_in_range(
            pd.Timestamp(start_dt + timedelta(days=1)),
            pd.Timestamp(end_dt),
        )
        return int(len(sessions))
    except Exception:
        days = 0
        cur = start_dt + timedelta(days=1)
        while cur <= end_dt:
            if cur.weekday() < 5:
                days += 1
            cur += timedelta(days=1)
        return days


def _extract_plain_reasons(
    result: Any,
    quant_eval: Dict[str, Any],
    *,
    trigger_type: Optional[str],
    current_price: float,
    dynamic_stop_loss: float,
    swing_take_profit: float,
    currency: str,
) -> List[str]:
    """Extract at most 3 beginner-friendly Traditional Chinese bullet points explaining why to act."""
    bullets: List[str] = []

    if trigger_type == "stop_loss":
        bullets.append(
            f"現價 **{current_price} {currency}** 已觸及或跌破嚴格停損線 **{dynamic_stop_loss} {currency}**，"
            "優先執行停損紀律以防止單筆虧損擴大。"
        )
    elif trigger_type == "take_profit":
        bullets.append(
            f"現價 **{current_price} {currency}** 已達成波段停利目標 **{swing_take_profit} {currency}**，"
            "建議分批獲利入袋或將移動停損點上調保護既得獲利。"
        )

    # Pull quantitative plain-English highlights
    cagr = quant_eval.get("revenue_cagr_3y_pct")
    gm_slope = quant_eval.get("gross_margin_slope_4q")
    bias_240 = quant_eval.get("bias_ma240_pct")
    yield_pctile = quant_eval.get("dividend_yield_percentile")

    if quant_eval.get("is_etf"):
        if isinstance(bias_240, (int, float)):
            status = "位於長線年線合理區間" if -5.0 <= float(bias_240) <= 10.0 else "位階偏離 240 日年線"
            bullets.append(f"240日年線乖離率 **{bias_240:+.2f}%**（{status}），代表目前股價與長線平均成本的距離。")
        if isinstance(yield_pctile, (int, float)):
            bullets.append(f"歷史殖利率位於近 5 年 **{yield_pctile:.1f}%** 分位，分位越高代表股息防禦力越強。")
    else:
        if isinstance(cagr, (int, float)):
            bullets.append(f"近 3 年營收複合成長率 **{cagr:.2f}%**，搭配毛利率季斜率 **{gm_slope if gm_slope is not None else 'N/A'}%/季**，檢視本業成長含金量。")

    # Add concise LLM / summary reason text
    raw_candidates = [
        getattr(result, "buy_reason", "") or "",
        getattr(result, "analysis_summary", "") or "",
    ]
    dash = getattr(result, "dashboard", None)
    if isinstance(dash, dict):
        core = dash.get("core_conclusion")
        if isinstance(core, dict) and core.get("one_sentence"):
            raw_candidates.insert(0, str(core["one_sentence"]))
        intel = dash.get("intelligence")
        if isinstance(intel, dict):
            cats = intel.get("positive_catalysts") or []
            if cats:
                raw_candidates.append(str(cats[0]))

    for cand in raw_candidates:
        if len(bullets) >= 3:
            break
        cleaned = re.sub(r"\s+", " ", str(cand or "")).strip()
        if not cleaned or len(cleaned) < 6:
            continue
        if len(cleaned) > 90:
            cleaned = cleaned[:87].rstrip("，。、 ") + "..."
        if cleaned not in bullets:
            bullets.append(cleaned)

    if not bullets:
        bullets.append("依雙軌量化模型與均線支撐綜合評分達標，建議依風控點位分批紀律執行。")

    return bullets[:3]


def _extract_numeric_level(quant_eval: Dict[str, Any], dash: Dict[str, Any], q_key: str, sniper_key: str) -> float:
    val = quant_eval.get(q_key)
    if isinstance(val, (int, float)) and float(val) > 0:
        return round(float(val), 2)
    battle = dash.get("battle_plan") if isinstance(dash, dict) else {}
    sniper = battle.get("sniper_points") if isinstance(battle, dict) else {}
    if isinstance(sniper, dict) and sniper.get(sniper_key):
        parsed = parse_sniper_value(str(sniper[sniper_key]))
        if parsed and parsed > 0:
            return round(float(parsed), 2)
    return 0.0


class AlertStateManager:
    """
    Persistent JSON state machine for high-threshold curated notifications
    and 5-trading-day / 3%-move anti-spam suppression.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        config: AlertFilterConfig = DEFAULT_ALERT_FILTER_CONFIG,
    ) -> None:
        self.state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self.config = config
        self._state: Dict[str, Any] = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        """Load JSON state from disk safely, recovering cleanly if absent or corrupt."""
        if not self.state_path.exists():
            return {"version": 1, "updated_at": "", "alerts": {}, "weekly_snapshots": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("alert_state root must be a dict")
            payload.setdefault("version", 1)
            payload.setdefault("updated_at", "")
            payload.setdefault("alerts", {})
            payload.setdefault("weekly_snapshots", {})
            return payload
        except Exception as exc:
            logger.warning("[AlertState] 讀取狀態檔失敗 (%s)，重新初始化: %s", self.state_path, exc)
            return {"version": 1, "updated_at": "", "alerts": {}, "weekly_snapshots": {}}

    def save_state(self) -> None:
        """Atomically persist state JSON to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.state_path)

    def get_last_alert(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean = str(symbol or "").strip().upper()
        alerts = self._state.get("alerts") or {}
        rec = alerts.get(clean)
        return dict(rec) if isinstance(rec, dict) else None

    def _build_decision(self, result: Any, reference_date: date) -> AlertDecision:
        symbol = str(getattr(result, "code", "") or "").strip().upper()
        name = str(getattr(result, "name", "") or symbol).strip()
        market = _infer_market(symbol)
        dash = getattr(result, "dashboard", None)
        dash = dash if isinstance(dash, dict) else {}
        quant_eval = dash.get("quant_evaluation")
        quant_eval = quant_eval if isinstance(quant_eval, dict) else {}

        currency = str(quant_eval.get("currency") or ("TWD" if market == "tw" else "USD"))
        advice = _canonical_advice(
            getattr(result, "operation_advice", "持有"),
            getattr(result, "decision_type", None),
        )
        try:
            score = int(round(float(getattr(result, "sentiment_score", 0) or 0)))
        except (TypeError, ValueError):
            score = 0

        current_price = float(
            getattr(result, "current_price", None)
            or quant_eval.get("current_price")
            or 0.0
        )
        current_price = round(current_price, 2)

        buy_point_1 = _extract_numeric_level(quant_eval, dash, "buy_point_1", "ideal_buy")
        buy_point_2 = _extract_numeric_level(quant_eval, dash, "buy_point_2", "secondary_buy")
        dynamic_stop_loss = _extract_numeric_level(quant_eval, dash, "dynamic_stop_loss", "stop_loss")
        swing_take_profit = _extract_numeric_level(quant_eval, dash, "swing_take_profit", "take_profit")
        atr_stop_loss = float(quant_eval.get("atr_stop_loss") or dynamic_stop_loss or 0.0)
        hard_cap_stop_loss = float(quant_eval.get("hard_cap_stop_loss") or (round(buy_point_1 * 0.92, 2) if buy_point_1 > 0 else 0.0))
        buy_point_2_status = str(quant_eval.get("buy_point_2_status") or "")
        position_sizing = str(quant_eval.get("position_sizing_advice") or self.config.default_position_advice)
        risk_notes = [str(x) for x in (quant_eval.get("risk_notes") or []) if str(x).strip()]

        # Determine high-threshold trigger eligibility:
        # 1. Price hit Stop-Loss (<= dynamic_stop_loss) or Take-Profit (>= swing_take_profit)
        # 2. Advice == "買入" AND score >= 82
        trigger_type: Optional[str] = None
        state_label = advice
        if current_price > 0 and dynamic_stop_loss > 0 and current_price <= dynamic_stop_loss:
            trigger_type = "stop_loss"
            state_label = "觸及停損"
        elif current_price > 0 and swing_take_profit > 0 and current_price >= swing_take_profit:
            trigger_type = "take_profit"
            state_label = "觸及停利"
        elif advice == "買入" and score >= self.config.min_buy_score:
            trigger_type = "high_confidence_buy"
            state_label = "買入"

        reasons = _extract_plain_reasons(
            result,
            quant_eval,
            trigger_type=trigger_type,
            current_price=current_price,
            dynamic_stop_loss=dynamic_stop_loss,
            swing_take_profit=swing_take_profit,
            currency=currency,
        )

        decision = AlertDecision(
            symbol=symbol,
            name=name,
            market=market,
            currency=currency,
            advice=advice,
            score=score,
            current_price=current_price,
            buy_point_1=buy_point_1,
            buy_point_2=buy_point_2,
            dynamic_stop_loss=dynamic_stop_loss,
            swing_take_profit=swing_take_profit,
            atr_stop_loss=round(atr_stop_loss, 2),
            hard_cap_stop_loss=round(hard_cap_stop_loss, 2),
            buy_point_2_status=buy_point_2_status,
            position_sizing_advice=position_sizing,
            reasons=reasons,
            risk_notes=risk_notes,
            trigger_type=trigger_type,
            state_label=state_label,
        )

        if trigger_type is None:
            decision.should_push = False
            decision.skip_reason_code = "below_threshold"
            decision.skip_reason_desc = (
                f"建議為「{advice}」且分數 {score} 分（未達買入≥{self.config.min_buy_score}分，亦未觸及停利/停損線），僅收錄於每日 Markdown 總表"
            )
            return decision

        # Consult 5-trading-day / 3% move anti-spam state machine
        prev = self.get_last_alert(symbol)
        if prev:
            prev_date_str = str(prev.get("last_alert_date") or "").strip()
            prev_state = str(prev.get("last_state_label") or prev.get("last_advice") or "").strip()
            prev_price = float(prev.get("last_reference_price") or 0.0)
            elapsed_days: Optional[int] = None
            if prev_date_str:
                try:
                    elapsed_days = count_trading_days_between(prev_date_str, reference_date, market=market)
                except ValueError:
                    elapsed_days = None

            price_diff_pct: Optional[float] = None
            if prev_price > 0 and current_price > 0:
                price_diff_pct = round(abs(current_price - prev_price) / prev_price * 100.0, 2)

            decision.trading_days_since_last = elapsed_days
            decision.price_diff_from_last_pct = price_diff_pct

            within_cooldown = elapsed_days is not None and 0 <= elapsed_days < self.config.cooldown_trading_days
            state_unchanged = bool(prev_state and prev_state == state_label)
            small_price_move = (
                price_diff_pct is not None and price_diff_pct <= self.config.price_move_threshold_pct
            )
            if within_cooldown and state_unchanged and small_price_move:
                decision.should_push = False
                decision.skip_reason_code = "antispam_cooldown"
                decision.skip_reason_desc = (
                    f"防洗版抑制：距前次推播僅 {elapsed_days} 個交易日（< {self.config.cooldown_trading_days} 日）、"
                    f"狀態同為「{state_label}」且波動僅 {price_diff_pct:.2f}%（≤ {self.config.price_move_threshold_pct:.1f}%），僅寫入每日 Markdown 總表"
                )
                return decision

        decision.should_push = True
        return decision

    def evaluate_batch(
        self,
        results: Sequence[Any],
        *,
        reference_date: Optional[date] = None,
    ) -> List[AlertDecision]:
        """
        Evaluate a batch of `AnalysisResult` objects against:
          1. High-threshold trigger filter (Buy >= 82 OR hit SL/TP)
          2. 5-trading-day / 3% move anti-spam cooldown
          3. Daily Top-3 Buy alert cap (highest confidence scores first)
        """
        ref_date = reference_date or datetime.now(ZoneInfo("Asia/Taipei")).date()
        decisions = [self._build_decision(r, ref_date) for r in (results or [])]

        # Enforce daily max 3 Buy alerts (highest confidence scores win)
        buy_candidates = [
            d for d in decisions if d.should_push and d.trigger_type == "high_confidence_buy"
        ]
        buy_candidates.sort(key=lambda d: (-d.score, d.symbol))
        for idx, cand in enumerate(buy_candidates):
            if idx >= self.config.max_daily_buy_alerts:
                cand.should_push = False
                cand.skip_reason_code = "daily_buy_cap"
                cand.skip_reason_desc = (
                    f"今日高信心買入訊號眾多，已達單日推播上限（分數最高前 {self.config.max_daily_buy_alerts} 檔），本檔收錄於每日 Markdown 總表"
                )

        return decisions

    def record_pushed_alerts(
        self,
        decisions: Sequence[AlertDecision],
        *,
        reference_date: Optional[date] = None,
    ) -> None:
        """Persist pushed alert state (`alerts[symbol]`) into JSON."""
        ref_date = reference_date or datetime.now(ZoneInfo("Asia/Taipei")).date()
        date_str = ref_date.strftime("%Y-%m-%d")
        alerts = self._state.setdefault("alerts", {})
        changed = False

        for d in decisions or []:
            if not getattr(d, "should_push", False):
                continue
            alerts[d.symbol] = {
                "symbol": d.symbol,
                "name": d.name,
                "market": d.market,
                "last_alert_date": date_str,
                "last_advice": d.advice,
                "last_state_label": d.state_label,
                "last_trigger_type": d.trigger_type,
                "last_reference_price": round(float(d.current_price), 2),
                "last_score": int(d.score),
                "last_stop_loss": round(float(d.dynamic_stop_loss), 2),
                "last_take_profit": round(float(d.swing_take_profit), 2),
            }
            changed = True

        if changed:
            self.save_state()

    def record_daily_snapshots(
        self,
        results: Sequence[Any],
        decisions: Optional[Sequence[AlertDecision]] = None,
        *,
        reference_date: Optional[date] = None,
    ) -> None:
        """Record daily observation snapshots (kept for 14 days) for Friday Weekly Discipline Report."""
        ref_date = reference_date or datetime.now(ZoneInfo("Asia/Taipei")).date()
        date_str = ref_date.strftime("%Y-%m-%d")
        decision_map = {d.symbol: d for d in (decisions or [])}

        snapshots = self._state.setdefault("weekly_snapshots", {})
        day_bucket = snapshots.setdefault(date_str, {})

        for res in results or []:
            sym = str(getattr(res, "code", "") or "").strip().upper()
            if not sym:
                continue
            d = decision_map.get(sym)
            if d is None:
                d = self._build_decision(res, ref_date)

            day_bucket[sym] = {
                "date": date_str,
                "symbol": sym,
                "name": d.name,
                "market": d.market,
                "currency": d.currency,
                "current_price": d.current_price,
                "advice": d.advice,
                "state_label": d.state_label,
                "score": d.score,
                "buy_point_1": d.buy_point_1,
                "buy_point_2": d.buy_point_2,
                "dynamic_stop_loss": d.dynamic_stop_loss,
                "swing_take_profit": d.swing_take_profit,
                "buy_point_2_status": d.buy_point_2_status,
                "risk_notes": list(d.risk_notes),
                "trigger_type": d.trigger_type,
                "alert_pushed": bool(d.should_push),
                "skip_reason_code": d.skip_reason_code,
            }

        # Prune older buckets beyond retention_days
        cutoff = (ref_date - timedelta(days=self.config.weekly_retention_days)).strftime("%Y-%m-%d")
        for old_key in [k for k in list(snapshots.keys()) if k < cutoff]:
            snapshots.pop(old_key, None)

        self.save_state()


# ---------------------------------------------------------------------------
# Curated Traditional Chinese Push Card Formatters (Telegram / Discord)
# ---------------------------------------------------------------------------
def format_curated_alert_card(d: AlertDecision) -> str:
    """Render a beginner-friendly Traditional Chinese alert card for Telegram/Discord."""
    if d.trigger_type == "stop_loss":
        badge = f"🚨 【觸及停損警戒】{d.name} ({d.symbol})"
    elif d.trigger_type == "take_profit":
        badge = f"🎊 【觸及停利目標】{d.name} ({d.symbol})"
    else:
        badge = f"🟢 【高信心買入精選】{d.name} ({d.symbol})"

    if d.buy_point_2 > 0 and d.buy_point_1 > 0:
        buy_zone = f"{d.buy_point_2} ～ {d.buy_point_1} {d.currency}"
    elif d.buy_point_1 > 0:
        buy_zone = f"{d.buy_point_1} {d.currency} 附近"
    else:
        buy_zone = f"{d.current_price} {d.currency} 附近"

    if d.buy_point_2_status:
        buy_zone = f"{buy_zone}（{d.buy_point_2_status}）"

    reason_lines = "\n".join(
        f"  {idx}. {reason}" for idx, reason in enumerate((d.reasons or [])[:3], start=1)
    )

    lines = [
        f"### {badge}",
        f"- **標的名稱與代號**：**{d.name} ({d.symbol})**",
        f"- **最終信心分數**：**{d.score} / 100** ｜ 目前建議：**{d.state_label}**（現價 `{d.current_price} {d.currency}`）",
        "- **💡 白話買賣理由（重點整理）**：",
        reason_lines,
        f"- **🎯 建議買入區間**：{buy_zone}",
        f"- **🛑 停損價**：**{d.dynamic_stop_loss} {d.currency}**（嚴守 20 日高點減 2.5×ATR 與買入價往下 8% 上限）",
        f"- **🎊 停利目標**：**{d.swing_take_profit} {d.currency}**",
        f"- **💰 新手分批進場資金比例建議**：{d.position_sizing_advice}",
    ]
    return "\n".join(lines)


def format_curated_alert_digest(
    pushed_decisions: Sequence[AlertDecision],
    *,
    report_date: Optional[str] = None,
    all_decisions: Optional[Sequence[AlertDecision]] = None,
) -> str:
    """Format pushed curated alert cards into a concise Traditional Chinese IM digest."""
    dt_str = report_date or datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    pushed = list(pushed_decisions or [])
    buy_cnt = sum(1 for d in pushed if d.trigger_type == "high_confidence_buy")
    risk_cnt = sum(1 for d in pushed if d.trigger_type in ("stop_loss", "take_profit"))
    suppressed_cnt = sum(
        1 for d in (all_decisions or []) if not d.should_push
    )

    header = [
        f"# 🎯 {dt_str} 高勝率精選訊號快報（防洗版過濾）",
        "",
        f"> 本日精選推播：**{len(pushed)} 檔**（高信心買入 `{buy_cnt}` 檔 / 停利停損警示 `{risk_cnt}` 檔）"
        + (f" ｜ 已過濾低勝率或重複雜訊 `{suppressed_cnt}` 檔" if all_decisions else ""),
        "",
        "---",
        "",
    ]
    cards = [format_curated_alert_card(d) for d in pushed]
    footer = [
        "",
        "---",
        "*💡 新手紀律提醒：不追高、不於下跌段未止跌前接刀；單筆最大風險嚴守 8%，完整分析請查閱每日 Markdown 報告。*",
    ]
    return "\n".join(header) + "\n\n---\n\n".join(cards) + "\n" + "\n".join(footer)


def format_daily_alert_summary_section(decisions: Sequence[AlertDecision]) -> str:
    """Render a summary Markdown table of push vs anti-spam filtering status for the daily report."""
    if not decisions:
        return ""
    lines = [
        "## 🛡️ 今日高門檻精選推播與防洗版過濾紀錄",
        "",
        "| 標的 (代號) | 現價 | 建議 / 狀態 | 信心分數 | 即時推播狀態 | 說明 |",
        "|------|------|------|------|------|------|",
    ]
    for d in sorted(decisions, key=lambda x: (-int(x.should_push), -x.score, x.symbol)):
        push_badge = "📣 **已發送推播**" if d.should_push else "📄 僅收錄總表"
        desc = (
            f"觸發 `{d.trigger_type}`"
            if d.should_push
            else (d.skip_reason_desc or "未達高門檻條件")
        )
        lines.append(
            f"| {d.name} ({d.symbol}) | {d.current_price} {d.currency} | {d.state_label} | {d.score} | {push_badge} | {desc} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Friday Weekly Investment Discipline Report Builder (每週投資紀律週報)
# ---------------------------------------------------------------------------
def generate_weekly_discipline_report(
    *,
    state_manager: Optional[AlertStateManager] = None,
    parquet_path: Optional[Path] = None,
    reference_date: Optional[date] = None,
    watchlist_codes: Optional[Sequence[str]] = None,
    extra_results: Optional[Sequence[Any]] = None,
) -> str:
    """
    Build the Friday Post-Market '每週投資紀律週報' (Weekly Investment Discipline Report).

    Combines:
      1. Persisted daily snapshots from `data/alert_state.json` (across the current week Mon-Fri)
      2. Historical bars & institutional net flow from `data/tw_daily_history.parquet` (if available)
      3. Optional `extra_results` from the current Friday run.
    """
    mgr = state_manager or AlertStateManager()
    ref_date = reference_date or datetime.now(ZoneInfo("Asia/Taipei")).date()
    week_start = ref_date - timedelta(days=ref_date.weekday())  # Monday
    week_end = ref_date
    start_str = week_start.strftime("%Y-%m-%d")
    end_str = week_end.strftime("%Y-%m-%d")

    snapshots = (mgr._state or {}).get("weekly_snapshots") or {}
    week_dates = sorted(d for d in snapshots.keys() if start_str <= d <= end_str)

    # Collect per-symbol weekly timeline {symbol: [day_snapshot, ...]}
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    pushed_buy_count = 0
    pushed_risk_count = 0
    suppressed_count = 0

    for d_str in week_dates:
        bucket = snapshots.get(d_str) or {}
        for sym, snap in bucket.items():
            by_symbol.setdefault(sym, []).append(snap)
            if snap.get("alert_pushed"):
                if snap.get("trigger_type") == "high_confidence_buy":
                    pushed_buy_count += 1
                else:
                    pushed_risk_count += 1
            elif snap.get("skip_reason_code") == "antispam_cooldown":
                suppressed_count += 1

    # Merge any extra current results not yet stored
    if extra_results:
        for res in extra_results:
            sym = str(getattr(res, "code", "") or "").strip().upper()
            if sym and sym not in by_symbol:
                dec = mgr._build_decision(res, ref_date)
                by_symbol[sym] = [{
                    "date": end_str,
                    "symbol": sym,
                    "name": dec.name,
                    "market": dec.market,
                    "currency": dec.currency,
                    "current_price": dec.current_price,
                    "advice": dec.advice,
                    "state_label": dec.state_label,
                    "score": dec.score,
                    "buy_point_1": dec.buy_point_1,
                    "buy_point_2": dec.buy_point_2,
                    "dynamic_stop_loss": dec.dynamic_stop_loss,
                    "swing_take_profit": dec.swing_take_profit,
                    "risk_notes": dec.risk_notes,
                }]

    # Enrich Taiwan weekly prices & 5-day institutional flows from tw_daily_history.parquet
    pq_path = Path(parquet_path) if parquet_path else _DEFAULT_TW_PARQUET_PATH
    tw_pq_stats: Dict[str, Dict[str, Any]] = {}
    if pq_path.exists():
        try:
            pq_df = pd.read_parquet(pq_path)
            if not pq_df.empty and "symbol" in pq_df.columns and "date" in pq_df.columns:
                pq_df["date"] = pq_df["date"].astype(str)
                week_pq = pq_df[(pq_df["date"] >= start_str) & (pq_df["date"] <= end_str)].sort_values(["symbol", "date"])
                target_syms = set(by_symbol.keys()) | {
                    str(c).strip().upper() for c in (watchlist_codes or []) if str(c).strip()
                }
                for sym, grp in week_pq.groupby("symbol"):
                    sym_u = str(sym).strip().upper()
                    if target_syms and sym_u not in target_syms:
                        continue
                    closes = pd.to_numeric(grp["close"], errors="coerce").dropna().tolist()
                    inst_sum_lots = int(round(pd.to_numeric(grp["inst_total_net"], errors="coerce").fillna(0).sum() / 1000.0))
                    if closes:
                        tw_pq_stats[sym_u] = {
                            "name": str(grp.iloc[-1].get("name") or sym_u),
                            "start_price": round(float(closes[0]), 2),
                            "end_price": round(float(closes[-1]), 2),
                            "inst_week_lots": inst_sum_lots,
                        }
        except Exception as exc:
            logger.warning("[WeeklyReport] 讀取 %s 失敗 (Fail-Open): %s", pq_path, exc)

    all_symbols = sorted(set(by_symbol.keys()) | set(tw_pq_stats.keys()))
    lines = [
        f"# 📅 每週投資紀律週報（{start_str} ～ {end_str}）",
        "",
        "## 1. 本週執行紀律與訊號統計",
        f"- **追蹤觀察標的總數**：**{len(all_symbols)} 檔**（含台美股固定觀察名單與量化初篩標的）",
        f"- **高門檻即時推播統計**：高信心買入精選（≥82分） `{pushed_buy_count}` 次 ｜ 停利/停損警示 `{pushed_risk_count}` 次",
        f"- **防洗版過濾統計**：成功抑制 5 個交易日內重複低波動雜訊共 `{suppressed_count}` 次",
        "- **核心紀律準則**：單筆最大虧損不超過買入參考價 **8%**、未完成止跌確認前不搶反彈、嚴格執行 **首波 30% / 加碼 40% / 保留 30% 現金** 分批佈局。",
        "",
        "## 2. 本週觀察名單趨勢變化總覽",
        "| 標的 (代號) | 期初 → 最新收盤 (週漲跌幅) | 信心分數變化 | 最新建議 / 狀態 | 關鍵風控點位（第一買點 / 動態停損） |",
        "|------|------|------|------|------|",
    ]

    risk_alerts_collected: List[str] = []

    if not all_symbols:
        lines.append("| 尚無本週紀錄 | - | - | 持有觀望 | 請確認每日排程已執行 |")
    else:
        for sym in all_symbols:
            hist = sorted(by_symbol.get(sym, []), key=lambda x: str(x.get("date") or ""))
            pq = tw_pq_stats.get(sym, {})

            if hist:
                first, latest = hist[0], hist[-1]
                name = str(latest.get("name") or pq.get("name") or sym)
                currency = str(latest.get("currency") or ("TWD" if sym.endswith((".TW", ".TWO")) else "USD"))
                start_p = float(first.get("current_price") or pq.get("start_price") or 0.0)
                end_p = float(latest.get("current_price") or pq.get("end_price") or 0.0)
                start_score = int(first.get("score") or 0)
                end_score = int(latest.get("score") or 0)
                score_str = f"{start_score} → **{end_score}** ({end_score - start_score:+d})"
                state_str = f"{latest.get('state_label') or latest.get('advice', '持有')}"
                bp1 = latest.get("buy_point_1", "N/A")
                sl = latest.get("dynamic_stop_loss", "N/A")
                key_levels = f"買點 {bp1} / 停損 {sl} {currency}"
                for rn in latest.get("risk_notes") or []:
                    risk_alerts_collected.append(f"**{name} ({sym})**：{rn}")
            else:
                name = str(pq.get("name") or sym)
                currency = "TWD" if sym.endswith((".TW", ".TWO")) else "USD"
                start_p = float(pq.get("start_price") or 0.0)
                end_p = float(pq.get("end_price") or 0.0)
                score_str = "N/A"
                state_str = f"三大法人週買超 {pq.get('inst_week_lots', 0):+,d} 張"
                key_levels = "請參考每日量化報告"

            if start_p > 0 and end_p > 0:
                chg_pct = (end_p - start_p) / start_p * 100.0
                price_str = f"{start_p:.2f} → {end_p:.2f} {currency} (**{chg_pct:+.2f}%**)"
            else:
                price_str = f"{end_p:.2f} {currency}"

            lines.append(
                f"| {name} ({sym}) | {price_str} | {score_str} | {state_str} | {key_levels} |"
            )

    lines.extend([
        "",
        "## 3. 重點風險提醒與下週注意事項",
    ])
    if risk_alerts_collected:
        for alert in dict.fromkeys(risk_alerts_collected):
            lines.append(f"- ⚠️ {alert}")
    else:
        lines.append("- ✅ 本週觀察名單暫無觸及重大停損或 ETF 溢價警示，持續維持分批紀律即可。")

    lines.extend([
        "",
        "## 4. 新手每週投資紀律檢核清單",
        "1. **不追高、不猜底**：股價距離月線/年線乖離過大時不追價；回測第二買點時必須確認「5 日均線走平或翻揚」且「近 3 日收盤未破底」才承接。",
        "2. **單筆虧損上限 8% 鐵律**：若跌破「20 日高點減 2.5 倍 ATR」與「買入價往下 8%」之較高停損線，務必果斷執行停損，絕不凹單攤平。",
        "3. **資金分批控管**：看好標的最多先投入 **首波試單 30% 資金**，確認趨勢站穩再加碼 **40% 資金**，永遠保留至少 **30% 機動現金**。",
        "",
        f"*週報產出時間：{datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')} (Asia/Taipei)*",
    ])
    return "\n".join(lines)
