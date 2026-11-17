# -*- coding: utf-8 -*-
"""Canonical score-to-decision scale shared by reports and DecisionSignal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


CANONICAL_DECISION_SCALE_VERSION = "decision-scale-v1"


@dataclass(frozen=True)
class DecisionScaleBand:
    min_score: int
    max_score: int
    signal_key: str
    action: str
    decision_type: str
    label_zh: str
    description_zh: str


CANONICAL_DECISION_SCALE: tuple[DecisionScaleBand, ...] = (
    DecisionScaleBand(80, 100, "strong_buy", "buy", "buy", "買入", "高勝率機會，可執行分批買入計畫"),
    DecisionScaleBand(60, 79, "buy", "buy", "buy", "買入", "偏正向機會，可於第一/第二買點分批布局"),
    DecisionScaleBand(40, 59, "watch", "hold", "hold", "持有", "訊號分歧或等待拉回支撐，建議續抱或觀望"),
    DecisionScaleBand(20, 39, "reduce", "sell", "sell", "賣出", "風險明顯抬升或跌破支撐，優先調節減碼"),
    DecisionScaleBand(0, 19, "sell", "sell", "sell", "賣出", "觸發動態停損或基本面惡化，優先停損退出"),
)


CANONICAL_DECISION_SCALE_PROMPT_ZH = """## 四維度量化信心分數與決策口徑（必須嚴格遵守）

- 最終信心分數 `sentiment_score`（0～100）必須依據以下四維度加權公式計算：
  **最終信心分數 = 35% × 財報基本面/ETF估值位階 + 25% × 技術位階 + 20% × 籌碼與總經 + 20% × AI 新聞一致性**
- `operation_advice` 僅使用繁體中文「**買入**」、「**持有**」、「**賣出**」三種核心建議，並與 `decision_type`（`buy|hold|sell`）一致：
  - **60～100 分**：`operation_advice="買入"`，`action="buy"`，`decision_type="buy"`
  - **40～59 分**：`operation_advice="持有"`，`action="hold"`，`decision_type="hold"`
  - **0～39 分**：`operation_advice="賣出"`，`action="sell"`，`decision_type="sell"`
- **台股 ETF 溢價風險護欄**：當台股 ETF 即時溢價率超過 **1.5%** 時，不論分數高低，嚴禁給出「買入」，必須降級為「持有」並於 `guardrail_reason` 與 `risk_alerts` 標註溢價風險。"""


CANONICAL_DECISION_SCALE_PROMPT_EN = """## Canonical Score and Action Scale

- `sentiment_score`, `operation_advice`, the three-state `decision_type` and the eight-state `action` must follow the same scale.
- 80-100: Strong Buy, `action=buy`, `decision_type=buy`.
- 60-79: Buy, `action=buy`, `decision_type=buy`.
- 40-59: Watch, `action=watch`, `decision_type=hold`.
- 20-39: Reduce, `action=reduce`, `decision_type=sell`.
- 0-19: Sell, `action=sell`, `decision_type=sell`.
- `decision_type` only keeps `buy|hold|sell` for statistics compatibility; finer-grained advice must go into `action`.
- If score >= 60 but the final `action` is `hold/watch`, or score < 40 but the final `action` is `hold/watch`, explain the downgrade in `guardrail_reason` or `dashboard.decision_stability.reason`."""


def normalize_score(value: Any) -> Optional[int]:
    """Return a bounded integer score when possible."""

    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 <= score <= 100:
        return score
    return None


def decision_band_for_score(value: Any) -> Optional[DecisionScaleBand]:
    """Return the canonical decision band for a 0-100 score."""

    score = normalize_score(value)
    if score is None:
        return None
    for band in CANONICAL_DECISION_SCALE:
        if band.min_score <= score <= band.max_score:
            return band
    return None


def signal_key_for_score(value: Any) -> Optional[str]:
    band = decision_band_for_score(value)
    return band.signal_key if band else None


def action_for_score(value: Any) -> Optional[str]:
    band = decision_band_for_score(value)
    return band.action if band else None


def decision_type_for_score(value: Any) -> Optional[str]:
    band = decision_band_for_score(value)
    return band.decision_type if band else None


def score_band_metadata(value: Any) -> dict[str, Any]:
    """Return stable metadata for persistence and diagnostics."""

    score = normalize_score(value)
    band = decision_band_for_score(score)
    if score is None or band is None:
        return {}
    return {
        "scale_version": CANONICAL_DECISION_SCALE_VERSION,
        "score": score,
        "score_band": f"{band.min_score}-{band.max_score}",
        "signal_key": band.signal_key,
        "canonical_action": band.action,
        "canonical_decision_type": band.decision_type,
    }


def extract_decision_guardrail_reason(payload: Any) -> Optional[str]:
    """Extract an applied score/action guardrail reason from a result payload."""

    data = payload if isinstance(payload, Mapping) else {}
    dashboard = data.get("dashboard") if isinstance(data.get("dashboard"), Mapping) else {}
    calibration = (
        dashboard.get("decision_score_calibration")
        if isinstance(dashboard.get("decision_score_calibration"), Mapping)
        else {}
    )
    stability = (
        dashboard.get("decision_stability")
        if isinstance(dashboard.get("decision_stability"), Mapping)
        else {}
    )
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}

    stability_applied = stability.get("applied")
    include_stability_reason = stability_applied not in (False, 0, "0", "false", "False")
    candidates = [
        data.get("guardrail_reason"),
        data.get("downgrade_reason"),
        data.get("decision_score_guardrail_reason"),
        metadata.get("guardrail_reason"),
        metadata.get("downgrade_reason"),
        calibration.get("guardrail_reason"),
        calibration.get("downgrade_reason"),
    ]
    if include_stability_reason:
        candidates.extend(
            [
                stability.get("guardrail_reason"),
                stability.get("downgrade_reason"),
                stability.get("reason"),
            ]
        )

    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def score_action_conflicts_without_guardrail(
    *,
    score: Any,
    action: Any,
    guardrail_reason: Any = None,
) -> bool:
    """Return True when a neutral action conflicts with a directional score."""

    if str(guardrail_reason or "").strip():
        return False
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"hold", "watch"}:
        return False
    score_action = action_for_score(score)
    return score_action in {"buy", "reduce", "sell"}
