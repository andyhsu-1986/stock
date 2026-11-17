# -*- coding: utf-8 -*-
"""Dual-track quantitative valuation and financial trend engine package."""

from engine.quant_evaluator import (
    DEFAULT_QUANT_WEIGHTS,
    QuantEvaluationResult,
    QuantEvaluator,
    QuantWeightConfig,
    compute_atr_14,
    compute_gross_margin_slope_4q,
    compute_revenue_cagr_3y,
    compute_weighted_confidence_score,
)

__all__ = [
    "DEFAULT_QUANT_WEIGHTS",
    "QuantEvaluationResult",
    "QuantEvaluator",
    "QuantWeightConfig",
    "compute_atr_14",
    "compute_gross_margin_slope_4q",
    "compute_revenue_cagr_3y",
    "compute_weighted_confidence_score",
]
