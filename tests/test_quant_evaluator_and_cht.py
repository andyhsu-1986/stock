# -*- coding: utf-8 -*-
"""
Unit & Integration tests for:
1. Non-TW/US market cleanup verification (no residual A-share/HK fetcher imports)
2. Dual-track quantitative pricing & financial trend engine (engine/quant_evaluator.py)
3. 4-target live evaluation (2 TW stocks: 2330.TW, 2618.TW; 1 TW ETF: 0050.TW; 1 US ETF: VOO)
4. Full Traditional Chinese (Taiwan financial terminology) AI prompt & report output
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider import DataFetcherManager
from engine.quant_evaluator import (
    DEFAULT_QUANT_WEIGHTS,
    QuantEvaluator,
    check_bottom_stabilization,
    compute_atr_14,
    compute_gross_margin_slope_4q,
    compute_revenue_cagr_3y,
    compute_weighted_confidence_score,
    compute_xq_three_conditions,
    format_quant_evaluation_prompt_block,
)
from src.analyzer import AnalysisResult, GeminiAnalyzer
from src.market_context import detect_market, get_market_guidelines
from src.notification import NotificationService


class QuantEvaluatorAndTraditionalChineseTestCase(unittest.TestCase):
    def test_01_market_cleanup_and_no_residual_imports(self):
        """Verify non-TW/US fetchers are removed and DataFetcherManager initializes cleanly."""
        data_provider_dir = Path(__file__).resolve().parents[1] / "data_provider"
        removed_files = [
            "akshare_fetcher.py",
            "efinance_fetcher.py",
            "tushare_fetcher.py",
            "pytdx_fetcher.py",
            "baostock_fetcher.py",
            "tencent_fetcher.py",
            "tickflow_fetcher.py",
            "tickplus_fetcher.py",
            "miaoxiang_fetcher.py",
            "futu_fetcher.py",
        ]
        for fname in removed_files:
            self.assertFalse(
                (data_provider_dir / fname).exists(),
                f"Non-TW/US fetcher {fname} should be removed",
            )

        manager = DataFetcherManager()
        fetcher_names = [f.name for f in manager._fetchers]
        self.assertIn("YfinanceFetcher", fetcher_names)
        self.assertEqual(detect_market("2330.TW"), "tw")
        self.assertEqual(detect_market("0050.TW"), "tw")
        self.assertEqual(detect_market("VOO"), "us")

        tw_rules = get_market_guidelines("2330.TW", "zh")
        us_rules = get_market_guidelines("VOO", "zh")
        self.assertIn("三大法人", tw_rules)
        self.assertIn("本益比", tw_rules)
        self.assertNotIn("北向资金", tw_rules)
        self.assertIn("240 日年線乖離率", us_rules)

    def test_02_modular_math_formulas_and_etf_premium_guardrail(self):
        """Verify 3Y CAGR, 4Q Gross Margin OLS slope, 14d ATR stop loss, and 35/25/20/20 weighted score."""
        # 1. 3Y Revenue CAGR: 100 -> 133.1 over 3 years = +10.0% CAGR
        rev_pairs = []
        for m in range(1, 13):
            rev_pairs.append((f"2023-{m:02d}-01", 100.0))
        for m in range(1, 13):
            rev_pairs.append((f"2024-{m:02d}-01", 110.0))
        for m in range(1, 13):
            rev_pairs.append((f"2025-{m:02d}-01", 121.0))
        for m in range(1, 13):
            rev_pairs.append((f"2026-{m:02d}-01", 133.1))
        cagr = compute_revenue_cagr_3y(rev_pairs)
        self.assertAlmostEqual(cagr, 10.0, places=1)

        # 2. 4Q Gross Margin slope: [40.0, 41.5, 43.0, 44.5] -> +1.5% per quarter
        slope = compute_gross_margin_slope_4q([40.0, 41.5, 43.0, 44.5])
        self.assertAlmostEqual(slope, 1.5, places=2)

        # 3. 14d ATR & Hard 8% Stop Loss Cap: max(20d High - 2.5 * ATR14, Buy_Ref * 0.92)
        df_bars = pd.DataFrame(
            {
                "open": [100.0] * 25,
                "high": [102.0] * 24 + [110.0],
                "low": [98.0] * 24 + [106.0],
                "close": [100.0] * 24 + [108.0],
            }
        )
        atr_14, high_20d, stop_loss, take_profit = compute_atr_14(df_bars, buy_reference_price=105.0)
        self.assertGreater(atr_14, 0.0)
        self.assertEqual(high_20d, 110.0)
        expected_stop = max(round(110.0 - 2.5 * atr_14, 2), round(105.0 * 0.92, 2))
        self.assertAlmostEqual(stop_loss, expected_stop, places=2)
        self.assertGreater(take_profit, 110.0)

        # Case where ATR stop is deeper than -8% from buy_reference_price -> hard 8% cap wins
        df_volatile = pd.DataFrame(
            {
                "open": [100.0] * 25,
                "high": [112.0] * 25,
                "low": [88.0] * 25,
                "close": [100.0] * 25,
            }
        )
        atr_vol, high_vol, stop_vol, _ = compute_atr_14(df_volatile, buy_reference_price=100.0)
        # ATR is 24.0 -> 112 - 2.5*24 = 52.0, while 100 * 0.92 = 92.0 -> must pick 92.0
        self.assertEqual(stop_vol, 92.0)

        # 3b. Bottom Stabilization Check (MA5 flat/rising + 3-day close no new low)
        df_falling_knife = pd.DataFrame(
            {"close": [120, 118, 115, 112, 109, 106, 103, 100, 97, 94]}
        )
        stab_knife = check_bottom_stabilization(df_falling_knife)
        self.assertFalse(stab_knife["ma5_stabilized"])
        self.assertFalse(stab_knife["no_new_low_3d"])
        self.assertFalse(stab_knife["bottom_stabilized"])

        df_stabilized = pd.DataFrame(
            {"close": [110, 106, 102, 100, 99, 98, 98, 99, 100, 101]}
        )
        stab_ok = check_bottom_stabilization(df_stabilized)
        self.assertTrue(stab_ok["ma5_stabilized"])
        self.assertTrue(stab_ok["no_new_low_3d"])
        self.assertTrue(stab_ok["bottom_stabilized"])

        # 3c. XQ Three-Condition Resonance (WeightedClose MACD flip + BB dual expansion + Rising MA20) & 3D Exit
        # Build a base series where 20MA is rising over the last 3 bars, MACD flips from negative to positive,
        # and Bollinger Upper rises while Lower falls (dual-sided expansion)
        closes_breakout = (
            [98.0] * 20
            + [100.5] * 10
            + [99.8] * 8
            + [100.0, 106.0]
        )
        df_breakout = pd.DataFrame(
            {
                "high": [c * 1.01 for c in closes_breakout],
                "low": [c * 0.99 for c in closes_breakout],
                "close": closes_breakout,
            }
        )
        xq_break = compute_xq_three_conditions(df_breakout, ma_up_bars=3, macd_flip_lookback=2)
        self.assertTrue(xq_break["macd_flipped_positive"])
        self.assertTrue(xq_break["bb_expanding"])
        self.assertTrue(xq_break["above_rising_ma20"])
        self.assertTrue(xq_break["three_condition_resonance"])
        self.assertFalse(xq_break["broken_ma20_3d"])

        # Build a breakdown series where the last 3 closes are strictly below MA20
        closes_breakdown = [100.0] * 30 + [95.0, 94.0, 93.0]
        df_breakdown = pd.DataFrame(
            {
                "high": [c * 1.01 for c in closes_breakdown],
                "low": [c * 0.99 for c in closes_breakdown],
                "close": closes_breakdown,
            }
        )
        xq_exit = compute_xq_three_conditions(df_breakdown)
        self.assertTrue(xq_exit["broken_ma20_3d"])
        self.assertFalse(xq_exit["three_condition_resonance"])

        # 4. 35/25/20/20 Weighted Confidence Score & ETF Premium > 1.5% Downgrade
        score_normal, advice_normal = compute_weighted_confidence_score(
            fundamental_score=80.0,  # * 0.35 = 28.0
            technical_score=80.0,    # * 0.25 = 20.0
            chip_macro_score=70.0,   # * 0.20 = 14.0
            ai_news_score=75.0,      # * 0.20 = 15.0 -> total = 77
            etf_premium_risk_flag=False,
        )
        self.assertEqual(score_normal, 77)
        self.assertEqual(advice_normal, "買入")

        score_downgraded, advice_downgraded = compute_weighted_confidence_score(
            fundamental_score=80.0,
            technical_score=80.0,
            chip_macro_score=70.0,
            ai_news_score=75.0,
            etf_premium_risk_flag=True,
        )
        self.assertEqual(advice_downgraded, "持有")
        self.assertLessEqual(score_downgraded, 58)

    def test_03_four_target_evaluation_and_traditional_chinese_report_showcase(self):
        """
        Execute live evaluation on:
          - 2 Taiwan Stocks: 2330.TW (台積電), 2618.TW (長榮航)
          - 1 Taiwan ETF:    0050.TW (元大台灣50)
          - 1 US ETF:        VOO (Vanguard S&P 500 ETF)
        Verify Parquet fundamentals caching and showcase Traditional Chinese report output.
        """
        evaluator = QuantEvaluator()
        analyzer = GeminiAnalyzer()
        notifier = NotificationService()

        targets = [
            ("2330.TW", "台積電"),
            ("2618.TW", "長榮航"),
            ("0050.TW", "元大台灣50"),
            ("VOO", "Vanguard S&P 500 ETF"),
        ]

        results = []
        for symbol, display_name in targets:
            q_eval = evaluator.evaluate(symbol, name=display_name)
            self.assertGreater(q_eval.current_price, 0.0, f"{symbol} current_price must be > 0")
            self.assertGreater(q_eval.buy_point_1, 0.0, f"{symbol} buy_point_1 must be > 0")
            self.assertGreater(q_eval.buy_point_2, 0.0, f"{symbol} buy_point_2 must be > 0")
            self.assertGreater(q_eval.dynamic_stop_loss, 0.0, f"{symbol} dynamic_stop_loss must be > 0")
            self.assertGreaterEqual(
                q_eval.dynamic_stop_loss,
                round(q_eval.buy_point_1 * 0.92, 2),
                f"{symbol} dynamic_stop_loss must respect hard 8% cap from buy_point_1",
            )
            self.assertTrue(bool(q_eval.buy_point_2_status), f"{symbol} must include buy_point_2 stabilization status")
            self.assertIn("30%", q_eval.position_sizing_advice)
            self.assertGreater(q_eval.swing_take_profit, 0.0, f"{symbol} swing_take_profit must be > 0")
            self.assertIn(q_eval.quant_action_advice, ("買入", "持有", "賣出"))

            if symbol in ("2330.TW", "2618.TW"):
                self.assertFalse(q_eval.is_etf)
                self.assertIsNotNone(q_eval.revenue_cagr_3y_pct)
                self.assertIsNotNone(q_eval.gross_margin_slope_4q)
                self.assertIsNotNone(q_eval.pe_5y_mean)
            else:
                self.assertTrue(q_eval.is_etf)
                self.assertIsNotNone(q_eval.bias_ma240_pct)

            # Build AnalysisResult enriched with QuantEvaluator
            res = AnalysisResult(
                code=symbol,
                name=display_name,
                sentiment_score=q_eval.base_confidence_score,
                trend_prediction="看多" if q_eval.technical_score >= 60 else "震盪",
                operation_advice=q_eval.quant_action_advice,
                confidence_level="高" if q_eval.base_confidence_score >= 70 else "中",
                analysis_summary=(
                    f"{display_name}({symbol}) 經雙軌量化引擎評估："
                    f"建議【{q_eval.quant_action_advice}】（最終信心分數 {q_eval.base_confidence_score} 分）。"
                    f"{q_eval.buy_zone_desc}；14日 ATR 動態停損設於 {q_eval.dynamic_stop_loss} {q_eval.currency}。"
                ),
                report_language="zh",
            )
            analyzer._apply_quant_evaluation_enrichment(res, q_eval.to_dict())
            results.append(res)

        # Verify data/tw_fundamentals.parquet was created and cached
        self.assertTrue(
            evaluator.fundamentals_parquet_path.exists(),
            "data/tw_fundamentals.parquet should be created and populated",
        )

        # Generate full Traditional Chinese Dashboard Report & Single-Stock Push Card
        dashboard_report = notifier.generate_dashboard_report(results)
        single_push_card = notifier.generate_single_stock_report(results[0])

        print("\n" + "=" * 80)
        print("【終端機實測展示】4 檔代表性標的（2 台股個股 + 1 台股 ETF + 1 美股 ETF）繁體中文決策儀表板")
        print("=" * 80)
        print(dashboard_report)
        print("\n" + "=" * 80)
        print("【Telegram / Discord 單股推播卡片預覽 (2330.TW 台積電)】")
        print("=" * 80)
        print(single_push_card)

        # Assert Traditional Chinese Taiwan financial terms & beginner-friendly risk controls appear in output
        sys_prompt = analyzer._get_analysis_system_prompt("zh", "2330.TW")
        self.assertIn("白話解釋", sys_prompt)
        self.assertIn("首波試單 30% 資金", sys_prompt)

        for term in (
            "AI 量化決策儀表板",
            "第一買點（順勢拉回）",
            "第二買點（價值支撐）",
            "動態移動停損價",
            "波段停利參考價",
            "近 3 年營收 CAGR",
            "近 4 季毛利率變化斜率",
            "240日年線 (MA240) / 年線乖離率",
            "四維度信心分數歸因 (35/25/20/20)",
            "首波試單 30% 資金",
            "8%",
        ):
            self.assertIn(term, dashboard_report)


if __name__ == "__main__":
    unittest.main()
