# -*- coding: utf-8 -*-
"""
Unit & Integration tests for:
1. utils/alert_state.py: Beginner-friendly anti-spam alert state machine
   (5-trading-day / 3% move suppression, >=82 Buy / SL / TP trigger, top-3 daily Buy cap,
    curated Traditional Chinese push cards, atomic JSON read/write)
2. Friday Weekly Investment Discipline Report generation (每週投資紀律週報)
3. GitHub Actions dual-timezone workflows (tw_market_daily.yml, us_etf_daily.yml)
   & legacy schedule cleanup + permissions & secret injection checks.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.analyzer import AnalysisResult


ROOT_DIR = Path(__file__).resolve().parents[1]


def _make_mock_result(
    code: str,
    name: str,
    *,
    score: int,
    advice: str = "買入",
    current_price: float = 100.0,
    buy_point_1: float = 98.0,
    buy_point_2: float = 93.0,
    dynamic_stop_loss: float = 90.0,
    swing_take_profit: float = 115.0,
    risk_notes: list[str] | None = None,
) -> AnalysisResult:
    market = "tw" if code.endswith((".TW", ".TWO")) else "us"
    currency = "TWD" if market == "tw" else "USD"
    quant_eval = {
        "symbol": code,
        "name": name,
        "market": market,
        "asset_track": "tw_stock" if market == "tw" else "us_etf",
        "is_etf": code.startswith("00") or code in ("VOO", "QQQ"),
        "currency": currency,
        "as_of_date": "2026-11-17",
        "current_price": current_price,
        "ma5": round(current_price * 0.99, 2),
        "ma20": round(current_price * 0.97, 2),
        "ma60": round(current_price * 0.94, 2),
        "ma240": round(current_price * 0.88, 2),
        "high_20d": round(current_price * 1.03, 2),
        "atr_14": 2.0,
        "buy_point_1": buy_point_1,
        "buy_point_2": buy_point_2,
        "atr_stop_loss": dynamic_stop_loss,
        "hard_cap_stop_loss": round(buy_point_1 * 0.92, 2),
        "dynamic_stop_loss": dynamic_stop_loss,
        "swing_take_profit": swing_take_profit,
        "buy_point_2_status": "✅ 止跌確認通過（5日均線走平翻揚且近3日收盤未破底）",
        "position_sizing_advice": (
            "首波試單 30% 資金（於第一買點順勢拉回建立基本部位）➔ "
            "第二波加碼 40% 資金（待 5 日均線走平翻揚且近 3 日收盤未破底確認止跌後投入）➔ "
            "保留 30% 機動現金"
        ),
        "revenue_cagr_3y_pct": 18.5,
        "gross_margin_slope_4q": 1.2,
        "operating_cash_flow_ratio_pct": 120.0,
        "fundamental_score": float(min(100, score)),
        "technical_score": float(min(100, score)),
        "chip_macro_score": 75.0,
        "risk_notes": risk_notes or [],
    }
    return AnalysisResult(
        code=code,
        name=name,
        sentiment_score=score,
        trend_prediction="看多" if score >= 65 else "震盪",
        operation_advice=advice,
        confidence_level="高" if score >= 80 else "中",
        current_price=current_price,
        analysis_summary=f"{name}({code}) 基本面穩健，股價站穩均線上方。",
        buy_reason="營收與毛利率持續成長，且三大法人站在買方，均線呈現多頭支撐。",
        report_language="zh",
        dashboard={
            "quant_evaluation": quant_eval,
            "core_conclusion": {
                "one_sentence": f"{name}({code}) 結構偏多，可依第一買點分批承接。",
            },
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": f"{buy_point_1} {currency}",
                    "secondary_buy": f"{buy_point_2} {currency}",
                    "stop_loss": f"{dynamic_stop_loss} {currency}",
                    "take_profit": f"{swing_take_profit} {currency}",
                },
                "position_strategy": {
                    "suggested_position": "首波試單 30% 資金",
                    "entry_plan": quant_eval["position_sizing_advice"],
                },
            },
            "intelligence": {
                "risk_alerts": risk_notes or [],
                "positive_catalysts": ["近3年營收穩健成長", "法人買盤支撐"],
            },
        },
    )


class AlertStateAndWorkflowsTestCase(unittest.TestCase):
    def test_01_high_threshold_filtering_top3_cap_and_json_persistence(self):
        """
        Verify:
        - Buy < 82 is excluded from IM push
        - Buy >= 82 is included up to Top 3 highest scores
        - SL/TP trigger is included even when score < 82 and does not eat Top-3 Buy slots
        - State persists cleanly to local JSON
        """
        from utils.alert_state import AlertStateManager, format_curated_alert_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "alert_state.json"
            mgr = AlertStateManager(state_path=state_path)

            results = [
                _make_mock_result("2330.TW", "台積電", score=92, advice="買入", current_price=1050.0, dynamic_stop_loss=980.0, swing_take_profit=1200.0),
                _make_mock_result("2454.TW", "聯發科", score=88, advice="買入", current_price=1300.0, dynamic_stop_loss=1200.0, swing_take_profit=1450.0),
                _make_mock_result("2317.TW", "鴻海", score=85, advice="買入", current_price=210.0, dynamic_stop_loss=195.0, swing_take_profit=235.0),
                _make_mock_result("2382.TW", "廣達", score=83, advice="買入", current_price=300.0, dynamic_stop_loss=275.0, swing_take_profit=340.0),
                _make_mock_result("2618.TW", "長榮航", score=81, advice="買入", current_price=48.0, dynamic_stop_loss=44.0, swing_take_profit=54.0),
                # Score < 82, advice = 持有, but hits stop loss (95.0 <= 98.0) -> MUST trigger SL alert!
                _make_mock_result("0050.TW", "元大台灣50", score=54, advice="持有", current_price=95.0, buy_point_1=94.0, dynamic_stop_loss=98.0, swing_take_profit=115.0),
                # Score < 82, advice = 持有, hits take profit (122.0 >= 120.0) -> MUST trigger TP alert!
                _make_mock_result("VOO", "Vanguard S&P 500 ETF", score=72, advice="持有", current_price=122.0, buy_point_1=110.0, dynamic_stop_loss=105.0, swing_take_profit=120.0),
            ]

            eval_date = date(2026, 11, 17)
            decisions = mgr.evaluate_batch(results, reference_date=eval_date)
            pushed = [d for d in decisions if d.should_push]
            pushed_codes = {d.symbol for d in pushed}

            # Top 3 Buy >= 82 (2330, 2454, 2317) + SL (0050) + TP (VOO)
            self.assertEqual(pushed_codes, {"2330.TW", "2454.TW", "2317.TW", "0050.TW", "VOO"})
            # Rank-4 Buy >= 82 (2382) capped by Top-3 limit; 2618 below 82 threshold
            by_code = {d.symbol: d for d in decisions}
            self.assertFalse(by_code["2382.TW"].should_push)
            self.assertEqual(by_code["2382.TW"].skip_reason_code, "daily_buy_cap")
            self.assertFalse(by_code["2618.TW"].should_push)
            self.assertEqual(by_code["2618.TW"].skip_reason_code, "below_threshold")

            # Card format verification (Traditional Chinese + beginner requirements)
            digest = format_curated_alert_digest(pushed, report_date="2026-11-17")
            for required_text in (
                "台積電",
                "2330.TW",
                "92",
                "建議買入區間",
                "停損價",
                "停利目標",
                "首波試單 30% 資金",
                "觸及停損",
                "觸及停利",
            ):
                self.assertIn(required_text, digest)

            # Record pushed state & daily snapshots to JSON and reload in fresh instance
            mgr.record_pushed_alerts(pushed, reference_date=eval_date)
            mgr.record_daily_snapshots(results, decisions, reference_date=eval_date)
            self.assertTrue(state_path.exists())

            raw_json = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("alerts", raw_json)
            self.assertIn("2330.TW", raw_json["alerts"])
            self.assertEqual(raw_json["alerts"]["2330.TW"]["last_reference_price"], 1050.0)

            mgr_reloaded = AlertStateManager(state_path=state_path)
            self.assertEqual(mgr_reloaded.get_last_alert("2330.TW")["last_advice"], "買入")

    def test_02_five_trading_day_and_three_pct_antispam_state_machine(self):
        """
        Verify:
        - Within 5 trading days, unchanged status + price change <= 3% -> suppressed (daily Markdown only)
        - Within 5 trading days, unchanged status + price change > 3% -> re-pushed!
        - Within 5 trading days, status changed (e.g. Buy -> Take Profit) -> re-pushed even if price diff < 3%!
        - After 5 trading days -> re-pushed even with unchanged status and < 3% price change!
        """
        from utils.alert_state import AlertStateManager

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "alert_state.json"
            mgr = AlertStateManager(state_path=state_path)

            day0 = date(2026, 11, 16)  # Monday
            res_day0 = [_make_mock_result("2330.TW", "台積電", score=86, advice="買入", current_price=1000.0, dynamic_stop_loss=920.0, swing_take_profit=1150.0)]
            d0 = mgr.evaluate_batch(res_day0, reference_date=day0)
            self.assertTrue(d0[0].should_push)
            mgr.record_pushed_alerts(d0, reference_date=day0)

            # Day 2 (2 trading days later): price = 1020 (+2% <= 3%), same Buy state -> SUPPRESSED!
            day2 = date(2026, 11, 18)
            res_day2_small = [_make_mock_result("2330.TW", "台積電", score=87, advice="買入", current_price=1020.0, dynamic_stop_loss=920.0, swing_take_profit=1150.0)]
            d2_small = mgr.evaluate_batch(res_day2_small, reference_date=day2)
            self.assertFalse(d2_small[0].should_push)
            self.assertEqual(d2_small[0].skip_reason_code, "antispam_cooldown")

            # Day 2: price moves +4% (1040 > 3%) -> PUSHED!
            res_day2_big = [_make_mock_result("2330.TW", "台積電", score=88, advice="買入", current_price=1040.0, dynamic_stop_loss=950.0, swing_take_profit=1150.0)]
            d2_big = mgr.evaluate_batch(res_day2_big, reference_date=day2)
            self.assertTrue(d2_big[0].should_push)

            # Day 2: state changes (hits take profit 1015 >= 1010, only +1.5% price move) -> PUSHED!
            res_day2_tp = [_make_mock_result("2330.TW", "台積電", score=86, advice="買入", current_price=1015.0, dynamic_stop_loss=920.0, swing_take_profit=1010.0)]
            d2_tp = mgr.evaluate_batch(res_day2_tp, reference_date=day2)
            self.assertTrue(d2_tp[0].should_push)
            self.assertEqual(d2_tp[0].trigger_type, "take_profit")

            # After 5 trading days (Mon Nov 16 -> Mon Nov 23 = 5 trading days later): +1% -> PUSHED!
            day5_next = date(2026, 11, 23)
            res_day5 = [_make_mock_result("2330.TW", "台積電", score=85, advice="買入", current_price=1010.0, dynamic_stop_loss=920.0, swing_take_profit=1150.0)]
            d5 = mgr.evaluate_batch(res_day5, reference_date=day5_next)
            self.assertTrue(d5[0].should_push)

    def test_03_weekly_friday_discipline_report_generation(self):
        """Verify Friday Weekly Investment Discipline Report aggregates weekly trends & risk alerts in zh-TW."""
        from utils.alert_state import AlertStateManager, generate_weekly_discipline_report

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "alert_state.json"
            mgr = AlertStateManager(state_path=state_path)

            mon = date(2026, 11, 16)
            fri = date(2026, 11, 20)
            mon_res = [
                _make_mock_result("2330.TW", "台積電", score=80, advice="買入", current_price=1000.0, dynamic_stop_loss=940.0, swing_take_profit=1150.0),
                _make_mock_result("0050.TW", "元大台灣50", score=56, advice="持有", current_price=98.0, dynamic_stop_loss=97.0, swing_take_profit=110.0, risk_notes=["尚未完成止跌確認"]),
            ]
            fri_res = [
                _make_mock_result("2330.TW", "台積電", score=86, advice="買入", current_price=1060.0, dynamic_stop_loss=970.0, swing_take_profit=1180.0),
                _make_mock_result("0050.TW", "元大台灣50", score=48, advice="持有", current_price=94.0, dynamic_stop_loss=97.0, swing_take_profit=110.0, risk_notes=["跌破移動停損警戒線"]),
            ]

            dec_mon = mgr.evaluate_batch(mon_res, reference_date=mon)
            mgr.record_daily_snapshots(mon_res, dec_mon, reference_date=mon)
            dec_fri = mgr.evaluate_batch(fri_res, reference_date=fri)
            mgr.record_pushed_alerts([d for d in dec_fri if d.should_push], reference_date=fri)
            mgr.record_daily_snapshots(fri_res, dec_fri, reference_date=fri)

            weekly_md = generate_weekly_discipline_report(
                state_manager=mgr,
                reference_date=fri,
                watchlist_codes=["2330.TW", "0050.TW"],
            )
            for expected in (
                "每週投資紀律週報",
                "本週觀察名單趨勢變化總覽",
                "台積電",
                "2330.TW",
                "+6.00%",
                "元大台灣50",
                "0050.TW",
                "重點風險提醒",
                "8%",
                "30%",
            ):
                self.assertIn(expected, weekly_md)

    def test_04_github_workflows_cleanup_schedules_permissions_and_secrets(self):
        """Verify non-TW/US schedule workflows are deleted and dual-timezone workflows are valid YAML."""
        wf_dir = ROOT_DIR / ".github" / "workflows"

        # Legacy schedule workflows must be removed
        for removed in ("00-daily-analysis.yml", "network-smoke.yml", "stale.yml"):
            self.assertFalse((wf_dir / removed).exists(), f"{removed} should be removed")

        # Validate tw_market_daily.yml (TW Mon-Fri 14:45 = UTC 06:45)
        tw_path = wf_dir / "tw_market_daily.yml"
        self.assertTrue(tw_path.exists(), "tw_market_daily.yml must exist")
        tw_yaml = yaml.safe_load(tw_path.read_text(encoding="utf-8"))
        # Note: PyYAML parses unquoted 'on' as boolean True unless quoted; support either key
        tw_on = tw_yaml.get("on", tw_yaml.get(True))
        self.assertIsInstance(tw_on, dict)
        self.assertEqual(tw_on["schedule"][0]["cron"], "45 6 * * 1-5")
        self.assertEqual(tw_yaml.get("permissions", {}).get("contents"), "write")
        tw_raw = tw_path.read_text(encoding="utf-8")
        self.assertIn("--market tw", tw_raw)
        self.assertIn("--weekly-report", tw_raw)
        self.assertIn("data/*.parquet", tw_raw)
        self.assertIn("alert_state.json", tw_raw)
        self.assertIn("secrets.FINMIND_API_TOKEN", tw_raw)
        self.assertIn("secrets.TELEGRAM_BOT_TOKEN", tw_raw)
        self.assertIn("secrets.DISCORD_WEBHOOK_URL", tw_raw)
        self.assertNotIn("600519", tw_raw)
        self.assertNotIn("akshare", tw_raw.lower())

        # Validate us_etf_daily.yml (TW Tue-Sat 06:30 = UTC 22:30 Mon-Fri)
        us_path = wf_dir / "us_etf_daily.yml"
        self.assertTrue(us_path.exists(), "us_etf_daily.yml must exist")
        us_yaml = yaml.safe_load(us_path.read_text(encoding="utf-8"))
        us_on = us_yaml.get("on", us_yaml.get(True))
        self.assertIsInstance(us_on, dict)
        self.assertEqual(us_on["schedule"][0]["cron"], "30 22 * * 1-5")
        self.assertEqual(us_yaml.get("permissions", {}).get("contents"), "write")
        us_raw = us_path.read_text(encoding="utf-8")
        self.assertIn("--market us", us_raw)
        self.assertIn("--no-tw-screener", us_raw)
        self.assertIn("secrets.TELEGRAM_BOT_TOKEN", us_raw)
        self.assertIn("secrets.DISCORD_WEBHOOK_URL", us_raw)
        self.assertNotIn("600519", us_raw)

        # Verify NO other scheduled workflows exist besides tw_market_daily.yml & us_etf_daily.yml
        for wf_file in wf_dir.glob("*.yml"):
            parsed = yaml.safe_load(wf_file.read_text(encoding="utf-8"))
            self.assertIsInstance(parsed, dict, f"{wf_file.name} must parse to a dict")
            wf_on = parsed.get("on", parsed.get(True)) or {}
            if isinstance(wf_on, dict) and "schedule" in wf_on:
                self.assertIn(
                    wf_file.name,
                    ("tw_market_daily.yml", "us_etf_daily.yml"),
                    f"Unexpected scheduled workflow {wf_file.name}",
                )

    def test_05_main_market_filter_separation(self):
        """Verify main._filter_codes_for_market cleanly separates TW and US schedules."""
        import main

        mixed_codes = ["VOO", "QQQ", "0050.TW", "2330.TW", "6488.TWO", "AAPL"]
        tw_only, tw_targets = main._filter_codes_for_market(mixed_codes, "tw")
        us_only, us_targets = main._filter_codes_for_market(mixed_codes, "us")
        all_codes, _ = main._filter_codes_for_market(mixed_codes, "all")

        self.assertEqual(tw_only, ["0050.TW", "2330.TW", "6488.TWO"])
        self.assertEqual(us_only, ["VOO", "QQQ", "AAPL"])
        self.assertEqual(all_codes, mixed_codes)
        self.assertIsNone(tw_targets)
        self.assertIsNone(us_targets)

    def test_06_pipeline_curated_alert_integration_and_gitignore(self):
        """Verify pipeline curated alert gate + daily Markdown summary + weekly CLI + gitignore rules."""
        import subprocess
        from unittest.mock import MagicMock, patch
        from utils.alert_state import AlertStateManager

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "alert_state.json"
            mgr = AlertStateManager(state_path=state_path)

            res_low = _make_mock_result(
                "2618.TW", "長榮航", score=70, advice="買入", current_price=48.0, dynamic_stop_loss=44.0, swing_take_profit=54.0
            )
            res_high = _make_mock_result(
                "2330.TW", "台積電", score=86, advice="買入", current_price=1050.0, dynamic_stop_loss=980.0, swing_take_profit=1180.0
            )

            # Case 1: Only score=70 (<82) and no SL/TP -> IM push MUST NOT be sent
            decisions_low = mgr.evaluate_batch([res_low], reference_date=date(2026, 11, 17))
            self.assertEqual([d for d in decisions_low if d.should_push], [])

            # Case 2: score=86 (>=82) -> IM push SENT via pipeline helper & persisted to JSON
            decisions_mixed = mgr.evaluate_batch([res_low, res_high], reference_date=date(2026, 11, 17))
            pushed_mixed = [d for d in decisions_mixed if d.should_push]
            self.assertEqual([d.symbol for d in pushed_mixed], ["2330.TW"])

            mock_notifier = MagicMock()
            mock_notifier.is_available.return_value = True
            mock_notifier.send.return_value = True

            from src.core.pipeline import StockAnalysisPipeline

            fake_pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
            fake_pipeline.config = SimpleNamespace(alert_min_buy_score=82)
            fake_pipeline.market_scope = "tw"
            fake_pipeline.notifier = mock_notifier
            fake_pipeline._refresh_saved_diagnostic_snapshot = MagicMock()
            fake_pipeline._build_notification_run_snapshot = lambda **kw: kw

            fake_pipeline._send_curated_alerts(
                results=[res_low, res_high],
                decisions=decisions_mixed,
                alert_manager=mgr,
            )
            mock_notifier.send.assert_called_once()
            sent_text = mock_notifier.send.call_args[0][0]
            self.assertIn("台積電 (2330.TW)", sent_text)
            self.assertNotIn("長榮航 (2618.TW)", sent_text)
            self.assertIsNotNone(mgr.get_last_alert("2330.TW"))

        # Verify gitignore permits data/*.parquet, data/alert_state.json, reports/*.md, while ignoring *.db
        for tracked_path in ("data/tw_daily_history.parquet", "data/alert_state.json", "reports/report_20261117_tw.md"):
            proc = subprocess.run(
                ["git", "check-ignore", tracked_path],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, f"{tracked_path} must NOT be ignored by .gitignore")

        proc_db = subprocess.run(
            ["git", "check-ignore", "data/stock_analysis.db"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc_db.returncode, 0, "data/stock_analysis.db MUST remain ignored")


if __name__ == "__main__":
    unittest.main()
