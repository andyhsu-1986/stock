# -*- coding: utf-8 -*-
"""Unit tests for Taiwan full-market quantitative screener and watchlist integration."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _make_synthetic_history(
    trading_days: int = 25,
    end_date: date = date(2026, 11, 16),
) -> pd.DataFrame:
    """Build synthetic multi-day TWSE + TPEx market history for deterministic scoring tests."""
    dates = [
        (end_date - timedelta(days=trading_days - 1 - i)).strftime("%Y-%m-%d")
        for i in range(trading_days)
    ]
    rows: List[Dict[str, object]] = []

    # 1. Strong uptrend stock (TWSE): 2330.TW (MA5 > MA10 > MA20, good volume, 5-day inst buy, good PE/Yield)
    for idx, d in enumerate(dates):
        close = 900.0 + idx * 3.0
        rows.append(
            {
                "date": d,
                "code": "2330",
                "symbol": "2330.TW",
                "name": "台積電",
                "market": "TWSE",
                "is_etf": False,
                "open": close - 2.0,
                "high": close + 4.0,
                "low": close - 3.0,
                "close": close,
                "volume": 25_000_000 if idx < trading_days - 1 else 38_000_000,
                "turnover_value": close * 38_000_000,
                "foreign_net": 5_000_000 if idx >= trading_days - 6 else 1_000_000,
                "trust_net": 1_500_000 if idx >= trading_days - 6 else 200_000,
                "dealer_net": 300_000,
                "inst_total_net": 6_800_000 if idx >= trading_days - 6 else 1_500_000,
                "pe_ratio": 18.5,
                "dividend_yield": 3.5,
                "pb_ratio": 4.2,
            }
        )

    # 2. Strong uptrend OTC stock (TPEx): 6488.TWO
    for idx, d in enumerate(dates):
        close = 400.0 + idx * 1.5
        rows.append(
            {
                "date": d,
                "code": "6488",
                "symbol": "6488.TWO",
                "name": "環球晶",
                "market": "TPEX",
                "is_etf": False,
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 3_000_000 if idx < trading_days - 1 else 4_500_000,
                "turnover_value": close * 4_500_000,
                "foreign_net": 800_000 if idx >= trading_days - 4 else -100_000,
                "trust_net": 400_000 if idx >= trading_days - 4 else 100_000,
                "dealer_net": 50_000,
                "inst_total_net": 1_250_000 if idx >= trading_days - 4 else 50_000,
                "pe_ratio": 15.2,
                "dividend_yield": 4.1,
                "pb_ratio": 2.8,
            }
        )

    # 3. Strong uptrend Taiwan ETF: 0050.TW (PE is 0.0, scored via ETF fundamental rule)
    for idx, d in enumerate(dates):
        close = 160.0 + idx * 0.5
        rows.append(
            {
                "date": d,
                "code": "0050",
                "symbol": "0050.TW",
                "name": "元大台灣50",
                "market": "TWSE",
                "is_etf": True,
                "open": close - 0.3,
                "high": close + 0.6,
                "low": close - 0.4,
                "close": close,
                "volume": 10_000_000 if idx < trading_days - 1 else 16_000_000,
                "turnover_value": close * 16_000_000,
                "foreign_net": 2_000_000 if idx >= trading_days - 5 else 500_000,
                "trust_net": 500_000,
                "dealer_net": 100_000,
                "inst_total_net": 2_600_000 if idx >= trading_days - 5 else 1_100_000,
                "pe_ratio": 0.0,
                "dividend_yield": 3.8,
                "pb_ratio": 0.0,
            }
        )

    # 4. Downtrend stock (MA5 < MA10 < MA20): should be filtered out
    for idx, d in enumerate(dates):
        close = 200.0 - idx * 2.0
        rows.append(
            {
                "date": d,
                "code": "9999",
                "symbol": "9999.TW",
                "name": "空頭股",
                "market": "TWSE",
                "is_etf": False,
                "open": close + 1.0,
                "high": close + 2.0,
                "low": close - 1.0,
                "close": close,
                "volume": 5_000_000,
                "turnover_value": close * 5_000_000,
                "foreign_net": -1_000_000,
                "trust_net": -500_000,
                "dealer_net": 0,
                "inst_total_net": -1_500_000,
                "pe_ratio": 12.0,
                "dividend_yield": 5.0,
                "pb_ratio": 1.2,
            }
        )

    # 5. Low liquidity stock (volume = 800,000 shares = 800 lots < 1,000 lots): should be filtered out
    for idx, d in enumerate(dates):
        close = 50.0 + idx * 0.2
        rows.append(
            {
                "date": d,
                "code": "8888",
                "symbol": "8888.TWO",
                "name": "量縮股",
                "market": "TPEX",
                "is_etf": False,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 800_000,
                "turnover_value": close * 800_000,
                "foreign_net": 100_000,
                "trust_net": 50_000,
                "dealer_net": 0,
                "inst_total_net": 150_000,
                "pe_ratio": 10.0,
                "dividend_yield": 4.0,
                "pb_ratio": 1.0,
            }
        )

    # 6. Penny stock (close < 10 TWD despite high volume & uptrend): should be filtered out
    for idx, d in enumerate(dates):
        close = 7.0 + idx * 0.08  # ends at 8.92 < 10.0
        rows.append(
            {
                "date": d,
                "code": "7777",
                "symbol": "7777.TW",
                "name": "雞蛋水餃股",
                "market": "TWSE",
                "is_etf": False,
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 5_000_000,
                "turnover_value": close * 5_000_000,
                "foreign_net": 500_000,
                "trust_net": 200_000,
                "dealer_net": 0,
                "inst_total_net": 700_000,
                "pe_ratio": 12.0,
                "dividend_yield": 3.0,
                "pb_ratio": 0.9,
            }
        )

    # 7. Disposition stock (處置股, high volume & uptrend): should be filtered out
    for idx, d in enumerate(dates):
        close = 120.0 + idx * 1.0
        rows.append(
            {
                "date": d,
                "code": "2455",
                "symbol": "2455.TW",
                "name": "處置股範例",
                "market": "TWSE",
                "is_etf": False,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 6_000_000,
                "turnover_value": close * 6_000_000,
                "foreign_net": 600_000,
                "trust_net": 300_000,
                "dealer_net": 0,
                "inst_total_net": 900_000,
                "pe_ratio": 16.0,
                "dividend_yield": 3.2,
                "pb_ratio": 2.1,
                "is_disposition": True,
            }
        )

    # 8. Full-cash delivery stock (全額交割股, high volume & uptrend): should be filtered out
    for idx, d in enumerate(dates):
        close = 35.0 + idx * 0.3
        rows.append(
            {
                "date": d,
                "code": "1213",
                "symbol": "1213.TW",
                "name": "全額交割範例",
                "market": "TWSE",
                "is_etf": False,
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.3,
                "close": close,
                "volume": 4_000_000,
                "turnover_value": close * 4_000_000,
                "foreign_net": 400_000,
                "trust_net": 100_000,
                "dealer_net": 0,
                "inst_total_net": 500_000,
                "pe_ratio": 14.0,
                "dividend_yield": 2.5,
                "pb_ratio": 1.5,
                "is_full_cash_delivery": True,
            }
        )

    return pd.DataFrame(rows)


class TWMarketScreenerTestCase(unittest.TestCase):
    def test_code_and_symbol_filtering(self):
        from screener.tw_market_screener import is_eligible_tw_code, to_yahoo_tw_symbol

        self.assertTrue(is_eligible_tw_code("2330"))
        self.assertTrue(is_eligible_tw_code("6488"))
        self.assertTrue(is_eligible_tw_code("0050"))
        self.assertTrue(is_eligible_tw_code("00878"))
        self.assertTrue(is_eligible_tw_code("006208"))

        self.assertFalse(is_eligible_tw_code("00632R"))
        self.assertFalse(is_eligible_tw_code("00679B"))
        self.assertFalse(is_eligible_tw_code("030001"))
        self.assertFalse(is_eligible_tw_code("2881A"))

        self.assertEqual(to_yahoo_tw_symbol("2330", "TWSE"), "2330.TW")
        self.assertEqual(to_yahoo_tw_symbol("6488", "TPEX"), "6488.TWO")

    def test_parquet_five_year_retention_and_dedup(self):
        from screener.tw_market_screener import TWMarketScreener

        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "tw_daily_history.parquet"
            screener = TWMarketScreener(parquet_path=parquet_path)

            old_date = (date.today() - timedelta(days=365 * 5 + 30)).strftime("%Y-%m-%d")
            recent_date = date.today().strftime("%Y-%m-%d")

            df_initial = pd.DataFrame(
                [
                    {
                        "date": old_date,
                        "code": "2330",
                        "symbol": "2330.TW",
                        "name": "台積電",
                        "market": "TWSE",
                        "is_etf": False,
                        "open": 500.0,
                        "high": 505.0,
                        "low": 498.0,
                        "close": 502.0,
                        "volume": 20_000_000,
                        "turnover_value": 10_000_000_000.0,
                        "foreign_net": 100_000,
                        "trust_net": 50_000,
                        "dealer_net": 0,
                        "inst_total_net": 150_000,
                        "pe_ratio": 20.0,
                        "dividend_yield": 2.5,
                        "pb_ratio": 5.0,
                    },
                    {
                        "date": recent_date,
                        "code": "2330",
                        "symbol": "2330.TW",
                        "name": "台積電",
                        "market": "TWSE",
                        "is_etf": False,
                        "open": 1000.0,
                        "high": 1010.0,
                        "low": 995.0,
                        "close": 1005.0,
                        "volume": 30_000_000,
                        "turnover_value": 30_000_000_000.0,
                        "foreign_net": 2_000_000,
                        "trust_net": 500_000,
                        "dealer_net": 100_000,
                        "inst_total_net": 2_600_000,
                        "pe_ratio": 19.0,
                        "dividend_yield": 3.0,
                        "pb_ratio": 4.5,
                    },
                ]
            )

            screener._save_history(df_initial)
            loaded = screener._load_history()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0]["date"], recent_date)
            self.assertEqual(loaded.iloc[0]["close"], 1005.0)

            df_update = df_initial.iloc[[1]].copy()
            df_update["close"] = 1015.0
            combined = screener._merge_and_prune_history(loaded, df_update)
            self.assertEqual(len(combined), 1)
            self.assertEqual(combined.iloc[0]["close"], 1015.0)

    def test_quantitative_scoring_and_top_n_selection(self):
        from screener.tw_market_screener import TWMarketScreener

        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "tw_daily_history.parquet"
            screener = TWMarketScreener(parquet_path=parquet_path, top_n=20)
            history_df = _make_synthetic_history(trading_days=25)

            ranked_df = screener.score_universe(history_df)
            symbols = ranked_df["symbol"].tolist()

            self.assertIn("2330.TW", symbols)
            self.assertIn("6488.TWO", symbols)
            self.assertIn("0050.TW", symbols)
            self.assertNotIn("9999.TW", symbols)
            self.assertNotIn("8888.TWO", symbols, "5-day avg volume < 1,000 lots (1,000,000 shares) must be excluded")
            self.assertNotIn("7777.TW", symbols, "Close price < 10 TWD must be excluded")
            self.assertNotIn("2455.TW", symbols, "Disposition stock (處置股) must be excluded")
            self.assertNotIn("1213.TW", symbols, "Full-cash delivery stock (全額交割股) must be excluded")

            # Also verify restricted_codes set on screener excludes matching code even without df flag
            screener_restricted = TWMarketScreener(
                parquet_path=parquet_path, top_n=20, restricted_codes={"6488"}
            )
            ranked_restricted = screener_restricted.score_universe(history_df)
            self.assertNotIn("6488.TWO", ranked_restricted["symbol"].tolist())

            for col in ("trend_score", "volume_score", "inst_score", "fundamental_score", "total_score"):
                self.assertIn(col, ranked_df.columns)
            self.assertTrue(ranked_df["total_score"].is_monotonic_decreasing)

    def test_watchlist_merge_preserves_fixed_targets_and_deduplicates(self):
        from screener.tw_market_screener import merge_watchlist_with_screened

        fixed_watchlist = ["VOO", "QQQ", "0050.TW", "2330.TW"]
        screened_top = ["2330.TW", "6488.TWO", "0050.TW", "2454.TW", "2317.TW"]

        merged = merge_watchlist_with_screened(fixed_watchlist, screened_top)
        self.assertEqual(
            merged,
            ["VOO", "QQQ", "0050.TW", "2330.TW", "6488.TWO", "2454.TW", "2317.TW"],
        )

    def test_main_merge_with_tw_screener_targets_and_fail_open(self):
        import argparse
        from unittest.mock import patch
        import main
        from src.config import get_config
        from src.services.stock_list_parser import parse_analysis_target

        config = get_config()
        config.tw_screener_enabled = True
        config.tw_screener_top_n = 3
        args = argparse.Namespace(no_tw_screener=False)

        base_codes = ["VOO", "0050.TW"]
        base_targets = [parse_analysis_target(c) for c in base_codes]

        with patch(
            "screener.tw_market_screener.get_top_tw_screened_stocks",
            return_value=["2330.TW", "0050.TW", "6488.TWO"],
        ):
            merged_codes, merged_targets = main._merge_with_tw_screener_targets(
                config, args, base_codes, analysis_targets=base_targets
            )
            self.assertEqual(merged_codes, ["VOO", "0050.TW", "2330.TW", "6488.TWO"])
            self.assertIsNotNone(merged_targets)
            self.assertEqual(len(merged_targets), 4)
            self.assertEqual(merged_targets[2].canonical_id, "2330.TW")
            self.assertEqual(merged_targets[3].canonical_id, "6488.TWO")

        # Verify fail-open when screener raises an unexpected error
        with patch(
            "screener.tw_market_screener.get_top_tw_screened_stocks",
            side_effect=RuntimeError("network down"),
        ):
            fallback_codes, fallback_targets = main._merge_with_tw_screener_targets(
                config, args, base_codes, analysis_targets=base_targets
            )
            self.assertEqual(fallback_codes, base_codes)
            self.assertEqual(fallback_targets, base_targets)


if __name__ == "__main__":
    unittest.main()
