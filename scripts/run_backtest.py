#!/usr/bin/env python
"""Run a backtest from the command line.

Example:
    python scripts/run_backtest.py --start 2025-01-01 --end 2025-12-31 \
        --symbols RELIANCE,INFY,HDFCBANK --capital 500000 --risk 0.25

The results are printed as a report and persisted, so they also appear on the
Backtest page of the dashboard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chief Agent backtest")
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="comma separated symbols")
    parser.add_argument("--capital", type=float, default=500_000.0)
    parser.add_argument("--risk", type=float, default=0.25, help="risk per trade in PERCENT (default 0.25)")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--max-instruments", type=int, default=40)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--slippage-multiplier", type=float, default=1.0)
    parser.add_argument("--fee-multiplier", type=float, default=1.0)
    parser.add_argument("--monte-carlo", action="store_true", default=True)
    parser.add_argument("--json", action="store_true", help="print the raw JSON payload")
    args = parser.parse_args()

    from chief_agent.backtest.montecarlo import run_monte_carlo
    from chief_agent.backtest.runner import BacktestRequest, BacktestRunner
    from chief_agent.data.db import init_db, session_scope
    from chief_agent.data.provider import MarketDataProvider
    from chief_agent.data.synthetic import generate_index_history
    from chief_agent.logging_setup import configure_logging

    configure_logging(level="WARNING")
    init_db()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    provider = MarketDataProvider()
    print(f"Data source: {provider.data_source}"
          + ("  (SIMULATED — results are not evidence of an edge)" if provider.is_simulated else ""))

    with session_scope() as session:
        runner = BacktestRunner(session, provider, max_instruments=args.max_instruments)
        request = BacktestRequest(
            start_date=start,
            end_date=end,
            universe=symbols,
            initial_capital=args.capital,
            risk_per_trade_pct=min(0.005, args.risk / 100.0),
            timeframe=args.timeframe,
            include_shorts=not args.long_only,
            slippage_multiplier=args.slippage_multiplier,
            fee_multiplier=args.fee_multiplier,
            max_universe=args.max_instruments,
        )
        result = runner.run(request)
        session.commit()

    metrics = result.metrics.to_dict()
    if args.json:
        print(json.dumps(result.to_dict(include_trades=False), indent=2, default=str))
        return 0

    def line(label: str, value: object) -> None:
        print(f"  {label:<32} {value}")

    print()
    print("=" * 62)
    print(f"  BACKTEST  {start} to {end}   ({len(result.trades)} trades)")
    print("=" * 62)
    line("Net return", f"{metrics['net_return_pct']:+.2%}")
    line("Annualised return", f"{metrics['annualised_return_pct']:+.2%}")
    line("Volatility", f"{metrics['volatility_pct']:.2%}")
    line("Sharpe / Sortino / Calmar", f"{metrics['sharpe']:.2f} / {metrics['sortino']:.2f} / {metrics['calmar']:.2f}")
    line("Maximum drawdown", f"{metrics['max_drawdown_pct']:+.2%}")
    line("Win rate", f"{metrics['win_rate']:.2%}")
    line("Average win / loss", f"{metrics['average_win']:,.0f} / {metrics['average_loss']:,.0f}")
    line("Expectancy", f"{metrics['expectancy_r']:+.4f} R  ({metrics['expectancy']:,.0f} per trade)")
    line("Profit factor", f"{metrics['profit_factor']:.2f}")
    line("Fees + slippage", f"{metrics['fees']:,.0f} + {metrics['slippage']:,.0f}")
    line("Average holding time", f"{metrics['average_holding_minutes']:.0f} minutes")
    line("Max consecutive losses", metrics["max_consecutive_losses"])
    line("Objective score", f"{metrics['objective_score']:.4f}")

    print("\n  By regime")
    for name, row in result.metrics.regime_performance.items():
        print(f"    {name:<18} {row['trades']:>4} trades   {row['net_pnl']:>10,.0f}   expectancy {row['expectancy_r']:+.3f}R")

    print("\n  By exit reason")
    for name, row in result.metrics.exit_reason_performance.items():
        print(f"    {name:<18} {row['trades']:>4} trades   avgR {row['average_r']:+.3f}")

    if args.monte_carlo:
        mc = run_monte_carlo(
            [t.net_pnl for t in result.trades], initial_capital=args.capital, n_simulations=1000
        ).to_dict()
        print("\n  Monte Carlo (1000 bootstrapped sequences)")
        print(f"    return        5th {mc['return_pct']['p5']:+.2%}   median {mc['return_pct']['median']:+.2%}   95th {mc['return_pct']['p95']:+.2%}")
        print(f"    max drawdown  median {mc['max_drawdown_pct']['median']:+.2%}   95th {mc['max_drawdown_pct']['p95']:+.2%}")
        print(f"    ruin probability (a 30% drawdown)  {mc['ruin_probability']:.2%}")

    print("\n  Costs are always included: brokerage, STT, exchange charges, SEBI fee, GST,")
    print("  stamp duty, plus modelled slippage and spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
