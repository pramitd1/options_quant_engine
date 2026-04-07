import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB = "research/signal_evaluation/signals_dataset_cumul.sqlite"


def main() -> None:
    out_dir = Path("research/reviews/watchlist_realized_evaluation") / f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        """
        SELECT substr(signal_timestamp,1,10) day,
               SUM(CASE WHEN trade_status='WATCHLIST' AND signed_return_60m_bps IS NOT NULL THEN 1 ELSE 0 END) wl_60m,
               SUM(CASE WHEN trade_status='TRADE' AND signed_return_60m_bps IS NOT NULL THEN 1 ELSE 0 END) tr_60m
        FROM signals
        GROUP BY day
        HAVING wl_60m > 0 OR tr_60m > 0
        ORDER BY day DESC
        """
    )
    days = [dict(r) for r in c.fetchall()]
    valid_days = [r["day"] for r in days]
    if not valid_days:
        raise RuntimeError("No days with populated realized outcomes found")

    horizons = [
        ("5m", "signed_return_5m_bps", "correct_5m"),
        ("15m", "signed_return_15m_bps", "correct_15m"),
        ("30m", "signed_return_30m_bps", "correct_30m"),
        ("60m", "signed_return_60m_bps", "correct_60m"),
        ("120m", "signed_return_120m_bps", "correct_120m"),
        ("session", "signed_return_session_close_bps", "correct_session_close"),
    ]

    def fetch_metrics(cohort_name: str, status_filter: str, extra_where: str = "1=1"):
        rows = []
        for day in valid_days:
            for hz, ret_col, corr_col in horizons:
                query = f"""
                      SELECT '{day}' AS day,
                          '{cohort_name}' AS cohort,
                           '{hz}' AS horizon,
                           COUNT(*) AS n,
                           AVG({ret_col}) AS avg_bps,
                           MIN({ret_col}) AS min_bps,
                           MAX({ret_col}) AS max_bps,
                           AVG(CASE WHEN {ret_col} > 0 THEN 1.0 ELSE 0.0 END) AS pos_rate,
                           AVG(CASE WHEN {corr_col} = 1 THEN 1.0 ELSE 0.0 END) AS correct_rate
                    FROM signals
                    WHERE substr(signal_timestamp,1,10)=?
                      AND trade_status {status_filter}
                      AND {ret_col} IS NOT NULL
                      AND {extra_where}
                """
                c.execute(query, (day,))
                row = dict(c.fetchone())
                if row["n"] and row["n"] > 0:
                    rows.append(row)
        return rows

    watch_all = fetch_metrics("WATCHLIST", "='WATCHLIST'")
    trade_all = fetch_metrics("TRADE", "='TRADE'")
    watch_strong = fetch_metrics(
        "WATCHLIST_STRONG",
        "='WATCHLIST'",
        "confirmation_status IN ('STRONG_CONFIRMATION','CONFIRMED') AND trade_strength>=74",
    )
    watch_caution = fetch_metrics(
        "WATCHLIST_CAUTION",
        "='WATCHLIST'",
        "provider_health_status IN ('WEAK','CAUTION') AND data_quality_status='CAUTION'",
    )

    c.execute(
        """
        SELECT substr(signal_timestamp,1,10) day,
               COUNT(*) watch_n,
               AVG(signed_return_60m_bps) avg_bps,
               AVG(CASE WHEN signed_return_60m_bps > 0 THEN 1.0 ELSE 0 END) pos_rate,
               AVG(CASE WHEN correct_60m = 1 THEN 1.0 ELSE 0 END) correct_rate,
               SUM(CASE WHEN signed_return_60m_bps >= 25 THEN 1 ELSE 0 END) gt_25bps,
               SUM(CASE WHEN signed_return_60m_bps >= 50 THEN 1 ELSE 0 END) gt_50bps,
               SUM(CASE WHEN signed_return_60m_bps >= 100 THEN 1 ELSE 0 END) gt_100bps,
               SUM(CASE WHEN confirmation_status IN ('STRONG_CONFIRMATION','CONFIRMED')
                         AND trade_strength>=74 AND signed_return_60m_bps >= 50
                        THEN 1 ELSE 0 END) strong_gt_50bps
        FROM signals
        WHERE trade_status='WATCHLIST'
          AND signed_return_60m_bps IS NOT NULL
        GROUP BY day
        ORDER BY day DESC
        """
    )
    missed_by_day = [dict(r) for r in c.fetchall()]

    c.execute(
        """
        SELECT substr(signal_timestamp,1,10) day,
               global_risk_state,
               COUNT(*) n,
               AVG(signed_return_60m_bps) avg_bps,
               AVG(CASE WHEN signed_return_60m_bps > 0 THEN 1.0 ELSE 0 END) pos_rate,
               AVG(CASE WHEN correct_60m = 1 THEN 1.0 ELSE 0 END) correct_rate
        FROM signals
        WHERE trade_status='WATCHLIST'
          AND signed_return_60m_bps IS NOT NULL
        GROUP BY day, global_risk_state
        ORDER BY day DESC, n DESC
        """
    )
    risk_slice = [dict(r) for r in c.fetchall()]

    c.execute(
        """
        SELECT 'WATCHLIST_ALL' cohort,
               COUNT(*) n,
               AVG(signed_return_60m_bps) avg_bps,
               AVG(CASE WHEN signed_return_60m_bps > 0 THEN 1.0 ELSE 0 END) pos_rate,
               AVG(CASE WHEN correct_60m = 1 THEN 1.0 ELSE 0 END) correct_rate
        FROM signals
        WHERE trade_status='WATCHLIST' AND signed_return_60m_bps IS NOT NULL
        UNION ALL
        SELECT 'WATCHLIST_STRONG',
               COUNT(*),
               AVG(signed_return_60m_bps),
               AVG(CASE WHEN signed_return_60m_bps > 0 THEN 1.0 ELSE 0 END),
               AVG(CASE WHEN correct_60m = 1 THEN 1.0 ELSE 0 END)
        FROM signals
        WHERE trade_status='WATCHLIST'
          AND signed_return_60m_bps IS NOT NULL
          AND confirmation_status IN ('STRONG_CONFIRMATION','CONFIRMED')
          AND trade_strength>=74
        UNION ALL
        SELECT 'TRADE',
               COUNT(*),
               AVG(signed_return_60m_bps),
               AVG(CASE WHEN signed_return_60m_bps > 0 THEN 1.0 ELSE 0 END),
               AVG(CASE WHEN correct_60m = 1 THEN 1.0 ELSE 0 END)
        FROM signals
        WHERE trade_status='TRADE' AND signed_return_60m_bps IS NOT NULL
        """
    )
    agg60 = [dict(r) for r in c.fetchall()]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "data_note": "WATCHLIST rows may be partially populated for the latest date until next-session realized outcomes are available.",
        "days_with_realized_outcomes": days,
        "aggregate_60m_cohort_comparison": agg60,
        "watchlist_missed_opportunity_by_day_60m": missed_by_day,
        "watchlist_risk_state_slice_60m": risk_slice,
    }

    (out_dir / "watchlist_realized_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, rows in [
        ("watchlist_all_horizon_metrics.csv", watch_all),
        ("trade_all_horizon_metrics.csv", trade_all),
        ("watchlist_strong_horizon_metrics.csv", watch_strong),
        ("watchlist_caution_horizon_metrics.csv", watch_caution),
        ("watchlist_missed_opportunity_60m_by_day.csv", missed_by_day),
        ("watchlist_risk_state_60m.csv", risk_slice),
        ("cohort_aggregate_60m.csv", agg60),
    ]:
        if rows:
            with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    conn.close()
    print(str(out_dir))
    print(json.dumps(agg60, indent=2))


if __name__ == "__main__":
    main()
