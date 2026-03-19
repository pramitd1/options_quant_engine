import json
from pathlib import Path


def main() -> None:
    src = Path("backtest_comparison_results_20260319_220319.json")
    out = Path("research/runtime_validation/backtest_comparison_summary_20260319_220319.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text())

    rows = []
    for m in data.get("comparison_metrics", []):
        if m.get("status") != "SUCCESS":
            continue
        rows.append(
            {
                "method": m.get("method"),
                "total_signals": m.get("total_signals"),
                "trade_signals": m.get("trade_signals"),
                "trade_rate_pct": round(float(m.get("trade_rate", 0.0)) * 100, 2),
                "avg_trade_strength": round(float(m.get("avg_trade_strength", 0.0)), 2),
                "avg_composite_score": round(float(m.get("avg_composite_score", 0.0)), 2),
                "expiry_accuracy_pct": round(float(m.get("correct_expiry", 0.0)) * 100, 2),
                "target_hit_rate_pct": round(float(m.get("target_hit_rate", 0.0)) * 100, 2),
                "stop_loss_hit_rate_pct": round(float(m.get("stop_loss_hit_rate", 0.0)) * 100, 2),
                "elapsed_seconds": round(float(m.get("elapsed_seconds", 0.0)), 2),
            }
        )

    leaders = {
        "best_composite": max(rows, key=lambda r: r["avg_composite_score"]) if rows else None,
        "best_strength": max(rows, key=lambda r: r["avg_trade_strength"]) if rows else None,
        "best_expiry_accuracy": max(rows, key=lambda r: r["expiry_accuracy_pct"]) if rows else None,
        "best_target_hit_rate": max(rows, key=lambda r: r["target_hit_rate_pct"]) if rows else None,
        "lowest_stop_loss_hit_rate": min(rows, key=lambda r: r["stop_loss_hit_rate_pct"]) if rows else None,
    }

    summary = {
        "source_file": str(src),
        "method_rows": rows,
        "leaders": leaders,
    }
    out.write_text(json.dumps(summary, indent=2))

    print(f"summary_artifact={out}")
    print(f"methods={len(rows)}")
    for r in rows:
        print(
            f"{r['method']}: signals={r['total_signals']}, trades={r['trade_signals']}, "
            f"trade_rate={r['trade_rate_pct']}%, strength={r['avg_trade_strength']}, "
            f"composite={r['avg_composite_score']}, expiry_acc={r['expiry_accuracy_pct']}%, "
            f"TP={r['target_hit_rate_pct']}%, SL={r['stop_loss_hit_rate_pct']}%, "
            f"elapsed={r['elapsed_seconds']}s"
        )
    print(f"leader_best_composite={leaders['best_composite']['method'] if leaders['best_composite'] else None}")


if __name__ == "__main__":
    main()
