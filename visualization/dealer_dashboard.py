"""
Dealer Positioning Dashboard

Displays structural analytics from the options market.
"""

import json


def _format_value(value, max_items=8):
    if isinstance(value, float):
        return round(value, 2)

    if isinstance(value, list):
        if len(value) <= max_items:
            return value
        preview = value[:max_items]
        return f"{preview} ... (+{len(value) - max_items} more)"

    if isinstance(value, dict):
        text = json.dumps(value, default=str)
        if len(text) <= 180:
            return value
        return f"{text[:180]}..."

    return value


def print_dealer_dashboard(summary: dict):
    print("\nDEALER POSITIONING DASHBOARD")
    print("--------------------------------------------------")
    ordered_keys = [
        ("Spot Price", "spot"),
        ("Gamma Exposure", "gamma_exposure"),
        ("Market Gamma", "market_gamma"),
        ("Delta Exposure", "delta_exposure"),
        ("Greek Gamma Exp", "gamma_exposure_greeks"),
        ("Theta Exposure", "theta_exposure"),
        ("Vega Exposure", "vega_exposure"),
        ("Rho Exposure", "rho_exposure"),
        ("Vanna Exposure", "vanna_exposure"),
        ("Charm Exposure", "charm_exposure"),
        ("Gamma Flip Level", "gamma_flip"),
        ("Spot vs Flip", "spot_vs_flip"),
        ("Gamma Regime", "gamma_regime"),
        ("Vanna Regime", "vanna_regime"),
        ("Charm Regime", "charm_regime"),
        ("Gamma Clusters", "gamma_clusters"),
        ("Dealer Inventory", "dealer_position"),
        ("Dealer Inv Basis", "dealer_inventory_basis"),
        ("Call OI Change", "call_oi_change"),
        ("Put OI Change", "put_oi_change"),
        ("Net OI Bias", "net_oi_change_bias"),
        ("Dealer Hedging Flow", "dealer_hedging_flow"),
        ("Dealer Hedging Bias", "dealer_hedging_bias"),
        ("Intraday Gamma State", "intraday_gamma_state"),
        ("Volatility Regime", "volatility_regime"),
        ("Vol Surface Regime", "vol_surface_regime"),
        ("ATM IV", "atm_iv"),
        ("Flow Signal", "flow_signal"),
        ("Smart Money Flow", "smart_money_flow"),
        ("Final Flow Signal", "final_flow_signal"),
        ("Gamma Event", "gamma_event"),
        ("Support Wall", "support_wall"),
        ("Resistance Wall", "resistance_wall"),
        ("Liquidity Levels", "liquidity_levels"),
        ("Liquidity Voids", "liquidity_voids"),
        ("Liquidity Void Signal", "liquidity_void_signal"),
        ("Liquidity Vacuum Zones", "liquidity_vacuum_zones"),
        ("Liquidity Vacuum State", "liquidity_vacuum_state"),
    ]

    for label, key in ordered_keys:
        print(f"{label:22}: {_format_value(summary.get(key))}")

    dealer_map = summary.get("dealer_liquidity_map")
    if dealer_map:
        print(f"{'Dealer Liquidity Map':22}: {_format_value(dealer_map)}")

    print(f"{'Large Move Probability':22}: {_format_value(summary.get('large_move_probability'))}")
    print(f"{'ML Move Probability':22}: {_format_value(summary.get('ml_move_probability'))}")
    print("--------------------------------------------------")

    if "scoring_breakdown" in summary and summary["scoring_breakdown"] is not None:
        print("SCORING BREAKDOWN")
        print("--------------------------------------------------")
        for key, value in summary["scoring_breakdown"].items():
            print(f"{key:22}: {_format_value(value)}")
        print("--------------------------------------------------")
