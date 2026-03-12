"""
Dealer Positioning Dashboard

Displays structural analytics from the options market.
"""


def print_dealer_dashboard(summary: dict):
    print("\nDEALER POSITIONING DASHBOARD")
    print("--------------------------------------------------")
    print("Spot Price:", summary.get("spot"))
    print("Gamma Exposure:", summary.get("gamma_exposure"))
    print("Market Gamma:", summary.get("market_gamma"))
    print("Gamma Flip Level:", summary.get("gamma_flip"))
    print("Spot vs Flip:", summary.get("spot_vs_flip"))
    print("Gamma Regime:", summary.get("gamma_regime"))
    print("Gamma Clusters:", summary.get("gamma_clusters"))
    print("Dealer Inventory:", summary.get("dealer_position"))
    print("Dealer Hedging Flow:", summary.get("dealer_hedging_flow"))
    print("Dealer Hedging Bias:", summary.get("dealer_hedging_bias"))
    print("Intraday Gamma State:", summary.get("intraday_gamma_state"))
    print("Volatility Regime:", summary.get("volatility_regime"))
    print("Vol Surface Regime:", summary.get("vol_surface_regime"))
    print("ATM IV:", summary.get("atm_iv"))
    print("Flow Signal:", summary.get("flow_signal"))
    print("Smart Money Flow:", summary.get("smart_money_flow"))
    print("Final Flow Signal:", summary.get("final_flow_signal"))
    print("Gamma Event:", summary.get("gamma_event"))
    print("Support Wall:", summary.get("support_wall"))
    print("Resistance Wall:", summary.get("resistance_wall"))
    print("Liquidity Levels:", summary.get("liquidity_levels"))
    print("Liquidity Voids:", summary.get("liquidity_voids"))
    print("Liquidity Void Signal:", summary.get("liquidity_void_signal"))
    print("Liquidity Vacuum Zones:", summary.get("liquidity_vacuum_zones"))
    print("Liquidity Vacuum State:", summary.get("liquidity_vacuum_state"))

    dealer_map = summary.get("dealer_liquidity_map")
    if dealer_map:
        print("Dealer Liquidity Map:", dealer_map)

    print("Large Move Probability:", summary.get("large_move_probability"))
    print("ML Move Probability:", summary.get("ml_move_probability"))
    print("--------------------------------------------------")

    if "scoring_breakdown" in summary and summary["scoring_breakdown"] is not None:
        print("SCORING BREAKDOWN")
        print("--------------------------------------------------")
        for key, value in summary["scoring_breakdown"].items():
            print(f"{key}: {value}")
        print("--------------------------------------------------")