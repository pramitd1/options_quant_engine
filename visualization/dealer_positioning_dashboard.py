import matplotlib.pyplot as plt

from analytics.gamma_exposure import calculate_gex
from analytics.gamma_flip import gamma_flip_level
from analytics.gamma_walls import classify_walls
from analytics.liquidity_vacuum import detect_liquidity_vacuum


def plot_dealer_dashboard(option_chain, spot, trade):

    """
    Dealer Positioning Dashboard

    Displays:
    - Gamma exposure curve
    - Gamma flip level
    - Support / resistance walls
    - Liquidity vacuum zones
    - Trade analytics
    """

    # ---------------------------------------
    # Compute gamma exposure
    # ---------------------------------------

    gex = calculate_gex(option_chain)

    flip = gamma_flip_level(option_chain)

    walls = classify_walls(option_chain)

    vacuum_zones = detect_liquidity_vacuum(option_chain)

    # ---------------------------------------
    # Plot gamma exposure curve
    # ---------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        gex.index,
        gex.values,
        label="Gamma Exposure"
    )

    # Spot line
    plt.axvline(
        spot,
        linestyle="--",
        label="Spot Price"
    )

    # Gamma flip
    plt.axvline(
        flip,
        linestyle="--",
        label="Gamma Flip"
    )

    # Support wall
    plt.axvline(
        walls["support_wall"],
        linestyle=":",
        label="Support Wall"
    )

    # Resistance wall
    plt.axvline(
        walls["resistance_wall"],
        linestyle=":",
        label="Resistance Wall"
    )

    # Liquidity vacuum zones
    for low, high in vacuum_zones:

        plt.axvspan(
            low,
            high,
            alpha=0.15
        )

    plt.title("Dealer Positioning Dashboard")

    plt.xlabel("Strike")

    plt.ylabel("Gamma Exposure")

    plt.legend()

    plt.grid(True)

    plt.show()

    # ---------------------------------------
    # Print trade analytics
    # ---------------------------------------

    print("\n----- Dealer Flow Summary -----")

    print("Symbol:", trade["symbol"])

    print("Trade Direction:", trade["trade_direction"])

    print("Smart Money Flow:", trade["smart_money_flow"])

    print("Dealer Hedging Bias:", trade["dealer_hedging_bias"])

    print("Gamma Regime:", trade["gamma_regime"])

    print("Volatility Regime:", trade["volatility_regime"])

    print("Move Probability:", trade["move_probability"])

    print("ML Move Probability:", trade["ml_move_probability"])

    print("Trade Strength:", trade["trade_strength"])