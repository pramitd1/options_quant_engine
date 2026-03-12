def select_strike(spot, step=50):

    atm = round(spot/step)*step

    return atm