from config.settings import TARGET_PROFIT_PERCENT, STOP_LOSS_PERCENT


def calculate_exit(entry_price):

    target = entry_price * (1 + TARGET_PROFIT_PERCENT / 100)

    stop = entry_price * (1 - STOP_LOSS_PERCENT / 100)

    return target, stop