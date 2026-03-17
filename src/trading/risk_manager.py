from config import MAX_POSITION_PER_CONTRACT


def approve_trade(signal):

    if signal["side"] == "HOLD":
        return False

    return True