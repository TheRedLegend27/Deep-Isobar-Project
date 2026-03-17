from config import ALPHA_THRESHOLD


def compute_alpha(model_probability, market_probability):

    return model_probability - market_probability


def generate_signal(contract, model_probability, market_probability):

    alpha = compute_alpha(model_probability, market_probability)

    if alpha > ALPHA_THRESHOLD:
        side = "BUY"

    elif alpha < -ALPHA_THRESHOLD:
        side = "SELL"

    else:
        side = "HOLD"

    return {
        "contract": contract,
        "side": side,
        "alpha": alpha
    }