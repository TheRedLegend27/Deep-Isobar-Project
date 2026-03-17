from trading.alpha_engine import generate_signal


def scan_markets():

    opportunities = []

    markets = [
        {
            "contract": "NYC_HIGH_GE_85",
            "model_prob": 0.31,
            "market_prob": 0.52
        }
    ]

    for m in markets:

        signal = generate_signal(
            m["contract"],
            m["model_prob"],
            m["market_prob"]
        )

        opportunities.append(signal)

    return opportunities