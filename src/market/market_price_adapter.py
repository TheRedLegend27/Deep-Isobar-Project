def implied_probability(bid, ask):

    if bid > ask:
        raise ValueError("Invalid order book")

    return (bid + ask) / 2