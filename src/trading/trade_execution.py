def execute_trade(signal):

    if signal["side"] == "HOLD":
        return

    print("Executing trade:", signal)