from market.market_scanner import scan_markets
from trading.trade_execution import execute_trade
from trading.risk_manager import approve_trade


def main():

    opportunities = scan_markets()

    for trade_signal in opportunities:

        if approve_trade(trade_signal):
            execute_trade(trade_signal)


if __name__ == "__main__":
    main()