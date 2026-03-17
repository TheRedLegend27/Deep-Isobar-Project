from datetime import date, datetime

from deep_isobar.core.types import MarketContract, OrderBookSnapshot


def fetch_live_contracts(market_source: str) -> list[MarketContract]:
    if market_source != "Kalshi":
        raise ValueError("This client only supports Kalshi")

    return [
        MarketContract(
            contract_id="CHI_HIGH_TEMP_F_GE_90_20260704",
            market_source="Kalshi",
            city="Chicago",
            metric="high_temp_f",
            comparison_operator="ge",
            threshold_f=90,
            target_date=date(2026, 7, 4),
            settlement_source="NWS",
        )
    ]


def fetch_orderbook_for_contract(
    market_source: str,
    contract_id: str,
) -> OrderBookSnapshot:
    if market_source != "Kalshi":
        raise ValueError("This client only supports Kalshi")

    return OrderBookSnapshot(
        timestamp_utc=datetime.utcnow(),
        contract_id=contract_id,
        market_source="Kalshi",
        best_bid=0.40,
        best_ask=0.46,
        last_trade_price=0.43,
        bid_size=10,
        ask_size=12,
        volume_24h=200,
        open_interest=500,
    )