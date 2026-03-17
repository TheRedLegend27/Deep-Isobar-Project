# Module: Prediction Market API Integration

Purpose

Collect real-time contract prices from prediction markets.

Target Platforms

Kalshi
Polymarket

Inputs

contract id

Outputs

bid price
ask price
volume
last trade

Example

contract: NYC_HIGH_GE_85
bid: 0.51
ask: 0.54

Responsibilities

fetch market list
fetch order book
normalize probability format

Edge Cases

API rate limits
missing market data
invalid order books

Tasks

build API client
build market normalization layer
