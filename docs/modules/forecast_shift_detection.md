# Module: Forecast Shift Detection

Purpose

Detect forecast changes between model runs.

Inputs

previous forecast
new forecast

Outputs

temperature shift
direction

Example

old forecast: 84
new forecast: 89

shift: +5

Responsibilities

detect forecast jumps
trigger probability recalculation

Edge Cases

missing previous forecast
extreme outlier shifts

Tasks

store forecast history
compute run-to-run change
