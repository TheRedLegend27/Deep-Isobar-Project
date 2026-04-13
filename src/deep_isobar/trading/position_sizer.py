"""Dynamic position sizer for Deep Isobar.

Computes a session exposure cap that scales down from a base dollar amount
based on two runtime signals:

1. **Anomaly confidence** — if the anomaly detector flagged problems, reduce
   exposure proportionally.
2. **Ensemble spread** — if GFS ensemble members disagree widely (high std),
   the model itself is uncertain; reduce exposure.

Interface::

    compute_exposure(anomaly_report, ensemble_std_f, cfg) -> SizingDecision
    log_sizing_decision(decision, logger) -> None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class SizingDecision:
    """Result of one dynamic sizing computation for a session."""

    final_exposure_usd: float
    base_exposure_usd: float
    anomaly_multiplier: float
    flag_multiplier: float        # product of all per-flag penalties
    spread_multiplier: float
    reasoning: str                # human-readable explanation for logging


def compute_exposure(
    anomaly_report: Any | None,
    ensemble_std_f: float | None,
    cfg: dict,
) -> SizingDecision:
    """Compute a dynamic exposure cap for the session.

    Args:
        anomaly_report: An ``AnomalyReport`` object (with ``.confidence`` and
            ``.flags`` attributes), or ``None`` if the anomaly check did not run.
        ensemble_std_f: GFS ensemble standard deviation in °F, or ``None`` if
            unavailable.
        cfg: The ``dynamic_sizing`` config dict from ``risk.multi_bracket``.

    Returns:
        A :class:`SizingDecision` describing the final cap and the reasoning.
    """
    base: float = cfg["base_exposure_usd"]
    min_exposure: float = cfg["min_exposure_usd"]
    anomaly_multipliers: dict = cfg.get("anomaly_multipliers", {})
    flag_penalties: dict = cfg.get("flag_penalties", {})
    thresholds: dict = cfg.get("spread_thresholds", {})

    # ── 1. Confidence multiplier ──────────────────────────────────────────────
    if anomaly_report is None:
        confidence_key = "NONE"
    else:
        confidence_key = (anomaly_report.confidence or "NONE").upper()

    confidence_mult: float = float(anomaly_multipliers.get(confidence_key, 1.0))

    # ── 2. Per-flag penalty (multiplicative) ──────────────────────────────────
    flag_mult: float = 1.0
    applied_flags: list[str] = []
    if anomaly_report is not None and anomaly_report.flags:
        for flag in anomaly_report.flags:
            code = flag.code if hasattr(flag, "code") else str(flag)
            if code in flag_penalties:
                flag_mult *= float(flag_penalties[code])
                applied_flags.append(code)

    # ── 3. Spread multiplier ──────────────────────────────────────────────────
    clean_thresh: float = float(thresholds.get("clean", 3.0))
    moderate_thresh: float = float(thresholds.get("moderate", 5.0))

    if ensemble_std_f is None:
        spread_mult = 1.0
        spread_label: str | None = None
    elif ensemble_std_f < clean_thresh:
        spread_mult = 1.0
        spread_label = None
    elif ensemble_std_f < moderate_thresh:
        spread_mult = 0.80
        spread_label = "moderate spread"
    else:
        spread_mult = 0.60
        spread_label = "wide spread"

    # ── 4. Raw final, clamp, round ────────────────────────────────────────────
    raw = base * confidence_mult * flag_mult * spread_mult
    clamped = max(min_exposure, min(base, raw))
    final = round(clamped, 2)

    # ── 5. Build reasoning string ─────────────────────────────────────────────
    parts: list[str] = [f"${base:.2f} base"]

    if confidence_mult != 1.0:
        parts.append(f"× {confidence_mult:.2f} ({confidence_key} confidence)")

    for code in applied_flags:
        penalty = float(flag_penalties[code])
        parts.append(f"× {penalty:.2f} ({code})")

    if spread_label is not None:
        parts.append(f"× {spread_mult:.2f} ({spread_label})")

    pre_clamp_rounded = round(raw, 2)
    parts.append(f"= ${pre_clamp_rounded:.2f} → clamped to ${final:.2f}")

    reasoning = " ".join(parts)

    return SizingDecision(
        final_exposure_usd=final,
        base_exposure_usd=base,
        anomaly_multiplier=confidence_mult,
        flag_multiplier=round(flag_mult, 6),
        spread_multiplier=spread_mult,
        reasoning=reasoning,
    )


def log_sizing_decision(decision: SizingDecision, logger: logging.Logger) -> None:
    """Log the sizing decision.

    Logs at INFO level when exposure was reduced from base; at DEBUG when
    the final exposure equals the base (no change — suppressed in normal runs).

    Args:
        decision: A :class:`SizingDecision` from :func:`compute_exposure`.
        logger: Logger instance to write to.
    """
    if decision.final_exposure_usd == decision.base_exposure_usd:
        logger.debug("Dynamic sizing: no adjustment — %s", decision.reasoning)
    else:
        logger.info("Dynamic sizing: %s", decision.reasoning)
