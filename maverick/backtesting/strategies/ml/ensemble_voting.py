"""Weighted signal voting for `StrategyEnsemble`, split out of `ensemble.py` to
keep that module under this repo's 500-line-per-module cap.
Accepts either current per-strategy weights or a causal per-bar weight history.
The voting thresholds and conflict resolution apply independently to each bar.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series

logger = logging.getLogger(__name__)


def combine_weighted_signals(
    individual_signals: dict[int, tuple[Series, Series]],
    weights: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[Series, Series]:
    """Combine individual strategy signals using enhanced weighted voting.

    Args:
        individual_signals: Dictionary of individual strategy signals
        weights: Per-strategy vector or (bars, strategies) weight history
        parameters: Ensemble parameters (`voting_method`, `entry_threshold`,
            `exit_threshold`, `min_signal_strength`)

    Returns:
        Tuple of combined (entry_signals, exit_signals)
    """
    if not individual_signals:
        empty_index = pd.Index([])
        return pd.Series(False, index=empty_index), pd.Series(False, index=empty_index)

    # Get data index from first strategy
    first_signals = next(iter(individual_signals.values()))
    data_index = first_signals[0].index

    # Initialize voting arrays
    entry_votes = np.zeros(len(data_index))
    exit_votes = np.zeros(len(data_index))
    total_weights = np.zeros(len(data_index))

    # Collect votes with weights and confidence scores
    valid_strategies = 0

    for i, (entry_signals, exit_signals) in individual_signals.items():
        strategy_count = weights.shape[-1]
        weight = (
            (weights[i] if weights.ndim == 1 else weights[:, i])
            if i < strategy_count
            else 0
        )

        if np.any(weight > 0):
            # Add weighted votes
            entry_votes += weight * entry_signals.astype(float)
            exit_votes += weight * exit_signals.astype(float)
            total_weights += weight
            valid_strategies += 1

    if not np.any(total_weights > 0) or valid_strategies == 0:
        logger.warning("No valid strategies with positive weights")
        return pd.Series(False, index=data_index), pd.Series(False, index=data_index)

    # Normalize votes by total weights
    entry_votes = np.divide(
        entry_votes,
        total_weights,
        out=np.zeros_like(entry_votes),
        where=total_weights > 0,
    )
    exit_votes = np.divide(
        exit_votes,
        total_weights,
        out=np.zeros_like(exit_votes),
        where=total_weights > 0,
    )

    # Enhanced voting mechanisms
    voting_method = parameters.get("voting_method", "weighted")

    if voting_method == "majority":
        # Simple majority vote (more than half of strategies agree)
        entry_threshold = 0.5
        exit_threshold = 0.5
    elif voting_method == "supermajority":
        # Require 2/3 agreement
        entry_threshold = 0.67
        exit_threshold = 0.67
    elif voting_method == "consensus":
        # Require near-unanimous agreement
        entry_threshold = 0.8
        exit_threshold = 0.8
    else:  # weighted (default)
        entry_threshold = parameters.get("entry_threshold", 0.5)
        exit_threshold = parameters.get("exit_threshold", 0.5)

    # Anti-conflict mechanism: don't signal entry and exit simultaneously
    combined_entry = entry_votes > entry_threshold
    combined_exit = exit_votes > exit_threshold

    # Resolve conflicts (simultaneous entry and exit signals)
    conflicts = combined_entry & combined_exit
    if conflicts.size > 0 and np.any(conflicts):
        logger.debug(f"Resolving {conflicts.sum()} signal conflicts")
        entry_strength = entry_votes[conflicts]
        exit_strength = exit_votes[conflicts]

        stronger_entry = entry_strength > exit_strength
        combined_entry[conflicts] = stronger_entry
        combined_exit[conflicts] = ~stronger_entry

    # Quality filter: require minimum signal strength
    min_signal_strength = parameters.get("min_signal_strength", 0.1)
    weak_entry_signals = (combined_entry) & (entry_votes < min_signal_strength)
    weak_exit_signals = (combined_exit) & (exit_votes < min_signal_strength)

    if weak_entry_signals.size > 0:
        combined_entry[weak_entry_signals] = False
    if weak_exit_signals.size > 0:
        combined_exit[weak_exit_signals] = False

    combined_entry = pd.Series(combined_entry, index=data_index)
    combined_exit = pd.Series(combined_exit, index=data_index)

    return combined_entry, combined_exit
