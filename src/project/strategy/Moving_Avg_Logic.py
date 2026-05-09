import logging
import math
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from project.Entity.moving_avg_entity import MovingAvgEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    slope_threshold: float = 0.0005
    ema50_near_threshold: float = 0.005
    open_position_threshold: int = 3


DEFAULT_CONFIG = StrategyConfig()


def _validate_dataframe(df: MovingAvgEntity) -> None:

    required_series = ["close", "EMA_21", "SMA_200", "EMA_50", "VWMA_200"]
    missing_columns = [col for col in required_series if col not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for i in required_series:
        if len(df[i]) < 2:
            raise ValueError(
                f"Series '{i}' must contain at least 2 data points for slope/cross calculations"
            )


def _relative_slope(current_value: float, previous_value: float) -> float:
    if previous_value == 0:
        return 0.0
    return (current_value - previous_value) / previous_value


def _trend_score(
    close: float, indicator_value: float, slope: float, slope_threshold: float
) -> int:
    if close > indicator_value and slope > slope_threshold:
        return 1
    if close < indicator_value and slope < -slope_threshold:
        return -1
    return 0


def _ema50_score(close: float, ema50: float, near_threshold: float) -> int:
    if math.isclose(ema50, 0.0):
        return 0

    distance_ema50 = abs(close - ema50) / ema50
    near_ema50 = distance_ema50 < near_threshold

    if close > ema50 and near_ema50:
        return 1
    if close < ema50 and near_ema50:
        return -1
    return 0


def _ema21_cross_score(
    close: float, close_prev: float, ema21: float, ema21_prev: float
) -> int:
    bull_cross = close > ema21 and close_prev <= ema21_prev
    bear_cross = close < ema21 and close_prev >= ema21_prev

    if bull_cross:
        return 1
    if bear_cross:
        return -1
    return 0


def _calculate_scores(df: MovingAvgEntity, config: StrategyConfig) -> Dict[str, int]:

    for i in ["close", "SMA_200", "VWMA_200", "EMA_21", "EMA_50"]:
        if pd.isna(df[i].iloc[-1]):
            raise ValueError(f"Series '{i}' contains NaN values at the last position")

    close = float(df["close"].iloc[-1])
    close_prev = float(df["close"].iloc[-2])

    sma_200 = float(df["SMA_200"].iloc[-1])
    sma_200_prev = float(df["SMA_200"].iloc[-2])
    sma_slope = _relative_slope(sma_200, sma_200_prev)

    vwma_200 = float(df["VWMA_200"].iloc[-1])
    vwma_200_prev = float(df["VWMA_200"].iloc[-2])
    vwma_slope = _relative_slope(vwma_200, vwma_200_prev)

    ema_50 = float(df["EMA_50"].iloc[-1])
    ema_21 = float(df["EMA_21"].iloc[-1])
    ema_21_prev = float(df["EMA_21"].iloc[-2])

    return {
        "sma": _trend_score(close, sma_200, sma_slope, config.slope_threshold),
        "vwma": _trend_score(close, vwma_200, vwma_slope, config.slope_threshold),
        "ema50": _ema50_score(close, ema_50, config.ema50_near_threshold),
        "ema21": _ema21_cross_score(close, close_prev, ema_21, ema_21_prev),
    }


def _decide_position_action(
    total_score: int, current_position: int, open_position_threshold: int
) -> Tuple[int, str]:
    if current_position == 0:
        if total_score >= open_position_threshold:
            return 1, "open_long"
        if total_score <= -open_position_threshold:
            return -1, "open_short"
        return 0, "hold"

    if current_position == 1 and total_score <= 0:
        return 0, "close_long"
    if current_position == -1 and total_score >= 0:
        return 0, "close_short"

    return current_position, "hold"


def _log_prediction(score: Dict[str, int], total_score: int, action: str) -> None:
    logger.info("SMA Score: %s", score.get("sma", 0))
    logger.info("EMA_21 Score: %s", score.get("ema21", 0))
    logger.info("EMA_50 Score: %s", score.get("ema50", 0))
    logger.info("VWMA Score: %s", score.get("vwma", 0))
    logger.info("Total Score: %s", total_score)

    action_messages = {
        "open_long": "Open long position",
        "open_short": "Open short position",
        "close_long": "Close long position",
        "close_short": "Close short position",
        "hold": "Hold",
    }
    logger.info(action_messages[action])


def moving_avg_prediction(
    dataframe: MovingAvgEntity,
    current_position: int = 0,
    config: StrategyConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> dict:

    _validate_dataframe(dataframe)

    score = _calculate_scores(dataframe, config)
    total_score = sum(score.values())
    new_position, action = _decide_position_action(
        total_score, current_position, config.open_position_threshold
    )

    if verbose:
        _log_prediction(score, total_score, action)

    return {
        "scores": score,
        "total_score": total_score,
        "position": new_position,
        "action": action,
    }
