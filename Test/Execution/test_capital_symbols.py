import pytest

from project.Execution.capital_symbols import CAPITAL_GOLD_SPEC, EPIC
from project.Risk.position_sizing import SymbolSpec, position_size

MT5_XAUUSD_SPEC = SymbolSpec(
    "XAUUSD",
    point=0.01,
    contract_size=100.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=100.0,
)


def test_capital_gold_spec_matches_confirmed_facts():
    assert EPIC == "GOLD"
    assert CAPITAL_GOLD_SPEC.symbol == "XAUUSD"
    assert CAPITAL_GOLD_SPEC.contract_size == 1.0
    assert CAPITAL_GOLD_SPEC.volume_min == 0.01
    assert CAPITAL_GOLD_SPEC.volume_step == 0.01
    assert CAPITAL_GOLD_SPEC.volume_max == 50000.0


@pytest.mark.parametrize(
    ("risk_fraction", "expected_capital_size", "expected_mt5_lots"),
    [
        (0.004, 2.0, 0.02),
        (0.004002, 2.0, 0.02),
    ],
)
def test_capital_size_matches_mt5_lots_in_ounces_after_rounding(
    risk_fraction, expected_capital_size, expected_mt5_lots
):
    capital_size = position_size(10_000, risk_fraction, 2_000, 1_980, CAPITAL_GOLD_SPEC)
    mt5_lots = position_size(10_000, risk_fraction, 2_000, 1_980, MT5_XAUUSD_SPEC)

    assert capital_size == expected_capital_size
    assert mt5_lots == expected_mt5_lots
    assert capital_size == pytest.approx(mt5_lots * 100)


def test_capital_size_below_minimum_matches_existing_position_size_behavior():
    assert position_size(10, 0.005, 100, 80, CAPITAL_GOLD_SPEC) == 0.0
    assert position_size(10, 0.005, 100, 80, MT5_XAUUSD_SPEC) == 0.0


def test_capital_size_above_maximum_matches_existing_position_size_behavior():
    assert position_size(10_000_000, 0.01, 100, 99, CAPITAL_GOLD_SPEC) == 50000.0
    assert position_size(10_000_000, 0.01, 100, 99, MT5_XAUUSD_SPEC) == 100.0
