from project.Risk.position_sizing import SymbolSpec

EPIC = "GOLD"

CAPITAL_GOLD_SPEC = SymbolSpec(
    "XAUUSD",
    contract_size=1.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=50000.0,
)
