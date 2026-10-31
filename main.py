import logging

import MetaTrader5 as mt5
import pandas as pd
from project.Execution.mt5_broker import MT5Broker
from project.Risk.position_sizing import SymbolSpec

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s"
)


if __name__ == "__main__":
    SYMBOL = "XAUUSD"  # ใช้ชื่อเดียวกับใน Market Watch

    mt5.initialize()
    info = mt5.symbol_info(SYMBOL)
    spec = SymbolSpec(
        symbol=SYMBOL,
        point=info.point,
        contract_size=info.trade_contract_size,
        volume_min=info.volume_min,
        volume_step=info.volume_step,
        volume_max=info.volume_max,
    )

    b = MT5Broker(spec, SYMBOL)
    print("เชื่อมต่อสำเร็จ equity =", b.equity())

    tick = mt5.symbol_info_tick(SYMBOL)
    b.open(
        "long", spec.volume_min, tick.ask, 0, stop=tick.ask - 20, ts=pd.Timestamp.now()
    )
    print("เปิดแล้ว:", b.position)

    input("ดูออเดอร์ในแท็บ Trade ของ MT5 แล้วกด Enter เพื่อปิด...")

    trade = b.close(0, 0, pd.Timestamp.now(), "manual")
    print("ปิดแล้ว:", trade)

    mt5.shutdown()
