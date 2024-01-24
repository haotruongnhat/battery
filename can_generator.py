import numpy as np
import can
import time
import pandas as pd
import struct

signal = pd.read_pickle(r"modules\anomaly\dataset\battery\labeled\train\cmu.pkl")[0]

while(True):
    with can.interface.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=500000) as bus:
        for value in signal:
            byte_data =list(struct.pack("f", value))
            msg = can.Message(
                arbitration_id=0x1,
                data=byte_data,
                is_extended_id=False
            )
            try:
                bus.send(msg)
            except can.CanError:
                print("Message NOT sent")

            time.sleep(0.005)