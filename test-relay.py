#!/usr/bin/env python3
# coding=utf8


import time as t
import smbus
import sys

DEVICE_BUS = 1
DEVICE_ADDR = 0x10
bus = smbus.SMBus(DEVICE_BUS)

while True:
    try:
        for i in range(1,5):
            bus.write_byte_data(DEVICE_ADDR, i, 0xFF)
            t.sleep(1)
            bus.write_byte_data(DEVICE_ADDR, i, 0x00)
            t.sleep(1) 
    except KeyboardInterrupt as e:
        print("Quit the Loop")
        bus.write_byte_data(DEVICE_ADDR, 1, 0x00)
        bus.write_byte_data(DEVICE_ADDR, 2, 0x00)
        bus.write_byte_data(DEVICE_ADDR, 3, 0x00)
        bus.write_byte_data(DEVICE_ADDR, 4, 0x00)
        sys.exit()

