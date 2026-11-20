# SPDX-FileCopyrightText: Copyright (c) 2026 Liz Clark for Adafruit Industries
#
# SPDX-License-Identifier: MIT
"""Print an 8x8 grid of distances from the VL53L5CX."""

import time

import board

from adafruit_vl53l5cx import vl53l5cx

i2c = board.I2C()

# The firmware upload is about 84KB, so it will take a few seconds
print("Uploading firmware...")
sensor = vl53l5cx.VL53L5CX(i2c)
print("Firmware uploaded!")

sensor.resolution = vl53l5cx.RESOLUTION_8X8
sensor.ranging_frequency = 10
sensor.start_ranging()

width = 8 if sensor.resolution == vl53l5cx.RESOLUTION_8X8 else 4
print("Starting measurements:")

while True:
    if sensor.data_ready:
        data = sensor.ranging_data
        for row in range(width):
            zones = data.distance[row * width : (row + 1) * width]
            print(" ".join(f"{mm:5d}" for mm in zones))
        print()
    time.sleep(0.1)
