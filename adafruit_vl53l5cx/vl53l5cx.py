# SPDX-FileCopyrightText: Copyright (c) 2026 Liz Clark for Adafruit Industries
#
# SPDX-License-Identifier: MIT
"""
:py:class:`~adafruit_vl53l5cx.vl53l5cx`
================================================================================

CircuitPython driver for the Adafruit VL53L5CX Time-of-Flight Distance 8x8 Pixel Camera


* Author(s): Liz Clark

Implementation Notes
--------------------

**Hardware:**

* `Adafruit VL53L5CX Time-of-Flight Distance 8x8 Pixel Camera <https://www.adafruit.com/product/6503>`_"

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice
* Adafruit's Register library: https://github.com/adafruit/Adafruit_CircuitPython_Register
"""

import struct
import time

from adafruit_bus_device.i2c_device import I2CDevice
from adafruit_register.register_accessor import I2CRegisterAccessor
from adafruit_register.register_struct import ROStruct, ROUnaryStruct, UnaryStruct
from micropython import const

try:
    from typing import List, Optional

    from busio import I2C
except ImportError:
    pass

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_VL53L5CX.git"

_DEFAULT_ADDR = const(0x29)

# Resolution
RESOLUTION_4X4 = const(16)
RESOLUTION_8X8 = const(64)

# Target order
TARGET_ORDER_CLOSEST = const(1)
TARGET_ORDER_STRONGEST = const(2)

# Ranging mode
RANGING_MODE_CONTINUOUS = const(1)
RANGING_MODE_AUTONOMOUS = const(3)

# Power mode
POWER_MODE_SLEEP = const(0)
POWER_MODE_WAKEUP = const(1)

# Registers
_PAGE = const(0x7FFF)
_DEVICE_ID = const(0x0000)  # page 0
_REV_ID = const(0x0001)  # page 0
_FW_ACCESS = const(0x0003)  # page 2
_I2C_ADDR = const(0x0004)  # page 0
_GO2_STATUS0 = const(0x0006)  # page 0
_GO2_STATUS1 = const(0x0007)  # page 0
_PWR_CTRL = const(0x0009)  # page 0
_GO1_ACCESS = const(0x000C)  # page 0
_MCU_STOP0 = const(0x0014)  # page 0
_MCU_STOP1 = const(0x0015)  # page 0
_MCU_WAKE = const(0x0020)  # page 1
_FW_STATUS = const(0x0021)  # page 1
_UI_HEADER = const(0x0000)  # page 2, start of the results frame

# Mailbox (page 2)
_UI_CMD_STATUS = const(0x2C00)
_UI_CMD_START = const(0x2C04)
_UI_CMD_END = const(0x2FFF)
_CONFIG_CMD = const(0x2C34)
_XTALK_CMD = const(0x2CF8)
_OFFSET_CMD = const(0x2E18)
_NVM_CMD = const(0x2FD8)
_AUTO_STOP = const(0x2FFC)

# DCI (device configuration interface) indexes
_DCI_ZONE_CFG = const(0x5450)
_DCI_FREQ = const(0x5458)
_DCI_INT_TIME = const(0x545C)
_DCI_RNG_MODE = const(0xAD30)
_DCI_DSS_CFG = const(0xAD38)
_DCI_TGT_ORDER = const(0xAE64)
_DCI_SHARPENER = const(0xAED8)
_DCI_GLARE = const(0xB870)
_DCI_SINGLE_RNG = const(0xD964)
_DCI_OUT_CFG = const(0xD968)
_DCI_OUT_EN = const(0xD970)
_DCI_OUT_LIST = const(0xD980)
_DCI_PIPE_CTRL = const(0xDB80)

# Results block indexes
_META_IDX = const(0x54B4)
_NB_TGT_IDX = const(0xDB84)
_SIGMA_IDX = const(0xDEC4)
_DIST_IDX = const(0xDF44)
_STATUS_IDX = const(0xE084)

# Output block headers requested from the firmware (1 target per zone):
# start, metadata, common data, ambient, SPAD count, number of targets,
# signal rate, range sigma, distance, reflectance, target status, motion
_OUTPUT_BH = (
    0x0000000D,
    0x54B400C0,
    0x54C00040,
    0x54D00104,
    0x55D00404,
    0xDB840401,
    0xDBC40404,
    0xDEC40402,
    0xDF440402,
    0xE0440401,
    0xE0840401,
    0xD85808C0,
)

_FW_SIZE = const(0x15000)
_NVM_SIZE = const(492)
_OFFSET_SIZE = const(488)
_XTALK_SIZE = const(776)

# I2C transfer chunk size
_CHUNK = const(1024)

_BLOB_MAGIC = b"VL53"
_BLOB_HEADER_SIZE = const(20)

_POLL_TIMEOUT = 2.0
_BOOT_TIMEOUT = 0.5
_STOP_TIMEOUT = 5.0


def _swap32(buf: bytearray, start: int = 0, end: Optional[int] = None) -> None:
    # The sensor is big-endian per 32-bit word; flip each word in place
    if end is None:
        end = len(buf)
    for i in range(start, end, 4):
        buf[i], buf[i + 1], buf[i + 2], buf[i + 3] = buf[i + 3], buf[i + 2], buf[i + 1], buf[i]


def _average_2x2(values: tuple, signed: bool) -> List[int]:
    # Collapse an 8x8 grid into 4x4 by averaging each 2x2 block, padded back
    # to 64 entries with zeros. Matches the ST integer math.
    result = [0] * 64
    for j in range(4):
        for i in range(4):
            k = (2 * i) + (16 * j)
            total = values[k] + values[k + 1] + values[k + 8] + values[k + 9]
            if signed:
                # C division truncates toward zero
                result[i + (4 * j)] = -(-total // 4) if total < 0 else total // 4
            else:
                result[i + (4 * j)] = (total & 0xFFFFFFFF) // 4
    return result


class RangingData:
    """One frame of ranging results from the VL53L5CX. Each list has one entry
    per zone (16 or 64), in row-major order.

    :param list distance: Distance to target per zone in millimeters
    :param list target_status: Measurement status per zone. 5 and 9 are valid
        measurements, 255 means no target was detected.
    :param list range_sigma: Estimated ranging noise per zone in millimeters
    :param int temperature: Silicon temperature in degrees Celsius
    """

    def __init__(
        self,
        distance: List[int],
        target_status: List[int],
        range_sigma: List[int],
        temperature: int,
    ) -> None:
        self.distance = distance
        self.target_status = target_status
        self.range_sigma = range_sigma
        self.temperature = temperature


class VL53L5CX:
    """Driver for the VL53L5CX 8x8 Time-of-Flight sensor.

    Initialization uploads about 84KB of firmware, so it takes a few seconds.

    :param ~busio.I2C i2c: The I2C bus the sensor is connected to
    :param int address: The I2C address of the sensor. Defaults to 0x29.
    :param str firmware_path: Path to a firmware file. Defaults to the
        ``vl53l5cx_firmware.bin`` installed with this library.
    """

    _page = UnaryStruct(_PAGE, "B")
    _device_id = ROUnaryStruct(_DEVICE_ID, "B")
    _revision_id = ROUnaryStruct(_REV_ID, "B")
    _i2c_addr = UnaryStruct(_I2C_ADDR, "B")
    _go2_status0 = ROUnaryStruct(_GO2_STATUS0, "B")
    _go2_status1 = ROUnaryStruct(_GO2_STATUS1, "B")
    _pwr_ctrl = UnaryStruct(_PWR_CTRL, "B")
    _auto_stop = ROUnaryStruct(_AUTO_STOP, "<I")
    _ui_header = ROStruct(_UI_HEADER, "4B")

    def __init__(
        self,
        i2c: I2C,
        address: int = _DEFAULT_ADDR,
        firmware_path: Optional[str] = None,
    ) -> None:
        self._i2c = i2c
        self.register_accessor = I2CRegisterAccessor(
            I2CDevice(i2c, address), address_width=2, lsb_first=False
        )
        self._scratch = bytearray(4)
        self._byte = bytearray(1)
        self._streamcount = 255
        self._frame = None
        self._resolution = RESOLUTION_4X4
        self._offset_data = None
        self._xtalk_data = None

        self._page = 0x00
        device_id = self._device_id
        revision_id = self._revision_id
        self._page = 0x02
        if device_id != 0xF0 or revision_id != 0x02:
            raise RuntimeError("Failed to find VL53L5CX")

        if firmware_path is None:
            firmware_path = f"{__file__.rsplit('/', 1)[0]}/vl53l5cx_firmware.bin"
        try:
            blob = open(firmware_path, "rb")
        except OSError as error:
            raise RuntimeError(f"Unable to open VL53L5CX firmware file {firmware_path}") from error

        with blob:
            _, config_size, xtalk_size, nvm_size = self._read_blob_header(blob)
            self._sw_reboot()
            self._upload_firmware(blob)
            config = blob.read(config_size)
            self._xtalk_data = blob.read(xtalk_size)
            nvm_cmd = blob.read(nvm_size)
        if len(config) != config_size or len(nvm_cmd) != nvm_size:
            raise ValueError("VL53L5CX firmware file is truncated")

        self._load_calibration(nvm_cmd)
        self._send_default_config(config)

    def _write_byte(self, register: int, value: int) -> None:
        self._byte[0] = value
        self.register_accessor.write_register(register, self._byte)

    def _write_seq(self, pairs: tuple) -> None:
        for register, value in pairs:
            self._write_byte(register, value)

    def _write_multi(self, register: int, data) -> None:
        view = memoryview(data)
        for offset in range(0, len(view), _CHUNK):
            self.register_accessor.write_register(register + offset, view[offset : offset + _CHUNK])

    def _read_multi(self, register: int, buf: bytearray) -> None:
        view = memoryview(buf)
        for offset in range(0, len(view), _CHUNK):
            self.register_accessor.read_register(register + offset, view[offset : offset + _CHUNK])

    def _poll_for_answer(
        self, size: int, pos: int, register: int, mask: int, expected: int
    ) -> None:
        buf = memoryview(self._scratch)[:size]
        deadline = time.monotonic() + _POLL_TIMEOUT
        while True:
            self.register_accessor.read_register(register, buf)
            if size >= 4 and buf[2] >= 0x7F:
                raise RuntimeError("VL53L5CX MCU error")
            if (buf[pos] & mask) == expected:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting on VL53L5CX register 0x{register:04X}")
            time.sleep(0.001)

    def _poll_for_boot(self) -> None:
        deadline = time.monotonic() + _BOOT_TIMEOUT
        while True:
            status = self._go2_status0
            if status & 0x80:
                if self._go2_status1 != 0:
                    raise RuntimeError("VL53L5CX failed to boot")
                return
            if status & 0x01:
                return
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for VL53L5CX to boot")
            time.sleep(0.001)

    def _dci_read(self, index: int, size: int) -> bytearray:
        cmd = bytes(
            (
                index >> 8,
                index & 0xFF,
                (size & 0xFF0) >> 4,
                (size & 0x0F) << 4,
                0x00,
                0x00,
                0x00,
                0x0F,
                0x00,
                0x02,
                0x00,
                0x08,
            )
        )
        self._write_multi(_UI_CMD_END - 11, cmd)
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)
        buf = bytearray(size + 12)
        self._read_multi(_UI_CMD_START, buf)
        _swap32(buf)
        return buf[4 : 4 + size]

    def _dci_write(self, index: int, data) -> None:
        size = len(data)
        frame = bytearray(size + 12)
        frame[0] = index >> 8
        frame[1] = index & 0xFF
        frame[2] = (size & 0xFF0) >> 4
        frame[3] = (size & 0x0F) << 4
        frame[4 : 4 + size] = data
        _swap32(frame, 4, 4 + size)
        frame[4 + size : 4 + size + 6] = b"\x00\x00\x00\x0f\x05\x01"
        frame[size + 10] = (size + 8) >> 8
        frame[size + 11] = (size + 8) & 0xFF
        self._write_multi(_UI_CMD_END - (size + 12) + 1, frame)
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)

    def _dci_replace(self, index: int, total_size: int, offset: int, data) -> None:
        buf = self._dci_read(index, total_size)
        buf[offset : offset + len(data)] = data
        self._dci_write(index, buf)

    @staticmethod
    def _read_blob_header(blob) -> tuple:
        header = blob.read(_BLOB_HEADER_SIZE)
        if len(header) != _BLOB_HEADER_SIZE or header[:4] != _BLOB_MAGIC:
            raise ValueError("Not a VL53L5CX firmware file")
        sizes = struct.unpack_from("<4I", header, 4)
        if sizes[0] != _FW_SIZE or sizes[2] != _XTALK_SIZE:
            raise ValueError("Unexpected VL53L5CX firmware file layout")
        return sizes

    def _sw_reboot(self) -> None:
        self._page = 0x00
        self._write_seq(((0x0009, 0x04), (0x000F, 0x40), (0x000A, 0x03)))
        _ = self._page
        self._write_seq(
            (
                (_GO1_ACCESS, 0x01),
                (0x0101, 0x00),
                (0x0102, 0x00),
                (0x010A, 0x01),
                (0x4002, 0x01),
                (0x4002, 0x00),
                (0x010A, 0x03),
                (0x0103, 0x01),
                (_GO1_ACCESS, 0x00),
                (0x000F, 0x43),
            )
        )
        time.sleep(0.001)
        self._write_seq(((0x000F, 0x40), (0x000A, 0x01)))
        time.sleep(0.1)

        # Wait for sensor boot
        self._page = 0x00
        self._poll_for_answer(1, 0, _GO2_STATUS0, 0xFF, 0x01)
        self._write_byte(0x000E, 0x01)
        self._page = 0x02

        # Enable FW access
        self._write_byte(_FW_ACCESS, 0x0D)
        self._page = 0x01
        self._poll_for_answer(1, 0, _FW_STATUS, 0x10, 0x10)
        self._page = 0x00

        # Enable host access to GO1
        _ = self._page
        self._write_byte(_GO1_ACCESS, 0x01)

        # Power ON status
        self._page = 0x00
        self._write_seq(
            (
                (0x0101, 0x00),
                (0x0102, 0x00),
                (0x010A, 0x01),
                (0x4002, 0x01),
                (0x4002, 0x00),
                (0x010A, 0x03),
                (0x0103, 0x01),
                (0x400F, 0x00),
                (0x021A, 0x43),
                (0x021A, 0x03),
                (0x021A, 0x01),
                (0x021A, 0x00),
                (0x0219, 0x00),
                (0x021B, 0x00),
            )
        )

        # Wake up MCU
        self._page = 0x00
        _ = self._page
        self._write_byte(_GO1_ACCESS, 0x00)
        self._page = 0x01
        self._write_seq(((_MCU_WAKE, 0x07), (_MCU_WAKE, 0x06)))

    def _upload_firmware(self, blob) -> None:
        # Stream the firmware from the file across pages 0x09, 0x0A and 0x0B
        chunk = bytearray(_CHUNK)
        for page, size in ((0x09, 0x8000), (0x0A, 0x8000), (0x0B, 0x5000)):
            self._page = page
            for offset in range(0, size, _CHUNK):
                if blob.readinto(chunk) != _CHUNK:
                    raise ValueError("VL53L5CX firmware file is truncated")
                self.register_accessor.write_register(offset, chunk)
        self._page = 0x01

        # Verify FW download
        self._page = 0x02
        self._write_byte(_FW_ACCESS, 0x0D)
        self._page = 0x01
        self._poll_for_answer(1, 0, _FW_STATUS, 0x10, 0x10)
        self._page = 0x00
        _ = self._page
        self._write_byte(_GO1_ACCESS, 0x01)

        # Reset MCU and wait for boot
        self._page = 0x00
        self._write_seq(
            ((0x0114, 0x00), (0x0115, 0x00), (0x0116, 0x42), (0x0117, 0x00), (0x000B, 0x00))
        )
        _ = self._page
        self._write_seq(((_GO1_ACCESS, 0x00), (0x000B, 0x01)))
        self._poll_for_boot()
        self._page = 0x02

    def _send_offset_data(self, resolution: int) -> None:
        buf = bytearray(self._offset_data)
        if resolution == RESOLUTION_4X4:
            buf[0x10:0x18] = b"\x0f\x04\x04\x00\x08\x10\x10\x07"
            _swap32(buf)
            signal = _average_2x2(struct.unpack_from("<64I", buf, 0x3C), False)
            ranges = _average_2x2(struct.unpack_from("<64h", buf, 0x140), True)
            struct.pack_into("<64I", buf, 0x3C, *signal)
            struct.pack_into("<64h", buf, 0x140, *ranges)
            _swap32(buf)
        # Strip the 8 byte DCI header and append the footer
        buf[0 : _OFFSET_SIZE - 8] = buf[8:_OFFSET_SIZE]
        buf[_OFFSET_SIZE - 8 : _OFFSET_SIZE] = b"\x00\x00\x00\x0f\x03\x01\x01\xe4"
        self._write_multi(_OFFSET_CMD, buf)
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)

    def _send_xtalk_data(self, resolution: int) -> None:
        buf = bytearray(self._xtalk_data)
        if resolution == RESOLUTION_4X4:
            buf[0x08:0x10] = b"\x0f\x04\x04\x17\x08\x10\x10\x07"
            buf[0x20:0x28] = b"\x00\x78\x00\x08\x00\x00\x00\x08"
            _swap32(buf)
            signal = _average_2x2(struct.unpack_from("<64I", buf, 0x34), False)
            struct.pack_into("<64I", buf, 0x34, *signal)
            _swap32(buf)
            buf[0x134:0x138] = b"\xa0\xfc\x01\x00"
            buf[0x78:0x7C] = b"\x00\x00\x00\x00"
        self._write_multi(_XTALK_CMD, buf)
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)

    def _load_calibration(self, nvm_cmd: bytes) -> None:
        self._write_multi(_NVM_CMD, nvm_cmd)
        self._poll_for_answer(4, 0, _UI_CMD_STATUS, 0xFF, 0x02)
        nvm = bytearray(_NVM_SIZE)
        self._read_multi(_UI_CMD_START, nvm)
        self._offset_data = nvm[:_OFFSET_SIZE]
        self._send_offset_data(RESOLUTION_4X4)
        self._send_xtalk_data(RESOLUTION_4X4)

    def _send_default_config(self, config: bytes) -> None:
        self._write_multi(_CONFIG_CMD, config)
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)
        self._dci_write(_DCI_PIPE_CTRL, b"\x01\x00\x01\x00")
        self._dci_write(_DCI_SINGLE_RNG, struct.pack("<I", 1))
        self._dci_replace(_DCI_GLARE, 40, 0x26, b"\x01")
        self._dci_replace(_DCI_GLARE, 40, 0x25, b"\x01")

    def start_ranging(self) -> None:
        """Start a ranging session. Set the resolution before calling this, since
        the size of each results frame is fixed when ranging starts."""
        resolution = self.resolution
        self._streamcount = 255

        # Patch each data block's size for the current resolution
        headers = []
        read_size = 0
        for block in _OUTPUT_BH:
            bh_type = block & 0x0F
            if 0x01 <= bh_type < 0x0D:
                header = (block & 0xFFFF000F) | (resolution << 4)
                read_size += bh_type * resolution
            else:
                header = block
                read_size += (block >> 4) & 0xFFF
            read_size += 4
            headers.append(header)
        read_size += 24

        self._dci_write(_DCI_OUT_LIST, struct.pack("<12I", *headers))
        self._dci_write(_DCI_OUT_CFG, struct.pack("<2I", read_size, len(headers) + 1))
        self._dci_write(_DCI_OUT_EN, struct.pack("<4I", 0x00000FFF, 0, 0, 0xC0000000))
        self._frame = bytearray(read_size)
        self._resolution = resolution

        # Start xshut bypass (interrupt mode)
        self._page = 0x00
        self._pwr_ctrl = 0x05
        self._page = 0x02

        # Start ranging session
        self._write_multi(_UI_CMD_END - 3, b"\x00\x03\x00\x00")
        self._poll_for_answer(4, 1, _UI_CMD_STATUS, 0xFF, 0x03)

    def stop_ranging(self) -> None:
        """Stop the current ranging session."""
        if self._auto_stop != 0x4FF:
            self._page = 0x00

            # MCU stop and wait for it
            self._write_byte(_MCU_STOP1, 0x16)
            self._write_byte(_MCU_STOP0, 0x01)
            deadline = time.monotonic() + _STOP_TIMEOUT
            while not self._go2_status0 & 0x80:
                time.sleep(0.01)
                if time.monotonic() > deadline:
                    break

        # Read GO2 status 1 when the MCU reports stopped
        if self._go2_status0 & 0x80:
            _ = self._go2_status1

        # Undo MCU stop
        self._page = 0x00
        self._write_byte(_MCU_STOP0, 0x00)
        self._write_byte(_MCU_STOP1, 0x00)

        # Stop xshut bypass
        self._pwr_ctrl = 0x04
        self._page = 0x02

    @property
    def data_ready(self) -> bool:
        """True when a new frame of ranging data is available."""
        stream, status, flags, int_flags = self._ui_header
        if (
            stream not in {self._streamcount, 255}
            and status == 0x05
            and (flags & 0x05) == 0x05
            and (int_flags & 0x10) == 0x10
        ):
            self._streamcount = stream
            return True
        return False

    @property
    def ranging_data(self) -> RangingData:
        """Read the latest frame of ranging data. Check `data_ready` first."""
        if self._frame is None:
            raise RuntimeError("Call start_ranging() before reading data")
        buf = self._frame
        self._read_multi(_UI_HEADER, buf)
        self._streamcount = buf[0]

        distance = target_status = range_sigma = nb_targets = None
        temperature = 0
        i = 16
        while i < len(buf):
            header = struct.unpack_from(">I", buf, i)[0]
            bh_type = header & 0x0F
            count = (header >> 4) & 0xFFF
            index = header >> 16
            msize = bh_type * count if 0x01 < bh_type < 0x0D else count
            start = i + 4
            if index == _META_IDX:
                temperature = struct.unpack_from("b", buf, start + 11)[0]
            elif index == _DIST_IDX:
                raw = struct.unpack_from(f">{count}h", buf, start)
                distance = [max(raw[n ^ 1], 0) // 4 for n in range(count)]
            elif index == _SIGMA_IDX:
                raw = struct.unpack_from(f">{count}H", buf, start)
                range_sigma = [raw[n ^ 1] >> 7 for n in range(count)]
            elif index == _STATUS_IDX:
                target_status = [buf[start + (n ^ 3)] for n in range(count)]
            elif index == _NB_TGT_IDX:
                nb_targets = [buf[start + (n ^ 3)] for n in range(count)]
            i = start + msize

        if target_status is not None and nb_targets is not None:
            for n, targets in enumerate(nb_targets):
                if targets == 0:
                    target_status[n] = 255
        return RangingData(distance, target_status, range_sigma, temperature)

    @property
    def resolution(self) -> int:
        """Number of ranging zones, ``RESOLUTION_4X4`` (16) or ``RESOLUTION_8X8`` (64).
        Change this while ranging is stopped."""
        zone_cfg = self._dci_read(_DCI_ZONE_CFG, 8)
        return zone_cfg[0] * zone_cfg[1]

    @resolution.setter
    def resolution(self, value: int) -> None:
        if value not in {RESOLUTION_4X4, RESOLUTION_8X8}:
            raise ValueError("Resolution must be RESOLUTION_4X4 or RESOLUTION_8X8")
        dss = self._dci_read(_DCI_DSS_CFG, 16)
        zone = self._dci_read(_DCI_ZONE_CFG, 8)
        if value == RESOLUTION_4X4:
            dss[0x04], dss[0x06], dss[0x09] = 64, 64, 4
            zone[0x00], zone[0x01], zone[0x04], zone[0x05] = 4, 4, 8, 8
        else:
            dss[0x04], dss[0x06], dss[0x09] = 16, 16, 1
            zone[0x00], zone[0x01], zone[0x04], zone[0x05] = 8, 8, 4, 4
        self._dci_write(_DCI_DSS_CFG, dss)
        self._dci_write(_DCI_ZONE_CFG, zone)
        self._send_offset_data(value)
        self._send_xtalk_data(value)
        self._resolution = value

    @property
    def ranging_frequency(self) -> int:
        """Ranging frequency in Hz. 1-60Hz at 4x4 resolution, 1-15Hz at 8x8."""
        return self._dci_read(_DCI_FREQ, 4)[0x01]

    @ranging_frequency.setter
    def ranging_frequency(self, value: int) -> None:
        max_hz = 15 if self._resolution == RESOLUTION_8X8 else 60
        if not 1 <= value <= max_hz:
            raise ValueError(f"Ranging frequency must be 1-{max_hz}Hz at this resolution")
        self._dci_replace(_DCI_FREQ, 4, 0x01, bytes((value,)))

    @property
    def integration_time(self) -> int:
        """Integration time in milliseconds, 2-1000ms. Only used in autonomous
        ranging mode."""
        return struct.unpack_from("<I", self._dci_read(_DCI_INT_TIME, 20))[0] // 1000

    @integration_time.setter
    def integration_time(self, value: int) -> None:
        if not 2 <= value <= 1000:
            raise ValueError("Integration time must be 2-1000ms")
        self._dci_replace(_DCI_INT_TIME, 20, 0x00, struct.pack("<I", value * 1000))

    @property
    def sharpener_percent(self) -> int:
        """Sharpener strength as a percentage, 0-99. Higher values reduce the
        signal bleeding from close targets into neighboring zones."""
        return self._dci_read(_DCI_SHARPENER, 16)[0x0D] * 100 // 255

    @sharpener_percent.setter
    def sharpener_percent(self, value: int) -> None:
        if not 0 <= value < 100:
            raise ValueError("Sharpener percent must be 0-99")
        self._dci_replace(_DCI_SHARPENER, 16, 0x0D, bytes((value * 255 // 100,)))

    @property
    def target_order(self) -> int:
        """Which target is reported when a zone sees several:
        ``TARGET_ORDER_CLOSEST`` or ``TARGET_ORDER_STRONGEST``."""
        return self._dci_read(_DCI_TGT_ORDER, 4)[0x00]

    @target_order.setter
    def target_order(self, value: int) -> None:
        if value not in {TARGET_ORDER_CLOSEST, TARGET_ORDER_STRONGEST}:
            raise ValueError("Target order must be TARGET_ORDER_CLOSEST or TARGET_ORDER_STRONGEST")
        self._dci_replace(_DCI_TGT_ORDER, 4, 0x00, bytes((value,)))

    @property
    def ranging_mode(self) -> int:
        """``RANGING_MODE_CONTINUOUS`` or ``RANGING_MODE_AUTONOMOUS``. Autonomous mode
        uses `integration_time` and has lower power consumption."""
        if self._dci_read(_DCI_RNG_MODE, 8)[0x01] == 0x03:
            return RANGING_MODE_AUTONOMOUS
        return RANGING_MODE_CONTINUOUS

    @ranging_mode.setter
    def ranging_mode(self, value: int) -> None:
        if value not in {RANGING_MODE_CONTINUOUS, RANGING_MODE_AUTONOMOUS}:
            raise ValueError(
                "Ranging mode must be RANGING_MODE_CONTINUOUS or RANGING_MODE_AUTONOMOUS"
            )
        buf = self._dci_read(_DCI_RNG_MODE, 8)
        if value == RANGING_MODE_CONTINUOUS:
            buf[0x01], buf[0x03] = 0x01, 0x03
            single_range = 0x00
        else:
            buf[0x01], buf[0x03] = 0x03, 0x02
            single_range = 0x01
        self._dci_write(_DCI_RNG_MODE, buf)
        self._dci_write(_DCI_SINGLE_RNG, struct.pack("<I", single_range))

    @property
    def power_mode(self) -> Optional[int]:
        """``POWER_MODE_WAKEUP`` or ``POWER_MODE_SLEEP``. Reads as None when the mode
        is neither, such as while ranging. Stop ranging before sleeping."""
        self._page = 0x00
        mode = self._pwr_ctrl
        self._page = 0x02
        if mode == 0x04:
            return POWER_MODE_WAKEUP
        if mode == 0x02:
            return POWER_MODE_SLEEP
        return None

    @power_mode.setter
    def power_mode(self, value: int) -> None:
        if value not in {POWER_MODE_SLEEP, POWER_MODE_WAKEUP}:
            raise ValueError("Power mode must be POWER_MODE_SLEEP or POWER_MODE_WAKEUP")
        if self.power_mode == value:
            return
        self._page = 0x00
        if value == POWER_MODE_WAKEUP:
            self._pwr_ctrl = 0x04
            self._poll_for_answer(1, 0, _GO2_STATUS0, 0x01, 0x01)
        else:
            self._pwr_ctrl = 0x02
            self._poll_for_answer(1, 0, _GO2_STATUS0, 0x01, 0x00)
        self._page = 0x02

    @property
    def address(self) -> int:
        """The 7-bit I2C address of the sensor. Setting this changes the address
        until the sensor is power cycled."""
        return self.register_accessor.i2c_device.device_address

    @address.setter
    def address(self, value: int) -> None:
        if not 0x08 <= value <= 0x77:
            raise ValueError("I2C address must be 0x08-0x77")
        self._page = 0x00
        self._i2c_addr = value
        self._page = 0x02
        self.register_accessor = I2CRegisterAccessor(
            I2CDevice(self._i2c, value), address_width=2, lsb_first=False
        )
