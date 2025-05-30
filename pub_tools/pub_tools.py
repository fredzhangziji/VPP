"""
此模块包含了所有项目都需要使用的公共工具库函数。
"""

import platform
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import time
import threading

import logging
import pub_tools.logging_config
logger = logging.getLogger(__name__)

def get_system_font_path():
    if platform.system() == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/Library/Fonts/华文黑体.ttf',
            '/Library/Fonts/Songti.ttc',
            '/Library/Fonts/SimHei.ttf',
        ]
    else:  # Linux/Windows
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        ]
    
    # 尝试找到可用的中文字体
    for path in font_paths:
        if os.path.exists(path):
            return path
            
    # 自动搜索系统字体
    font_path = [f for f in fm.findSystemFonts() if any(x in f.lower() for x in ['pingfang', 'heiti', 'song', 'hei', 'noto', 'simhei', 'arialunicode', '华文', 'songti'])]
    if font_path:
        return font_path[0]
    print("未找到合适的中文字体，当前系统可用字体如下：")
    print(fm.findSystemFonts())
    return None

class SnowflakeGenerator:
    EPOCH = 1672531200000  # Custom epoch: 2023-01-01 00:00:00 UTC in milliseconds

    WORKER_ID_BITS = 5
    DATACENTER_ID_BITS = 5
    SEQUENCE_BITS = 12

    MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)
    MAX_DATACENTER_ID = -1 ^ (-1 << DATACENTER_ID_BITS)
    MAX_SEQUENCE = -1 ^ (-1 << SEQUENCE_BITS)

    WORKER_ID_SHIFT = SEQUENCE_BITS
    DATACENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS
    TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATACENTER_ID_BITS

    def __init__(self, datacenter_id: int, worker_id: int):
        if not (0 <= datacenter_id <= self.MAX_DATACENTER_ID):
            raise ValueError(f"Datacenter ID must be between 0 and {self.MAX_DATACENTER_ID}")
        if not (0 <= worker_id <= self.MAX_WORKER_ID):
            raise ValueError(f"Worker ID must be between 0 and {self.MAX_WORKER_ID}")

        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        self._lock = threading.Lock()

    def _current_millis(self) -> int:
        return int(time.time() * 1000)

    def _til_next_millis(self, last_timestamp: int) -> int:
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp

    def generate_id(self) -> int:
        with self._lock:
            timestamp = self._current_millis()

            if timestamp < self.last_timestamp:
                raise SystemError(
                    f"Clock moved backwards. Refusing to generate id for {self.last_timestamp - timestamp} milliseconds"
                )

            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE
                if self.sequence == 0:  # Sequence overflow
                    timestamp = self._til_next_millis(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            new_id = (
                ((timestamp - self.EPOCH) << self.TIMESTAMP_LEFT_SHIFT)
                | (self.datacenter_id << self.DATACENTER_ID_SHIFT)
                | (self.worker_id << self.WORKER_ID_SHIFT)
                | self.sequence
            )
            return new_id

_SNOWFLAKE_DATACENTER_ID = int(os.getenv('SNOWFLAKE_DATACENTER_ID', '0'))
_SNOWFLAKE_WORKER_ID = int(os.getenv('SNOWFLAKE_WORKER_ID', '0'))

try:
    _snowflake_generator_instance = SnowflakeGenerator(
        datacenter_id=_SNOWFLAKE_DATACENTER_ID,
        worker_id=_SNOWFLAKE_WORKER_ID
    )
except ValueError as e:
    print(f"Error initializing SnowflakeGenerator: {e}. Using default IDs (0,0).")
    _snowflake_generator_instance = SnowflakeGenerator(datacenter_id=0, worker_id=0)

def generate_snowflake_id() -> int:
    """
    Generates a unique 64-bit ID using the Snowflake algorithm.
    Ensure SNOWFLAKE_DATACENTER_ID and SNOWFLAKE_WORKER_ID environment
    variables are set appropriately for your distributed instances.
    """
    return _snowflake_generator_instance.generate_id()

if __name__ == '__main__':
    print(f"Initializing SnowflakeGenerator with Datacenter ID: {_SNOWFLAKE_DATACENTER_ID}, Worker ID: {_SNOWFLAKE_WORKER_ID}")
    ids = set()
    for i in range(10):
        new_id = generate_snowflake_id()
        ids.add(new_id)
        print(f"Generated ID: {new_id}")
        if i < 9:
             time.sleep(0.0001)
    print(f"Generated 10 IDs. Number of unique IDs: {len(ids)}")
    print("\nTesting sequence rollover...")
    first_id_in_milli = generate_snowflake_id()
    print(f"First ID in current millisecond: {first_id_in_milli}")
    count_in_milli = 1
    ids_in_milli = {first_id_in_milli}
    start_time = time.time()
    while True:
        current_id = generate_snowflake_id()
        if (current_id >> 22) != (first_id_in_milli >> 22):
            print(f"Timestamp changed. Generated {count_in_milli} IDs in the previous millisecond.")
            break
        ids_in_milli.add(current_id)
        count_in_milli += 1
        if count_in_milli > SnowflakeGenerator.MAX_SEQUENCE + 10:
             print(f"Exceeded MAX_SEQUENCE. Generated {count_in_milli} IDs.")
             break
        if time.time() - start_time > 2:
            print("Sequence rollover test timed out.")
            break
    print(f"Total unique IDs in that millisecond burst: {len(ids_in_milli)}")

