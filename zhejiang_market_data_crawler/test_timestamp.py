#!/usr/bin/env python3
from datetime import datetime, timedelta

def print_timestamps():
    # 测试基准日期
    base_date = datetime.strptime('2025-06-02', '%Y-%m-%d')
    print(f"基准日期: {base_date}")
    
    # 测试原来的逻辑 v1-v5
    print("\n原来的逻辑:")
    for i in range(1, 6):
        hours = ((i - 1) * 15) // 60
        minutes = ((i - 1) * 15) % 60
        timestamp = base_date.replace(hour=hours, minute=minutes)
        print(f"v{i} -> 小时:{hours}, 分钟:{minutes} -> {timestamp}")
    
    # 测试修改后的逻辑 v1-v5
    print("\n修改后的逻辑:")
    for i in range(1, 6):
        hours = (i * 15) // 60
        minutes = (i * 15) % 60
        timestamp = base_date.replace(hour=hours, minute=minutes)
        print(f"v{i} -> 小时:{hours}, 分钟:{minutes} -> {timestamp}")
    
    # 测试v96 (次日00:00)
    i = 96
    hours_old = ((i - 1) * 15) // 60
    minutes_old = ((i - 1) * 15) % 60
    hours_new = (i * 15) // 60
    minutes_new = (i * 15) % 60
    
    print("\nv96的计算结果:")
    print(f"原逻辑 v96 -> 小时:{hours_old}, 分钟:{minutes_old}")
    print(f"新逻辑 v96 -> 小时:{hours_new}, 分钟:{minutes_new}")
    
    # v96特殊处理为次日00:00
    timestamp = base_date + timedelta(days=1)
    timestamp = timestamp.replace(hour=0, minute=0)
    print(f"v96特殊处理 -> {timestamp}")

if __name__ == "__main__":
    print_timestamps() 