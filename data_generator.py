from config import parameters
import random
import json
import pandas as pd
from datetime import datetime

UP_STATIONS = parameters["up_station_count"]
TOTAL_STATIONS = parameters["up_station_count"] * 2
NUM_TIMESTAMPS = parameters["num_timestamps"]
NUM_PASSENGERS = parameters["num_passenger_requests"]
NUM_FREIGHTS = parameters["num_freight_requests"]

# ============【生成乘客需求数据】============
def generate_passenger_demand(up=True):
    if up:
        return [
            {
                'origin': (origin := random.randint(0, UP_STATIONS - 2)),
                'destination': random.randint(origin + 1, UP_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'num_passengers': random.randint(1, 4)
            } for _ in range(NUM_PASSENGERS)
        ]
    else:
        return [
            {
                'origin': (origin := random.randint(UP_STATIONS, TOTAL_STATIONS - 2)),
                'destination': random.randint(origin + 1, TOTAL_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'num_passengers': random.randint(1, 4)
            } for _ in range(NUM_PASSENGERS)
        ]

# ============【生成货物需求数据】============
def generate_freight_demand(up=True):
    if up:
        return [
            {
                'origin': (origin := random.randint(0, UP_STATIONS - 2)),
                'destination': random.randint(origin + 1, UP_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'volume': random.randint(1, 3)
            } for _ in range(NUM_FREIGHTS)
        ]
    else:
        return [
            {
                'origin': (origin := random.randint(UP_STATIONS, TOTAL_STATIONS - 2)),
                'destination': random.randint(origin + 1, TOTAL_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'volume': random.randint(1, 3)
            } for _ in range(NUM_FREIGHTS)
        ]

# ============【主控逻辑：生成+保存】============
def main():

    now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化

    passenger_demand_up = generate_passenger_demand(up=True)
    passenger_demand_down = generate_passenger_demand(up=False)
    freight_demand_up = generate_freight_demand(up=True)
    freight_demand_down = generate_freight_demand(up=False)

    # 保存 JSON 文件
    json_filename = f'demand_data_{now}.json'
    with open(json_filename, 'w') as f:
        json.dump({
            'passenger_demand_up': passenger_demand_up,
            'passenger_demand_down': passenger_demand_down,
            'freight_demand_up': freight_demand_up,
            'freight_demand_down': freight_demand_down
        }, f, indent=2, ensure_ascii=False)

    # 保存 Excel 文件（分 sheet）
    excel_filename = f'demand_data_{now}.xlsx'
    with pd.ExcelWriter(excel_filename) as writer:
        pd.DataFrame(passenger_demand_up).to_excel(writer, sheet_name='上行乘客需求', index=False)
        pd.DataFrame(passenger_demand_down).to_excel(writer, sheet_name='下行乘客需求', index=False)
        pd.DataFrame(freight_demand_up).to_excel(writer, sheet_name='上行货运需求', index=False)
        pd.DataFrame(freight_demand_down).to_excel(writer, sheet_name='下行货运需求', index=False)

    print("✅ 需求数据生成完毕！")

# ============【运行入口】============
if __name__ == "__main__":
    main()