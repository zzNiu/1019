# from config import parameters
# import random
# import json
# import pandas as pd
# from datetime import datetime
#
# # 从配置文件读取参数
# UP_STATIONS = parameters["up_station_count"]
# TOTAL_STATIONS = parameters["up_station_count"] * 2
# NUM_TIMESTAMPS = parameters["num_timestamps"]
# NUM_PASSENGERS = parameters["num_passenger_requests"]
# NUM_FREIGHTS = parameters["num_freight_requests"]
#
# # ============【生成乘客需求数据 - 采用晚高峰平滑模型】============
# def generate_passenger_demand(up=True):
#     """
#     生成乘客需求数据。
#     到达时间采用梯形平滑高峰模型，模拟晚高峰客流。
#     """
#     # 定义晚高峰时段结构
#     EVENING_RAMP_UP_START = 30
#     EVENING_PEAK_START = 45
#     EVENING_PEAK_END = 74
#     EVENING_RAMP_DOWN_END = 89
#
#     def _generate_single_arrival_time():
#         """内部函数：根据模型生成单个到达时间点"""
#         # 假设70%的客流与高峰现象相关
#         if random.random() < 0.7:
#             # 在高峰相关的客流中，按权重分配到不同阶段
#             phase_choice = random.choices(
#                 ['ramp_up', 'peak', 'ramp_down'], weights=[0.25, 0.5, 0.25], k=1
#             )[0]
#
#             if phase_choice == 'ramp_up':
#                 # --- 上升期 ---
#                 mean = EVENING_PEAK_START
#                 std_dev = (EVENING_PEAK_START - EVENING_RAMP_UP_START) / 3
#                 arrival_time = random.normalvariate(mean, std_dev)
#                 arrival_time = mean - abs(arrival_time - mean)
#                 return int(round(max(EVENING_RAMP_UP_START, arrival_time)))
#
#             elif phase_choice == 'peak':
#                 # --- 高峰平台期 ---
#                 return random.randint(EVENING_PEAK_START, EVENING_PEAK_END)
#
#             else:  # ramp_down
#                 # --- 下降期 ---
#                 mean = EVENING_PEAK_END
#                 std_dev = (EVENING_RAMP_DOWN_END - EVENING_PEAK_END) / 3
#                 arrival_time = random.normalvariate(mean, std_dev)
#                 arrival_time = mean + abs(arrival_time - mean)
#                 return int(round(min(EVENING_RAMP_DOWN_END, arrival_time)))
#         else:
#             # --- 平峰期 ---
#             while True:
#                 arrival_time = random.randint(0, NUM_TIMESTAMPS - 1)
#                 if not (EVENING_RAMP_UP_START <= arrival_time <= EVENING_RAMP_DOWN_END):
#                     return arrival_time
#
#     # 根据上述模型，生成指定数量的乘客请求列表
#     passenger_requests = []
#     for _ in range(NUM_PASSENGERS):
#         arrival_time = _generate_single_arrival_time()
#
#         if up:
#             origin = random.randint(0, UP_STATIONS - 2)
#             destination = random.randint(origin + 1, UP_STATIONS - 1)
#         else:
#             origin = random.randint(UP_STATIONS, TOTAL_STATIONS - 2)
#             destination = random.randint(origin + 1, TOTAL_STATIONS - 1)
#
#         passenger_requests.append({
#             'origin': origin,
#             'destination': destination,
#             'arrival_time': arrival_time,
#             'num_passengers': random.randint(1, 5)  # 保持原设定，每次请求1位乘客
#         })
#     return passenger_requests
#
# # ============【生成货物需求数据 - 保持完全随机】============
# def generate_freight_demand(up=True):
#     """
#     生成货物需求数据。
#     到达时间采用完全随机模式。
#     """
#     if up:
#         return [
#             {
#                 'origin': (origin := random.randint(0, UP_STATIONS - 2)),
#                 'destination': random.randint(origin + 1, UP_STATIONS - 1),
#                 'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
#                 'volume': random.randint(1, 1)
#             } for _ in range(NUM_FREIGHTS)
#         ]
#     else:
#         return [
#             {
#                 'origin': (origin := random.randint(UP_STATIONS, TOTAL_STATIONS - 2)),
#                 'destination': random.randint(origin + 1, TOTAL_STATIONS - 1),
#                 'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
#                 'volume': random.randint(1, 4)
#             } for _ in range(NUM_FREIGHTS)
#         ]
#
# # ============【主控逻辑：生成+保存】============
# def main():
#     now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化
#
#     print("🚀 开始生成需求数据...")
#     passenger_demand_up = generate_passenger_demand(up=True)
#     passenger_demand_down = generate_passenger_demand(up=False)
#     freight_demand_up = generate_freight_demand(up=True)
#     freight_demand_down = generate_freight_demand(up=False)
#     print("...数据生成完成。")
#
#     # 保存 JSON 文件
#     json_filename = f'120_需求分布_demand_data_{now}.json'
#     print(f"📄 正在保存到 JSON 文件: {json_filename}")
#     with open(json_filename, 'w') as f:
#         json.dump({
#             'passenger_demand_up': passenger_demand_up,
#             'passenger_demand_down': passenger_demand_down,
#             'freight_demand_up': freight_demand_up,
#             'freight_demand_down': freight_demand_down
#         }, f, indent=2, ensure_ascii=False)
#
#     # 保存 Excel 文件（分 sheet）
#     excel_filename = f'demand_data_{now}.xlsx'
#     print(f"📊 正在保存到 Excel 文件: {excel_filename}")
#     with pd.ExcelWriter(excel_filename) as writer:
#         pd.DataFrame(passenger_demand_up).to_excel(writer, sheet_name='上行乘客需求', index=False)
#         pd.DataFrame(passenger_demand_down).to_excel(writer, sheet_name='下行乘客需求', index=False)
#         pd.DataFrame(freight_demand_up).to_excel(writer, sheet_name='上行货运需求', index=False)
#         pd.DataFrame(freight_demand_down).to_excel(writer, sheet_name='下行货运需求', index=False)
#
#     print("\n✅ 需求数据生成完毕！")
#
# # ============【运行入口】============
# if __name__ == "__main__":
#     # 确保您的 config.py 文件中 num_timestamps 设置为 120
#     if NUM_TIMESTAMPS != 120:
#         print(f"⚠️ 警告: 为了匹配晚高峰模型，建议将 config.py 中的 num_timestamps 设置为 120。当前值为: {NUM_TIMESTAMPS}")
#     main()

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
                'num_passengers': random.randint(1, 5)
            } for _ in range(NUM_PASSENGERS)
        ]
    else:
        return [
            {
                'origin': (origin := random.randint(UP_STATIONS, TOTAL_STATIONS - 2)),
                'destination': random.randint(origin + 1, TOTAL_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'num_passengers': random.randint(1, 5)
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
                'volume': random.randint(1, 4)
            } for _ in range(NUM_FREIGHTS)
        ]
    else:
        return [
            {
                'origin': (origin := random.randint(UP_STATIONS, TOTAL_STATIONS - 2)),
                'destination': random.randint(origin + 1, TOTAL_STATIONS - 1),
                'arrival_time': random.randint(0, NUM_TIMESTAMPS - 1),
                'volume': random.randint(1, 4)
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
    json_filename = f'需求数据_demand_data_{now}.json'
    with open(json_filename, 'w') as f:
        json.dump({
            'passenger_demand_up': passenger_demand_up,
            'passenger_demand_down': passenger_demand_down,
            'freight_demand_up': freight_demand_up,
            'freight_demand_down': freight_demand_down
        }, f, indent=2, ensure_ascii=False)

    # 保存 Excel 文件（分 sheet）
    excel_filename = f'需求数据_demand_data_{now}.xlsx'
    with pd.ExcelWriter(excel_filename) as writer:
        pd.DataFrame(passenger_demand_up).to_excel(writer, sheet_name='上行乘客需求', index=False)
        pd.DataFrame(passenger_demand_down).to_excel(writer, sheet_name='下行乘客需求', index=False)
        pd.DataFrame(freight_demand_up).to_excel(writer, sheet_name='上行货运需求', index=False)
        pd.DataFrame(freight_demand_down).to_excel(writer, sheet_name='下行货运需求', index=False)

    print("✅ 需求数据生成完毕！")

# ============【运行入口】============
if __name__ == "__main__":
    main()