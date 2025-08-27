# 配置文件 - 只包含参数定义

# 全局配置参数
t_s_s1 = 5
MAX_MODULES = 5
MAX_MODULES_STOCK = 4
MIN_MODULES_STOCK = 0
NUM_PASSENGERS = 60
NUM_FREIGHTS = 40
NUM_VEHICLES = 10
NUM_POPSIZE = 3
NUM_TIMESTAMPS = 50
MAX_GENERATIONS = 50
UP_STATIONS = 10
DOWN_STATIONS = 10
TOTAL_STATIONS = UP_STATIONS + DOWN_STATIONS

num_HallOfFame = 5

# 默认需求数据文件路径（已移至demand_loader.py）

parameters = {

    'num_HallOfFame': num_HallOfFame,

    "NUM_VEHICLES": NUM_VEHICLES,
    'max_modules': MAX_MODULES,

    'NUM_POPSIZE': NUM_POPSIZE,

    'MAX_GENERATIONS': MAX_GENERATIONS,

    'max_modules_stock': MAX_MODULES_STOCK,
    'min_modules_stock': MIN_MODULES_STOCK,

    'module_cost': 1.0,

    'passenger_waiting_cost': 10.0,
    'freight_waiting_cost': 5.0,

    'min_headway': 2,
    'max_headway': 20,

    'passenger_per_module': 15,
    'freight_per_module': 10,

    't_s_s1': t_s_s1,
    'travel_time': t_s_s1,

    'num_timestamps': NUM_TIMESTAMPS,

    'up_station_count': UP_STATIONS,
    'down_station_count': DOWN_STATIONS,

    "num_passenger_requests": NUM_PASSENGERS,  # 示例值
    "num_freight_requests": NUM_FREIGHTS,  # 示例值

    'cxpb': 0.7,  # 交叉概率
    'mutpb': 0.6,  # 变异概率

    'C_F': 2.049,
    'C_V': 5.56,
    'alpha': 0.5,

    'beta': 5
}


# 需求数据加载功能已移至 demand_loader.py
# 使用 from demand_loader import load_global_demand_data, load_latest_demand_data