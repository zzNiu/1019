# 配置文件 - 只包含参数定义

# 目标函数系数
ω = 0.20
φ = 1 - ω

# 全局配置参数
# 站点间行驶时间
t_s_s1 = 5

# 上行站点数量
UP_STATIONS = 10
DOWN_STATIONS = 10
TOTAL_STATIONS = UP_STATIONS + DOWN_STATIONS

# 公交班次数量
NUM_VEHICLES = 6

# 最大模块数量
MAX_MODULES = 5

# 最小车头时距
MINI_HEADWAY=2
# 最大车头时距
MAX_HEADWAY=20

# 站点存储容量上限
MAX_MODULES_STOCK = 4
MIN_MODULES_STOCK = 0


# 乘客需求数量
NUM_PASSENGERS = 60
# 货物需求数量
NUM_FREIGHTS = 40


# 乘客等待成本系数
CP = 0.11
# 货物等待成本系数
CWf = 0.11
# 固定运输成本
C_F = 2.049
# 可变运输成本系数
C_V = 5.56
# 成本函数指数
alpha = 0.5

beta = 5

# 每个客运模块的容量
kappa_P = 15

# 每个货运模块的容量
kappa_F = 15

# 惩罚成本系数
C_unserved=15

# 时间长度，用于生成需求数据
NUM_TIMESTAMPS = 120


# 遗传算法相关参数
# 种群规模
NUM_POPSIZE = 10
# 迭代代数
MAX_GENERATIONS = 500

# ==================== 新增：提前停止（收敛）参数 ====================
# “耐心”：连续 N 代没有明显改善则停止
EARLY_STOPPING_PATIENCE = 15
# “阈值”：被认为是“明显改善”所需的最小适应度降低值
EARLY_STOPPING_TOLERANCE = 1e-4
# =================================================================

# 名人堂数量
num_HallOfFame = 5
# 交叉概率
CROSSOVER_POSSIBILITY = 0.7
# 变异概率
MUTATED_POSSIBILITY = 0.6

parameters = {

    'num_HallOfFame': num_HallOfFame,

    "NUM_VEHICLES": NUM_VEHICLES,
    'max_modules': MAX_MODULES,

    'NUM_POPSIZE': NUM_POPSIZE,

    'MAX_GENERATIONS': MAX_GENERATIONS,

    'max_modules_stock': MAX_MODULES_STOCK,
    'min_modules_stock': MIN_MODULES_STOCK,

    'passenger_waiting_cost': CP,
    'freight_waiting_cost': CWf,

    'min_headway': MINI_HEADWAY,
    'max_headway': MAX_HEADWAY,

    'passenger_per_module': 15,
    'freight_per_module': 10,

    't_s_s1': t_s_s1,

    'num_timestamps': NUM_TIMESTAMPS,

    'up_station_count': UP_STATIONS,
    'down_station_count': DOWN_STATIONS,

    "num_passenger_requests": NUM_PASSENGERS,  # 示例值
    "num_freight_requests": NUM_FREIGHTS,  # 示例值

    "penalty_cost_per_unit": C_unserved,

    'cxpb': CROSSOVER_POSSIBILITY,  # 交叉概率
    'mutpb': MUTATED_POSSIBILITY,  # 变异概率

    'C_F': C_F,
    'C_V': C_V,
    'alpha': alpha,

    'beta': beta,

    # ==================== 新增：将收敛参数添加到字典 ====================
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'early_stopping_tolerance': EARLY_STOPPING_TOLERANCE
    # =================================================================
}