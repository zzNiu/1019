import json
import pandas as pd

# 导入需求矩阵生成函数
from demand_matrix import initialize_demand_matrices


def load_demand_data(file_path):
    """加载需求数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# def to_dataframe(demand_data):
#     """将需求数据转换为DataFrame格式"""
#     return {
#         'df_passenger_up': pd.DataFrame(demand_data['passenger_demand_up']),
#         'df_passenger_down': pd.DataFrame(demand_data['passenger_demand_down']),
#         'df_freight_up': pd.DataFrame(demand_data['freight_demand_up']),
#         'df_freight_down': pd.DataFrame(demand_data['freight_demand_down'])
#     }


def load_global_demand_data(demand_data_path, parameters):
    """
    加载全局需求数据并生成需求矩阵

    Args:
        demand_data_path: 需求数据文件路径
        parameters: 系统参数字典

    Returns:
        global_demand_data: 全局需求数据字典
        data: 原始需求数据
    """
    # 加载需求原始数据
    data = load_demand_data(demand_data_path)

    # 生成上下行的时空需求矩阵
    NUM_POPSIZE = parameters['NUM_POPSIZE']

    a_p_up, a_f_up = initialize_demand_matrices(
        NUM_POPSIZE,
        range(parameters["up_station_count"]),
        parameters["num_timestamps"],
        data['passenger_demand_up'],
        data['freight_demand_up'],
        parameters['max_headway'],
        parameters['up_station_count'],
        margin=15,
        parameters=parameters
    )

    a_p_down, a_f_down = initialize_demand_matrices(
        NUM_POPSIZE,
        range(parameters["up_station_count"], parameters["up_station_count"] * 2),
        parameters["num_timestamps"],
        data['passenger_demand_down'],
        data['freight_demand_down'],
        parameters['max_headway'],
        parameters['up_station_count'],
        margin=15,
        parameters=parameters
    )

    # 汇总成全局字典
    global_demand_data = {
        "a_matrix_p_up": a_p_up,
        "a_matrix_f_up": a_f_up,
        "a_matrix_p_down": a_p_down,
        "a_matrix_f_down": a_f_down,
        "passenger_demand_up": data["passenger_demand_up"],
        "passenger_demand_down": data["passenger_demand_down"],
        "freight_demand_up": data["freight_demand_up"],
        "freight_demand_down": data["freight_demand_down"]
    }

    return global_demand_data, data