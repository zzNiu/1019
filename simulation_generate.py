# simulation_generate.py

import copy
import math
import pandas as pd
import random
from typing import Dict, List, Tuple

from config import ω, φ
from df_schedule_construct import reconstruct_schedule_dataframe


class IntegratedBusModuleSystem:
    """集成化公交车模块调度系统 - 结合仿真和递推关系"""

    def __init__(self, alpha=0, beta=5, C_p=10, C_f=8):
        """
        初始化系统参数

        Args:
            alpha: 模块连接数量下限
            beta: 模块连接数量上限
            C_p: 乘客模块容量
            C_f: 货运模块容量
        """
        self.alpha = alpha
        self.beta = beta
        self.C_p = C_p
        self.C_f = C_f


    def calculate_station_module_requirements(self, n: int, k: int,
                                              p_n_k: int, f_n_k: int,
                                              store_modules: int,
                                              onboard_pass_before: int, onboard_cargo_before: int,
                                              onboard_pass_after: int, onboard_cargo_after: int,
                                              off_pass: int, off_cargo: int,
                                              waiting_pass: int, waiting_cargo: int) -> Dict:
        """
        计算第n班车在第k个站点的模块需求和调整范围

        思路1实现：基于下车后在车数量 + 等待需求来计算模块调整需求
        这样可以确保模块调整后能够容纳所有可能上车的乘客/货物

        Returns:
            包含模块需求分析和调整范围的完整字典
        """

        # print('开始计算相关的变化范围')

        # 1. 下车后在车数量
        remaining_pass = onboard_pass_before - off_pass
        remaining_cargo = onboard_cargo_before - off_cargo

        # print('remaining_pass:', remaining_pass)
        # print('remaining_cargo:', remaining_cargo)

        if remaining_pass != onboard_pass_after or remaining_cargo != onboard_cargo_after:
            print('哎呦卧槽 出错了呀 这怎么能算不对呢')

        # 2. 下车后在车占用模块数量（最小需求）
        U_pass = math.ceil(remaining_pass / self.C_p) if remaining_pass > 0 else 0
        U_cargo = math.ceil(remaining_cargo / self.C_f) if remaining_cargo > 0 else 0
        U_total = U_pass + U_cargo

        # print('最小需求')
        # print('U_pass:', U_pass)
        # print('U_cargo:', U_cargo)

        # 3. 考虑等待乘客/货物的总需求模块数（优化后的需求计算）
        # 这是模块调整的目标：既要满足在车需求，又要尽可能满足等待需求
        total_pass_need = remaining_pass + waiting_pass
        total_cargo_need = remaining_cargo + waiting_cargo

        # print('总的需求数量')
        # print('total_pass_need:', total_pass_need)
        # print('total_cargo_need:', total_cargo_need)

        T_pass = math.ceil(total_pass_need / self.C_p) if total_pass_need > 0 else 0
        T_cargo = math.ceil(total_cargo_need / self.C_f) if total_cargo_need > 0 else 0
        T_total = T_pass + T_cargo

        # print('需求模块数量')
        # print('T_pass:', T_pass)
        # print('T_cargo:', T_cargo)

        # 4. 当前可用模块总数
        available_modules = p_n_k + f_n_k + store_modules

        # print('可用模块数量')
        # print('p_n_k:', p_n_k, 'f_n_k:', f_n_k, 'store_modules:', store_modules)

        if store_modules < 0:
            print('有问题了')

        # 5. 总模块数调整范围（思路1：优先考虑总需求）
        # 最小值：至少满足在车需求
        total_min = U_total
        # 最大值：不超过可用模块和系统上限，但优先考虑总需求
        total_max = min(available_modules, self.beta)  # 允许略微超过理论需求以提供缓冲
        # total_max = min(available_modules, self.beta, T_total + 2)  # 允许略微超过理论需求以提供缓冲

        # print('最少、最大需求模块数量')
        # print('total_min:', total_min)
        # print('total_max:', total_max)

        # 6. 模块增量范围
        current_total = p_n_k + f_n_k

        delta_min = total_min - current_total
        delta_max = total_max - current_total

        # 7. 乘客模块调整范围（思路1：基于总需求优化）
        # 最小值：至少满足在车乘客需求 p_min = max(0, U_pass)
        p_min = U_pass

        # 8. 货物模块调整范围（思路1：基于总需求优化）
        # 最小值：至少满足在车货物需求 f_min = max(0, U_cargo)
        f_min = U_cargo

        return {
            'station_info': {
                'bus_id': n,
                'station_id': k,
                'current_p_modules': p_n_k,
                'current_f_modules': f_n_k,
                'store_modules': store_modules,
                'current_total': current_total
            },
            'passenger_analysis': {
                'onboard_before': onboard_pass_before,
                'alighting': off_pass,
                'waiting': waiting_pass,
                'remaining_onboard': remaining_pass,
                'total_demand': total_pass_need,
                'min_modules_needed': U_pass,
                'optimal_modules': T_pass,
                # 'utilization_rate': p_utilization,
                # 'coverage_rate': p_coverage
            },
            'freight_analysis': {
                'onboard_before': onboard_cargo_before,
                'alighting': off_cargo,
                'waiting': waiting_cargo,
                'remaining_onboard': remaining_cargo,
                'total_demand': total_cargo_need,
                'min_modules_needed': U_cargo,
                'optimal_modules': T_cargo,
                # 'utilization_rate': f_utilization,
                # 'coverage_rate': f_coverage
            },
            'module_constraints': {
                'total_available': available_modules,
                'total_min_required': U_total,
                'total_optimal': T_total,
                'system_min_limit': self.alpha,
                'system_max_limit': self.beta,
                'total_max': total_max,
                'feasible_total_range': (total_min, total_max),
                # 'delta_range': (delta_min, delta_max)
            },
            'add':{
                'passenger_modules_min': p_min,
                'freight_modules_min': f_min
            }
        }

    def generate_feasible_module_allocation(self, module_analysis: Dict) -> Tuple[int, int, int, int, dict]:
        """
        基于分析结果生成可行的模块分配方案

        Args:
            module_analysis: calculate_station_module_requirements的返回结果

        Returns:
            (下一站乘客模块数, 下一站货物模块数, 乘客模块变化量, 货物模块变化量)
        """
        # print('基于分析结果生成可行的模块分配方案')
        p_min = module_analysis['add']['passenger_modules_min']
        # print('p_min:', p_min)
        f_min = module_analysis['add']['freight_modules_min']
        # print('f_min:', f_min)
        total_max = module_analysis['module_constraints']['total_max']
        # print('total_max:', total_max)
        p_n_k = module_analysis['station_info']['current_p_modules']
        # print('p_n_k:', p_n_k)
        f_n_k = module_analysis['station_info']['current_f_modules']
        # print('f_n_k:', f_n_k)

        # 【新增】从分析结果中获取当前站点的库存，以供weizhi函数使用
        store_modules = module_analysis['station_info']['store_modules']

        # 乘客模块
        delta_p_min =  p_min - p_n_k
        # delta_p_min = p_n_k - p_min
        delta_p_max = total_max - p_n_k - f_min

        # print('delta_p_min = p_min - p_n_k:', delta_p_min)
        # print('delta_p_max = total_max - p_n_k - f_min:', delta_p_max)

        delta_p_range = range(delta_p_min, delta_p_max + 1)
        delta_p = random.randint(delta_p_min, delta_p_max)
        # delta_p = random.randint(p_min - p_n_k, total_max - p_n_k - f_min)

        # print('delta_p:', delta_p)

        # 货物模块
        delta_f_min =  f_min - f_n_k
        delta_f_max = total_max - f_n_k - (p_n_k + delta_p)

        # print('delta_f_min = f_min - f_n_k:', delta_f_min)
        # print('delta_f_max = total_max - f_n_k - p_n_k - delta_p:', delta_f_max)

        delta_f_range = range(delta_f_min, delta_f_max + 1)

        # print('delta_p_range:', delta_p_range)
        # print('delta_f_range:', delta_f_range)

        delta_f = random.randint(delta_f_min, delta_f_max)

        while True:

            # print("qian", p_n_k, f_n_k, delta_p, delta_f)
            # print('尝试')
            if weizhi(f_n_k, p_n_k, delta_f, delta_p, store_modules):
                break
            else:
                print('p_min:', p_min, 'p_n_k:', p_n_k)
                delta_p_min = p_min - p_n_k
                # delta_p_min = p_n_k - p_min
                print('total_max:', total_max, 'p_n_k:', p_n_k, 'f_min:', f_min)
                delta_p_max = total_max - p_n_k - f_min

                delta_p = random.randint(delta_p_min, delta_p_max)
                print('delta_p:', delta_p)

                print('f_min:', f_min, 'f_n_k:', f_n_k)
                delta_f_min = f_min - f_n_k
                print('total_max:', total_max, 'f_n_k:', f_n_k, 'p_n_k:', p_n_k, 'delta_p:', delta_p)
                delta_f_max = total_max - f_n_k - (p_n_k + delta_p)

                delta_f = random.randint(delta_f_min, delta_f_max)
                print('delta_f:', delta_f)

        # print("hou", p_n_k, f_n_k, delta_p, delta_f)
        #
        # print('车辆编号:',module_analysis['station_info']['bus_id'])
        # print('站点编号:',module_analysis['station_info']['station_id'])

        p_n_k_1 = p_n_k + delta_p
        f_n_k_1 = f_n_k + delta_f

        adjustment_ranges = {
            'passenger_modules': {
                'min': p_min,
                'current': p_n_k,
                'delta_range': delta_p_range
            },
            'freight_modules': {
                'min': f_min,
                'current': f_n_k,
                'delta_range': delta_f_range  # 这里没有考虑乘客模块的变化量，在后面实际变化的时候把变化量考虑进去
            }
        }

        module_analysis['adjustment_ranges'] = adjustment_ranges

        return p_n_k_1, f_n_k_1, delta_p, delta_f, module_analysis

# 判断变化后的模块数量是否符合实际情况
def weizhi(current_f_modules, current_p_modules, delta_f, delta_p, store_modules):

    new_store_modules = store_modules - (delta_p + delta_f)

    # if current_f_modules + delta_f !=0 or current_p_modules + delta_p != 0 and new_store_modules >= 0 and new_store_modules <= 4:
    if (current_f_modules + delta_f != 0 or current_p_modules + delta_p != 0) and 0 <= new_store_modules <= 4:
        return True
    else:
        return False


def simulate_with_integrated_module_system(individual, parameters, global_demand_data,
                                           passenger_demand_up, passenger_demand_down,
                                           freight_demand_up, freight_demand_down):
    """
    集成化仿真函数 - 结合递推关系和仿真验证

    Returns:
        vehicle_schedule: 车辆调度方案
        total_cost: 总成本
        remaining_passengers: 剩余乘客
        remaining_freights: 剩余货物
        failure_records: 失败记录
        df_enriched: 丰富的数据框
        module_analysis_records: 模块分析记录
    """

    # 初始化集成系统
    module_system = IntegratedBusModuleSystem(
        alpha=parameters['alpha'],
        beta=parameters['beta'],
        C_p=parameters['passenger_per_module'],
        C_f=parameters['freight_per_module']
    )

    # 初始化变量
    df_enriched = []
    failure_records = []
    module_analysis_records = []

    # 获取需求矩阵的副本
    a_matrix_p_up = copy.deepcopy(global_demand_data["a_matrix_p_up"])
    a_matrix_f_up = copy.deepcopy(global_demand_data["a_matrix_f_up"])
    a_matrix_p_down = copy.deepcopy(global_demand_data["a_matrix_p_down"])
    a_matrix_f_down = copy.deepcopy(global_demand_data["a_matrix_f_down"])

    # 合并调度信息
    vehicle_dispatch_up = individual["up"]["vehicle_dispatch"]
    vehicle_dispatch_down = individual["down"]["vehicle_dispatch"]
    combined_vehicle_dispatch = {**vehicle_dispatch_up, **vehicle_dispatch_down}

    # 获取车辆信息
    all_vehicles, vehicle_schedule, _ = collect_vehicle_info(
        individual, parameters,
        passenger_demand_up, passenger_demand_down,
        freight_demand_up, freight_demand_down
    )

    # 初始化站点模块存储
    station_module_stock = {sid: {"modules": 0} for sid in range(parameters["up_station_count"] * 2)}

    infeasible = False

    total_cost = 0
    total_passenger_waiting_time_cost = 0
    total_freight_waiting_time_cost = 0

    # ==================== 新增逻辑：开始 ====================
    total_served_passengers = 0
    total_served_freight = 0
    # ==================== 新增逻辑：结束 ====================

    max_simulation_time = 0

    # 主仿真循环
    for direction in ["up", "down"]:
        # print(f'=== 开始仿真方向: {direction} ===')

        # 匹配需求数据
        if direction == "up":
            num_stations = parameters["up_station_count"]
            a_matrix_p = a_matrix_p_up
            a_matrix_f = a_matrix_f_up
        else:
            num_stations = parameters["up_station_count"] + parameters["up_station_count"]
            a_matrix_p = a_matrix_p_down
            a_matrix_f = a_matrix_f_down

        # 遍历车辆
        for vehicle in all_vehicles:

            if vehicle["direction"] != direction:
                continue  # continue中断当前循环 break中断整个循环

            vid = vehicle["global_vid"]
            offset = vehicle["station_offset"]
            arrival_time = vehicle["dispatch"]["arrival_time"]

            # 更新最大仿真时间
            last_departure_time = arrival_time + (vehicle["num_stations"] - 1) * parameters["t_s_s1"]
            max_simulation_time = max(max_simulation_time, last_departure_time)

            # 初始化车辆状态
            onboard_passengers = {}
            onboard_freight = {}

            # 当前模块数量（获得初始模块数量）
            current_p_modules = individual[direction]["initial_allocation"][vid]["passenger_modules"]
            current_f_modules = individual[direction]["initial_allocation"][vid]["freight_modules"]

            # 初始化下一个站点的模块数量
            next_p = 0
            next_f = 0

            for sid in range(vehicle["num_stations"]):
                station_id = sid + offset

                # 更新到达时间
                if sid > 0:
                    arrival_time += parameters["t_s_s1"]
                    current_p_modules = next_p
                    current_f_modules = next_f

                # print(f"  车辆{vid} 到达站点{station_id} 时间{arrival_time}")

                # 获取站点存储数量
                store_modules = station_module_stock[station_id]["modules"]

                # 计算在车数量（下车前）
                onboard_p_before = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_before = sum(sum(f.values()) for f in onboard_freight.values())

                # 计算下车数量
                alighted_p = sum(sum(p_dict.values()) for dest, p_dict in onboard_passengers.items() if dest == station_id)
                alighted_f = sum(sum(f_dict.values()) for dest, f_dict in onboard_freight.items() if dest == station_id)

                # 执行下车操作
                onboard_passengers.pop(station_id, None)
                onboard_freight.pop(station_id, None)

                # 计算下车后在车数量
                onboard_p_after = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_after = sum(sum(f.values()) for f in onboard_freight.values())

                # 计算等待需求
                waiting_p = 0
                waiting_f = 0

                for s_dest in range(station_id + 1, num_stations):
                    for t in range(arrival_time + 1):
                        if (station_id in a_matrix_p and s_dest in a_matrix_p[station_id]
                                and t in a_matrix_p[station_id][s_dest]):
                            waiting_p += a_matrix_p[station_id][s_dest][t]

                        if (station_id in a_matrix_f and s_dest in a_matrix_f[station_id]
                                and t in a_matrix_f[station_id][s_dest]):
                            waiting_f += a_matrix_f[station_id][s_dest][t]

                # print(f"    下车前: 乘客{onboard_p_before}, 货物{onboard_f_before}")
                # print(f"    下车: 乘客{alighted_p}, 货物{alighted_f}")
                # print(f"    下车后: 乘客{onboard_p_after}, 货物{onboard_f_after}")
                # print(f"    等待: 乘客{waiting_p}, 货物{waiting_f}")

                # 1. 计算模块需求和调整范围(先统计相关的数据)
                module_analysis = module_system.calculate_station_module_requirements(
                    n=vid, k=station_id,
                    p_n_k=current_p_modules,
                    f_n_k=current_f_modules,
                    store_modules=store_modules,
                    onboard_pass_before=onboard_p_before,
                    onboard_cargo_before=onboard_f_before,
                    off_pass=alighted_p,
                    off_cargo=alighted_f,
                    onboard_pass_after=onboard_p_after,
                    onboard_cargo_after=onboard_f_after,
                    waiting_pass=waiting_p,
                    waiting_cargo=waiting_f
                )

                # 2. 计算模块调整方案，输出分析结果 执行模块调整（在上车前）
                # print('执行之前')
                # print('-------')
                # # print(module_analysis)
                # print('-------')
                adjusted_p_modules, adjusted_f_modules, delta_p, delta_f, enhanced_module_analysis = module_system.generate_feasible_module_allocation(module_analysis)
                # print('执行之后')

                # print('打印调整结果和调整方案')
                # print('adjusted_p_modules', adjusted_p_modules, 'adjusted_f_modules', adjusted_f_modules, 'delta_p', delta_p, 'delta_f', delta_f)

                # print(f"    模块调整: 乘客 {current_p_modules}->{adjusted_p_modules}({delta_p:+d}), 货物 {current_f_modules}->{adjusted_f_modules}({delta_f:+d})")

                # 3. 更新调整后的容量
                adjusted_p_capacity = adjusted_p_modules * parameters["passenger_per_module"]
                adjusted_f_capacity = adjusted_f_modules * parameters["freight_per_module"]

                # 4. 验证调整后容量是否满足在车需求
                if onboard_p_after > adjusted_p_capacity:
                    print(f"❌ 模块调整后乘客容量仍然超限: 车辆{vid} 站点{station_id}")
                    print(f"   在车需求: {onboard_p_after}, 调整后容量: {adjusted_p_capacity}")

                    failure_records.append({
                        "station_id": station_id,
                        "vehicle_id": vid,
                        "timestamp": arrival_time,
                        "type": "passenger_capacity_exceeded_after_adjustment",
                        "required": onboard_p_after,
                        "available": adjusted_p_capacity,
                        "shortage": onboard_p_after - adjusted_p_capacity
                    })
                    infeasible = True
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), [], []

                if onboard_f_after > adjusted_f_capacity:
                    print(f"❌ 模块调整后货物容量仍然超限: 车辆{vid} 站点{station_id}")
                    print(f"   在车需求: {onboard_f_after}, 调整后容量: {adjusted_f_capacity}")

                    failure_records.append({
                        "station_id": station_id,
                        "vehicle_id": vid,
                        "timestamp": arrival_time,
                        "type": "freight_capacity_exceeded_after_adjustment",
                        "required": onboard_f_after,
                        "available": adjusted_f_capacity,
                        "shortage": onboard_f_after - adjusted_f_capacity
                    })
                    infeasible = True
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), [], []

                # 5. 更新站点模块库存（模块调整的影响）
                station_module_stock_before = station_module_stock[station_id]["modules"]
                station_module_stock[station_id]["modules"] -= (delta_p + delta_f)
                station_module_stock_after = station_module_stock[station_id]["modules"]
                # print('station_module_stock_before:', station_module_stock_before)
                # print('delta_p:', delta_p, 'delta_f:', delta_f)
                # print('station_module_stock_after:', station_module_stock_after)
                if station_module_stock_after < 0 :
                    print('-逆天了 竟然小于0')

                # 6. 检查站点库存约束 (已注释掉)
                # if (station_module_stock_after > parameters["max_modules_stock"] or
                #         station_module_stock_after < parameters["min_modules_stock"]):
                #     print(f"❌ 站点库存超限: 站点{station_id} 库存{station_module_stock_after}")
                #     failure_records.append({
                #         "station_id": station_id,
                #         "type": "station_stock_violation",
                #         "stock_level": station_module_stock_after
                #     })
                #     infeasible = True
                #     return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), []

                # === 思路1核心：基于调整后的容量进行上车操作 ===
                # print('开始上车')

                # 7. 乘客上车逻辑（基于调整后的容量）
                available_p_capacity = adjusted_p_capacity - onboard_p_after
                boarded_p = 0
                served_passenger_waiting_time = 0

                # print(f"    调整后乘客可用容量: {available_p_capacity}")

                if available_p_capacity > 0:
                    for s_dest in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_p >= available_p_capacity:
                                break

                            if (station_id in a_matrix_p and s_dest in a_matrix_p[station_id] and
                                    t in a_matrix_p[station_id][s_dest]):

                                demand_p = a_matrix_p[station_id][s_dest][t]
                                board_now_p = min(demand_p, available_p_capacity - boarded_p)

                                if board_now_p > 0:
                                    waiting_time = arrival_time - t
                                    served_passenger_waiting_time += board_now_p * waiting_time

                                    if s_dest not in onboard_passengers:
                                        onboard_passengers[s_dest] = {}
                                    if arrival_time not in onboard_passengers[s_dest]:
                                        onboard_passengers[s_dest][arrival_time] = 0
                                    onboard_passengers[s_dest][arrival_time] += board_now_p

                                    boarded_p += board_now_p
                                    a_matrix_p[station_id][s_dest][t] -= board_now_p

                # 8. 货物上车逻辑（基于调整后的容量）
                available_f_capacity = adjusted_f_capacity - onboard_f_after
                boarded_f = 0
                served_freight_waiting_time = 0

                if available_f_capacity > 0:
                    for s_dest in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_f >= available_f_capacity:
                                break

                            if (station_id in a_matrix_f and s_dest in a_matrix_f[station_id] and
                                    t in a_matrix_f[station_id][s_dest]):

                                demand_f = a_matrix_f[station_id][s_dest][t]
                                board_now_f = min(demand_f, available_f_capacity - boarded_f)

                                if board_now_f > 0:
                                    waiting_time = arrival_time - t
                                    served_freight_waiting_time += board_now_f * waiting_time

                                    if s_dest not in onboard_freight:
                                        onboard_freight[s_dest] = {}
                                    if arrival_time not in onboard_freight[s_dest]:
                                        onboard_freight[s_dest][arrival_time] = 0
                                    onboard_freight[s_dest][arrival_time] += board_now_f

                                    boarded_f += board_now_f
                                    a_matrix_f[station_id][s_dest][t] -= board_now_f

                # print(f"    上车: 乘客{boarded_p}, 货物{boarded_f}")

                # ==================== 新增逻辑：开始 ====================
                # 累加本站上车数量到总服务数量
                total_served_passengers += boarded_p
                total_served_freight += boarded_f
                # ==================== 新增逻辑：结束 ====================

                # 9. 累计等待时间成本
                total_passenger_waiting_time_cost += served_passenger_waiting_time
                total_freight_waiting_time_cost += served_freight_waiting_time

                # 10. 为下一站准备模块配置（思路1：当前站已完成调整）
                if sid < vehicle["num_stations"] - 1:  # 不是最后一站
                    # 当前站的调整结果就是下一站的起始配置
                    next_p = adjusted_p_modules
                    next_f = adjusted_f_modules

                    # 记录建议的下一站配置
                    enhanced_module_analysis['suggested_next_allocation'] = {
                        'passenger_modules': next_p,
                        'freight_modules': next_f,
                        'total_modules': next_p + next_f,
                        'delta_p': delta_p,
                        'delta_f': delta_f
                    }

                # print('module_analysis:', module_analysis)

                # 11. 记录模块分析结果
                module_analysis_records.append({
                    'timestamp': arrival_time,
                    'vehicle_id': vid,
                    'station_id': station_id,
                    'direction': direction,
                    'analysis': enhanced_module_analysis
                })

                # print('记录详细信息')

                # 12. 记录详细信息（思路1：更新记录内容）
                df_enriched.append({
                    "车辆ID": vid,
                    "站点ID": station_id,
                    "方向": direction,
                    "到达时间": arrival_time,

                    # 模块信息（调整前后）
                    "调整前乘客模块": current_p_modules,
                    "调整前货物模块": current_f_modules,
                    "调整前总模块数": current_p_modules + current_f_modules,
                    "调整后乘客模块": adjusted_p_modules,
                    "调整后货物模块": adjusted_f_modules,
                    "调整后总模块数": adjusted_p_modules + adjusted_f_modules,
                    "模块增量_乘客": delta_p,
                    "模块增量_货物": delta_f,

                    # 乘客/货物流动信息
                    "下车前在车乘客": onboard_p_before,
                    "下车前在车货物": onboard_f_before,
                    "下车乘客": alighted_p,
                    "下车货物": alighted_f,
                    "下车后在车乘客": onboard_p_after,
                    "下车后在车货物": onboard_f_after,

                    "等待乘客需求": waiting_p,
                    "等待货物需求": waiting_f,
                    "上车乘客": boarded_p,
                    "上车货物": boarded_f,


                    # ==================== 代码修改处 开始 ====================
                    "上车后在车乘客数量": onboard_p_after + boarded_p,
                    "上车后在车货物数量": onboard_f_after + boarded_f,
                    # ==================== 代码修改处 结束 ====================

                    # 容量信息（调整后）
                    "调整后乘客总容量": adjusted_p_capacity,
                    "调整后货物总容量": adjusted_f_capacity,
                    "调整后乘客可用容量": available_p_capacity,
                    "调整后货物可用容量": available_f_capacity,

                    # 成本信息
                    "乘客等待时间成本": served_passenger_waiting_time,
                    "货物等待时间成本": served_freight_waiting_time,

                    # 库存信息
                    "站点进站前库存": station_module_stock_before,
                    "站点出站后库存": station_module_stock_after,

                    # 模块分析结果
                    "最少需要乘客模块": module_analysis['passenger_analysis']['min_modules_needed'],
                    "最优乘客模块": module_analysis['passenger_analysis']['optimal_modules'],
                    "最少需要货物模块": module_analysis['freight_analysis']['min_modules_needed'],
                    "最优货物模块": module_analysis['freight_analysis']['optimal_modules'],
                    "乘客模块调整范围": f"{module_analysis['adjustment_ranges']['passenger_modules']['delta_range']}",
                    "货物模块调整范围": f"{module_analysis['adjustment_ranges']['freight_modules']['delta_range']}",
                    "总模块可行范围": f"{module_analysis['module_constraints']['feasible_total_range']}",
                })

    # 如果不可行，提前返回
    if infeasible:
        print("❌ 方案不可行")
        return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), []

    # unserved_passenger_waiting_cost = 0
    # unserved_freight_waiting_cost = 0
    #
    # for a_matrix_p in [a_matrix_p_up, a_matrix_p_down]:
    #     for s in a_matrix_p:
    #         for s_dest in a_matrix_p[s]:
    #             for t in a_matrix_p[s][s_dest]:
    #                 remaining_p = a_matrix_p[s][s_dest][t]
    #                 if remaining_p > 0:
    #                     waiting_time = max_simulation_time - t
    #                     unserved_passenger_waiting_cost += remaining_p * waiting_time
    #
    # for a_matrix_f in [a_matrix_f_up, a_matrix_f_down]:
    #     for s in a_matrix_f:
    #         for s_dest in a_matrix_f[s]:
    #             for t in a_matrix_f[s][s_dest]:
    #                 remaining_f = a_matrix_f[s][s_dest][t]
    #                 if remaining_f > 0:
    #                     waiting_time = max_simulation_time - t
    #                     unserved_freight_waiting_cost += remaining_f * waiting_time

    # # 计算总成本
    # total_passenger_waiting_time_cost += unserved_passenger_waiting_cost
    # total_freight_waiting_time_cost += unserved_freight_waiting_cost

    # unserved_penalty_cost = (unserved_passenger_waiting_cost + unserved_freight_waiting_cost) * parameters['penalty_cost_per_unit']

    passenger_waiting_cost = total_passenger_waiting_time_cost * parameters["passenger_waiting_cost"]
    freight_waiting_cost = total_freight_waiting_time_cost * parameters["freight_waiting_cost"]

    df_enriched = pd.DataFrame(df_enriched)  # 确保 df_enriched 是一个DataFrame

    # 计算运营成本 (基于每段路程的实际模块配置)
    print('计算运营成本 (基于每段路程的实际模块配置)')
    modular_bus_cost = 0
    # 车辆在站点 k 调整模块后，以该配置运行到站点 k+1
    # 只需要考虑不是终点的站点，即 sid < vehicle["num_stations"] - 1 的情况

    # 过滤出非终点站点的记录，即对应运行路段的起点
    df_operating_segments = df_enriched.copy()

    # 获取每个车辆的站点总数
    vehicle_station_counts = {v["global_vid"]: v["num_stations"] for v in all_vehicles}

    # 为df_operating_segments添加车辆总站点数和当前站点的索引
    df_operating_segments['num_stations'] = df_operating_segments['车辆ID'].map(vehicle_station_counts)
    df_operating_segments['sid'] = df_operating_segments.apply(
        lambda row: row['站点ID'] - (parameters["up_station_count"] if row['方向'] == 'down' else 0), axis=1)

    # 过滤掉每个方向上的最后一站（因为不再有后续路段运营成本）
    df_operating_segments = df_operating_segments[
        df_operating_segments['sid'] < df_operating_segments['num_stations'] - 1
        ]

    # 遍历每个路段的起点站记录来计算运营成本
    for index, row in df_operating_segments.iterrows():
        adjusted_p_modules = row["调整后乘客模块"]
        adjusted_f_modules = row["调整后货物模块"]

        # 路段总模块数
        total_modules = adjusted_p_modules + adjusted_f_modules

        # 路段运营成本 = 时间间隔 * (固定成本 + 可变成本 * 总模块数^alpha)
        segment_cost = parameters["t_s_s1"] * (parameters["C_F"] + parameters["C_V"] * total_modules ** parameters["alpha"])

        modular_bus_cost += segment_cost

    # ... (后续成本计算和返回保持不变)

    # # 计算运营成本
    # print('计算运营成本')
    # modular_bus_cost = 0
    # for direction in ["up", "down"]:
    #     vehicle_dispatch = individual[direction]["vehicle_dispatch"]
    #     vehicle_initial_allocation = individual[direction]["initial_allocation"]
    #
    #     for vid, vehicle in vehicle_dispatch.items():
    #         p_modules = vehicle_initial_allocation[vid]["passenger_modules"]
    #         f_modules = vehicle_initial_allocation[vid]["freight_modules"]
    #         modular_bus_cost += parameters["t_s_s1"] * (
    #                 parameters["C_F"] + parameters["C_V"] * (p_modules + f_modules) ** parameters["alpha"]
    #         )

    # 计算未服务需求的等待时间成本
    print('计算未服务需求的惩罚成本')
    # ==================== 代码修改处 开始 ====================
    # 计算未服务的剩余需求
    remaining_passengers = 0
    for s in a_matrix_p_up:
        for s_dest in a_matrix_p_up[s]:
            for t in a_matrix_p_up[s][s_dest]:
                remaining_passengers += a_matrix_p_up[s][s_dest][t]
    for s in a_matrix_p_down:
        for s_dest in a_matrix_p_down[s]:
            for t in a_matrix_p_down[s][s_dest]:
                remaining_passengers += a_matrix_p_down[s][s_dest][t]

    remaining_freights = 0
    for s in a_matrix_f_up:
        for s_dest in a_matrix_f_up[s]:
            for t in a_matrix_f_up[s][s_dest]:
                remaining_freights += a_matrix_f_up[s][s_dest][t]
    for s in a_matrix_f_down:
        for s_dest in a_matrix_f_down[s]:
            for t in a_matrix_f_down[s][s_dest]:
                remaining_freights += a_matrix_f_down[s][s_dest][t]

    # 计算未服务惩罚成本
    unserved_penalty_cost = (remaining_passengers + remaining_freights) * parameters['penalty_cost_per_unit']

    total_cost = ω * modular_bus_cost + φ * (passenger_waiting_cost + freight_waiting_cost + unserved_penalty_cost)

    # === 新增：将三种成本打包 ===
    cost_components = {
        "mav_transport_cost": float(modular_bus_cost),
        "passenger_waiting_cost": float(passenger_waiting_cost),
        "freight_waiting_cost": float(freight_waiting_cost),
        "waiting_cost": float(passenger_waiting_cost) + float(freight_waiting_cost),
        "unserved_penalty_cost": float(unserved_penalty_cost),
        "unserved_passengers": float(remaining_passengers),
        "unserved_freights": float(remaining_freights),
    }

    print('individual.cost_components:', cost_components)

    # 计算剩余需求（分别计算上行和下行）
    remaining_passengers = 0
    remaining_freights = 0

    # 计算上行剩余需求
    for s in a_matrix_p_up:
        for s_dest in a_matrix_p_up[s]:
            for t in a_matrix_p_up[s][s_dest]:
                remaining_passengers += a_matrix_p_up[s][s_dest][t]

    remaining_passengers_up = remaining_passengers

    for s in a_matrix_f_up:
        for s_dest in a_matrix_f_up[s]:
            for t in a_matrix_f_up[s][s_dest]:
                remaining_freights += a_matrix_f_up[s][s_dest][t]

    remaining_freights_up = remaining_freights

    # 计算下行剩余需求
    for s in a_matrix_p_down:
        for s_dest in a_matrix_p_down[s]:
            for t in a_matrix_p_down[s][s_dest]:
                remaining_passengers += a_matrix_p_down[s][s_dest][t]

    remaining_passengers_down = remaining_passengers - remaining_passengers_up

    for s in a_matrix_f_down:
        for s_dest in a_matrix_f_down[s]:
            for t in a_matrix_f_down[s][s_dest]:
                remaining_freights += a_matrix_f_down[s][s_dest][t]

    remaining_freights_down = remaining_freights - remaining_freights_up

    # ==================== 新增逻辑：开始 ====================
    print(f"✅ 仿真完成 - 总成本: {total_cost:.2f}")
    print(f"   系统服务乘客: {total_served_passengers}, 系统服务货物: {total_served_freight}")
    print(f"   up剩余乘客: {remaining_passengers_up}, up剩余货物: {remaining_freights_up}")
    print(f"   down剩余乘客: {remaining_passengers_down}, down剩余货物: {remaining_freights_down}")
    print(f"   系统剩余乘客: {remaining_passengers}, 系统剩余货物: {remaining_freights}")
    # ==================== 新增逻辑：结束 ====================

    df_enriched = pd.DataFrame(df_enriched)

    print('返回函数返回值')

    return vehicle_schedule, total_cost, remaining_passengers, remaining_freights, failure_records, df_enriched, module_analysis_records, cost_components


# ====================================================================================================================
# 【方案一】新增的评估函数：模拟并**评估**给定的模块调整策略
# ====================================================================================================================
def simulate_and_evaluate_individual(individual, parameters, global_demand_data,
                                     passenger_demand_up, passenger_demand_down,
                                     freight_demand_up, freight_demand_down):
    """
    确定性评估函数。
    接收一个包含完整 module_adjustments 的个体，并严格按照其进行仿真和成本计算。
    """

    # 初始化集成系统
    module_system = IntegratedBusModuleSystem(
        alpha=parameters['alpha'],
        beta=parameters['beta'],
        C_p=parameters['passenger_per_module'],
        C_f=parameters['freight_per_module']
    )

    # 初始化变量
    df_enriched = []
    failure_records = []
    module_analysis_records = []

    # 获取需求矩阵的副本
    a_matrix_p_up = copy.deepcopy(global_demand_data["a_matrix_p_up"])
    a_matrix_f_up = copy.deepcopy(global_demand_data["a_matrix_f_up"])
    a_matrix_p_down = copy.deepcopy(global_demand_data["a_matrix_p_down"])
    a_matrix_f_down = copy.deepcopy(global_demand_data["a_matrix_f_down"])

    # 合并调度信息
    # ... (与原函数一致)
    vehicle_dispatch_up = individual["up"]["vehicle_dispatch"]
    vehicle_dispatch_down = individual["down"]["vehicle_dispatch"]
    combined_vehicle_dispatch = {**vehicle_dispatch_up, **vehicle_dispatch_down}

    # 获取车辆信息
    all_vehicles, vehicle_schedule, _ = collect_vehicle_info(
        individual, parameters,
        passenger_demand_up, passenger_demand_down,
        freight_demand_up, freight_demand_down
    )

    # 初始化站点模块存储
    station_module_stock = {sid: {"modules": 0} for sid in range(parameters["up_station_count"] * 2)}

    infeasible = False
    total_cost = 0
    total_passenger_waiting_time_cost = 0
    total_freight_waiting_time_cost = 0
    max_simulation_time = 0

    # 主仿真循环
    for direction in ["up", "down"]:
        if direction == "up":
            num_stations = parameters["up_station_count"]
            a_matrix_p = a_matrix_p_up
            a_matrix_f = a_matrix_f_up
        else:
            num_stations = parameters["up_station_count"] + parameters["up_station_count"]
            a_matrix_p = a_matrix_p_down
            a_matrix_f = a_matrix_f_down

        # 遍历车辆
        for vehicle in all_vehicles:
            if vehicle["direction"] != direction:
                continue

            vid = vehicle["global_vid"]
            offset = vehicle["station_offset"]
            arrival_time = vehicle["dispatch"]["arrival_time"]
            last_departure_time = arrival_time + (vehicle["num_stations"] - 1) * parameters["t_s_s1"]
            max_simulation_time = max(max_simulation_time, last_departure_time)

            onboard_passengers = {}
            onboard_freight = {}

            current_p_modules = individual[direction]["initial_allocation"][vid]["passenger_modules"]
            current_f_modules = individual[direction]["initial_allocation"][vid]["freight_modules"]

            # print('--------之前---------')
            # print('current_p_modules:', current_p_modules)
            # print('current_f_modules:', current_f_modules)
            # print('--------之前---------')

            next_p = 0
            next_f = 0

            # print('vid:', vid)

            for sid in range(vehicle["num_stations"]):
                # print('sid:', sid)
                station_id = sid + offset
                # print('station_id:', station_id)

                # print('起始站点station_id:', station_id)

                if sid > 0:
                    arrival_time += parameters["t_s_s1"]
                    current_p_modules = next_p
                    current_f_modules = next_f

                store_modules = station_module_stock[station_id]["modules"]

                onboard_p_before = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_before = sum(sum(f.values()) for f in onboard_freight.values())

                alighted_p = sum(sum(p_dict.values()) for dest, p_dict in onboard_passengers.items() if dest == station_id)
                alighted_f = sum(sum(f_dict.values()) for dest, f_dict in onboard_freight.items() if dest == station_id)

                onboard_passengers.pop(station_id, None)
                onboard_freight.pop(station_id, None)

                onboard_p_after = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_after = sum(sum(f.values()) for f in onboard_freight.values())

                # print('onboard_p_after:', onboard_p_after)
                # print('onboard_f_after:', onboard_f_after)

                waiting_p = 0
                waiting_f = 0

                for s_dest in range(station_id + 1, num_stations):
                    # print('目标站点s_dest:', s_dest)
                    # print('末尾站点num_stations:', num_stations)
                    for t in range(arrival_time + 1):

                        if (station_id in a_matrix_p and s_dest in a_matrix_p[station_id]
                                and t in a_matrix_p[station_id][s_dest]):
                            waiting_p += a_matrix_p[station_id][s_dest][t]

                        if (station_id in a_matrix_f and s_dest in a_matrix_f[station_id]
                                and t in a_matrix_f[station_id][s_dest]):
                            waiting_f += a_matrix_f[station_id][s_dest][t]

                # 1. 计算模块需求和调整范围 (与原函数一致)
                module_analysis = module_system.calculate_station_module_requirements(
                    n=vid, k=station_id,
                    p_n_k=current_p_modules, f_n_k=current_f_modules, store_modules=store_modules,
                    onboard_pass_before=onboard_p_before, onboard_cargo_before=onboard_f_before,
                    off_pass=alighted_p, off_cargo=alighted_f,
                    onboard_pass_after=onboard_p_after, onboard_cargo_after=onboard_f_after,
                    waiting_pass=waiting_p, waiting_cargo=waiting_f
                )

                # print('current_p_modules:', current_p_modules)
                # print('current_f_modules:', current_f_modules)

                # 【关键修改】从个体中读取预先确定的模块调整量
                try:
                    delta_p = individual[direction]["module_adjustments"][vid][station_id]["delta_p"]
                    delta_f = individual[direction]["module_adjustments"][vid][station_id]["delta_f"]
                    # print('delta_p:', delta_p)
                    # print('delta_f:', delta_f)
                except KeyError:
                    # print('station_id:', station_id)
                    # print('vid:', vid)
                    # print('个体中缺少调整策略，则视作0调整')
                    # 如果个体中缺少调整策略，则视作0调整
                    delta_p = 0
                    delta_f = 0

                adjusted_p_modules = current_p_modules + delta_p
                adjusted_f_modules = current_f_modules + delta_f

                # print('adjusted_p_modules:', adjusted_p_modules)
                # print('adjusted_f_modules:', adjusted_f_modules)

                # 2. 更新调整后的容量 (与原函数一致)
                adjusted_p_capacity = adjusted_p_modules * parameters["passenger_per_module"]
                adjusted_f_capacity = adjusted_f_modules * parameters["freight_per_module"]

                # 3. 验证调整后容量是否满足在车需求 (与原函数一致)
                if onboard_p_after > adjusted_p_capacity or onboard_f_after > adjusted_f_capacity:
                    print('onboard_p_after:', onboard_p_after, 'adjusted_p_capacity:', adjusted_p_capacity)
                    print('onboard_f_after:', onboard_f_after, 'adjusted_f_capacity:', adjusted_f_capacity)
                    print(f"❌ 模块调整后容量超限: 车辆{vid} 站点{station_id}")
                    infeasible = True
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), [], []

                # 4. 更新站点模块库存 (与原函数一致)
                station_module_stock_before = station_module_stock[station_id]["modules"]
                station_module_stock[station_id]["modules"] -= (delta_p + delta_f)
                station_module_stock_after = station_module_stock[station_id]["modules"]

                if station_module_stock_after < 0:
                    print('逆天了 竟然小于0')

                # 5. 上下车逻辑 (与原函数一致)
                available_p_capacity = adjusted_p_capacity - onboard_p_after
                boarded_p = 0
                served_passenger_waiting_time = 0
                if available_p_capacity > 0:
                    for s_dest in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_p >= available_p_capacity: break
                            if (station_id in a_matrix_p and s_dest in a_matrix_p[station_id] and
                                    t in a_matrix_p[station_id][s_dest]):
                                demand_p = a_matrix_p[station_id][s_dest][t]
                                board_now_p = min(demand_p, available_p_capacity - boarded_p)
                                if board_now_p > 0:
                                    waiting_time = arrival_time - t
                                    served_passenger_waiting_time += board_now_p * waiting_time
                                    if s_dest not in onboard_passengers: onboard_passengers[s_dest] = {}
                                    if arrival_time not in onboard_passengers[s_dest]: onboard_passengers[s_dest][
                                        arrival_time] = 0
                                    onboard_passengers[s_dest][arrival_time] += board_now_p
                                    boarded_p += board_now_p
                                    a_matrix_p[station_id][s_dest][t] -= board_now_p

                available_f_capacity = adjusted_f_capacity - onboard_f_after
                boarded_f = 0
                served_freight_waiting_time = 0
                if available_f_capacity > 0:
                    for s_dest in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_f >= available_f_capacity: break
                            if (station_id in a_matrix_f and s_dest in a_matrix_f[station_id] and
                                    t in a_matrix_f[station_id][s_dest]):
                                demand_f = a_matrix_f[station_id][s_dest][t]
                                board_now_f = min(demand_f, available_f_capacity - boarded_f)
                                if board_now_f > 0:
                                    waiting_time = arrival_time - t
                                    served_freight_waiting_time += board_now_f * waiting_time
                                    if s_dest not in onboard_freight: onboard_freight[s_dest] = {}
                                    if arrival_time not in onboard_freight[s_dest]: onboard_freight[s_dest][
                                        arrival_time] = 0
                                    onboard_freight[s_dest][arrival_time] += board_now_f
                                    boarded_f += board_now_f
                                    a_matrix_f[station_id][s_dest][t] -= board_now_f

                total_passenger_waiting_time_cost += served_passenger_waiting_time
                total_freight_waiting_time_cost += served_freight_waiting_time

                if sid < vehicle["num_stations"] - 1:
                    next_p = adjusted_p_modules
                    next_f = adjusted_f_modules

                # 记录模块分析结果（此处的module_analysis是基于需求计算的，不包含随机调整）
                module_analysis_records.append({
                    'timestamp': arrival_time,
                    'vehicle_id': vid,
                    'station_id': station_id,
                    'direction': direction,
                    'analysis': module_analysis
                })

                # 记录详细信息 (与原函数一致)
                df_enriched.append({
                    "车辆ID": vid,
                    "站点ID": station_id,
                    "方向": direction,
                    "到达时间": arrival_time,

                    # 模块信息（调整前后）
                    "调整前乘客模块": current_p_modules,
                    "调整前货物模块": current_f_modules,
                    "调整前总模块数": current_p_modules + current_f_modules,
                    "调整后乘客模块": adjusted_p_modules,
                    "调整后货物模块": adjusted_f_modules,
                    "调整后总模块数": adjusted_p_modules + adjusted_f_modules,
                    "模块增量_乘客": delta_p,
                    "模块增量_货物": delta_f,

                    # 乘客/货物流动信息
                    "下车前在车乘客": onboard_p_before,
                    "下车前在车货物": onboard_f_before,
                    "下车乘客": alighted_p,
                    "下车货物": alighted_f,
                    "下车后在车乘客": onboard_p_after,
                    "下车后在车货物": onboard_f_after,

                    "等待乘客需求": waiting_p,
                    "等待货物需求": waiting_f,
                    "上车乘客": boarded_p,
                    "上车货物": boarded_f,

                    # ==================== 代码修改处 开始 ====================
                    "上车后在车乘客数量": onboard_p_after + boarded_p,
                    "上车后在车货物数量": onboard_f_after + boarded_f,
                    # ==================== 代码修改处 结束 ====================

                    # 容量信息（调整后）
                    "调整后乘客总容量": adjusted_p_capacity,
                    "调整后货物总容量": adjusted_f_capacity,
                    "调整后乘客可用容量": available_p_capacity,
                    "调整后货物可用容量": available_f_capacity,

                    # 成本信息
                    "乘客等待时间成本": served_passenger_waiting_time,
                    "货物等待时间成本": served_freight_waiting_time,

                    # 库存信息
                    "站点进站前库存": station_module_stock_before,
                    "站点出站后库存": station_module_stock_after,

                    # # 模块分析结果
                    # "最少需要乘客模块": module_analysis['passenger_analysis']['min_modules_needed'],
                    # "最优乘客模块": module_analysis['passenger_analysis']['optimal_modules'],
                    # "最少需要货物模块": module_analysis['freight_analysis']['min_modules_needed'],
                    # "最优货物模块": module_analysis['freight_analysis']['optimal_modules'],
                    # "乘客模块调整范围": f"{module_analysis['adjustment_ranges']['passenger_modules']['delta_range']}",
                    # "货物模块调整范围": f"{module_analysis['adjustment_ranges']['freight_modules']['delta_range']}",
                    # "总模块可行范围": f"{module_analysis['module_constraints']['feasible_total_range']}",
                })
                # df_enriched.append({
                #     "车辆ID": vid, "站点ID": station_id, "方向": direction, "到达时间": arrival_time,
                #     "调整前乘客模块": current_p_modules, "调整前货物模块": current_f_modules,
                #     "调整后乘客模块": adjusted_p_modules, "调整后货物模块": adjusted_f_modules,
                #     "模块增量_乘客": delta_p, "模块增量_货物": delta_f,
                #     "下车后在车乘客": onboard_p_after, "上车乘客": boarded_p,
                #     "下车后在车货物": onboard_f_after, "上车货物": boarded_f,
                #     # ==================== 代码修改处 开始 ====================
                #     "上车后在车乘客数量": onboard_p_after + boarded_p,
                #     "上车后在车货物数量": onboard_f_after + boarded_f,
                #     # ==================== 代码修改处 结束 ====================
                #     "乘客等待时间成本": served_passenger_waiting_time, "货物等待时间成本": served_freight_waiting_time,
                #     "站点进站前库存": station_module_stock_before, "站点出站后库存": station_module_stock_after,
                # })

    if infeasible:
        print("❌ 方案不可行")
        return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), [], []

    # # 计算成本、剩余需求等... (此部分与原函数一致)
    # unserved_passenger_waiting_cost = 0
    # unserved_freight_waiting_cost = 0
    #
    # for a_matrix_p in [a_matrix_p_up, a_matrix_p_down]:
    #     for s in a_matrix_p:
    #         for s_dest in a_matrix_p[s]:
    #             for t in a_matrix_p[s][s_dest]:
    #                 remaining_p = a_matrix_p[s][s_dest][t]
    #                 if remaining_p > 0:
    #                     waiting_time = max_simulation_time - t
    #                     unserved_passenger_waiting_cost += remaining_p * waiting_time
    #
    # for a_matrix_f in [a_matrix_f_up, a_matrix_f_down]:
    #     for s in a_matrix_f:
    #         for s_dest in a_matrix_f[s]:
    #             for t in a_matrix_f[s][s_dest]:
    #                 remaining_f = a_matrix_f[s][s_dest][t]
    #                 if remaining_f > 0:
    #                     waiting_time = max_simulation_time - t
    #                     unserved_freight_waiting_cost += remaining_f * waiting_time

    # total_passenger_waiting_time_cost += unserved_passenger_waiting_cost
    # total_freight_waiting_time_cost += unserved_freight_waiting_cost

    passenger_waiting_cost = total_passenger_waiting_time_cost * parameters["passenger_waiting_cost"]
    freight_waiting_cost = total_freight_waiting_time_cost * parameters["freight_waiting_cost"]

    df_enriched = pd.DataFrame(df_enriched)

    # 计算运营成本 (基于每段路程的实际模块配置)
    modular_bus_cost = 0

    # 过滤出非终点站点的记录
    df_operating_segments = df_enriched.copy()

    # 获取每个车辆的站点总数
    vehicle_station_counts = {v["global_vid"]: v["num_stations"] for v in all_vehicles}

    # 为df_operating_segments添加车辆总站点数和当前站点的索引
    df_operating_segments['num_stations'] = df_operating_segments['车辆ID'].map(vehicle_station_counts)
    df_operating_segments['sid'] = df_operating_segments.apply(
        lambda row: row['站点ID'] - (parameters["up_station_count"] if row['方向'] == 'down' else 0), axis=1)

    # 过滤掉每个方向上的最后一站（因为不再有后续路段运营成本）
    df_operating_segments = df_operating_segments[
        df_operating_segments['sid'] < df_operating_segments['num_stations'] - 1
        ]

    # 遍历每个路段的起点站记录来计算运营成本
    for index, row in df_operating_segments.iterrows():
        adjusted_p_modules = row["调整后乘客模块"]
        adjusted_f_modules = row["调整后货物模块"]

        # 路段总模块数
        total_modules = adjusted_p_modules + adjusted_f_modules

        # 路段运营成本 = 时间间隔 * (固定成本 + 可变成本 * 总模块数^alpha)
        segment_cost = parameters["t_s_s1"] * (
                parameters["C_F"] + parameters["C_V"] * (total_modules) ** parameters["alpha"]
        )
        modular_bus_cost += segment_cost

    # modular_bus_cost = 0
    #
    # for direction in ["up", "down"]:
    #     vehicle_dispatch = individual[direction]["vehicle_dispatch"]
    #     vehicle_initial_allocation = individual[direction]["initial_allocation"]
    #     for vid, vehicle in vehicle_dispatch.items():
    #         p_modules = vehicle_initial_allocation[vid]["passenger_modules"]
    #         f_modules = vehicle_initial_allocation[vid]["freight_modules"]
    #         modular_bus_cost += parameters["t_s_s1"] * (
    #                 parameters["C_F"] + parameters["C_V"] * (p_modules + f_modules) ** parameters["alpha"]
    #         )

    # ==================== 代码修改处 开始 ====================
    # 计算未服务的剩余需求
    remaining_passengers = 0
    for s in a_matrix_p_up:
        for s_dest in a_matrix_p_up[s]:
            for t in a_matrix_p_up[s][s_dest]:
                remaining_passengers += a_matrix_p_up[s][s_dest][t]
    for s in a_matrix_p_down:
        for s_dest in a_matrix_p_down[s]:
            for t in a_matrix_p_down[s][s_dest]:
                remaining_passengers += a_matrix_p_down[s][s_dest][t]

    remaining_freights = 0
    for s in a_matrix_f_up:
        for s_dest in a_matrix_f_up[s]:
            for t in a_matrix_f_up[s][s_dest]:
                remaining_freights += a_matrix_f_up[s][s_dest][t]
    for s in a_matrix_f_down:
        for s_dest in a_matrix_f_down[s]:
            for t in a_matrix_f_down[s][s_dest]:
                remaining_freights += a_matrix_f_down[s][s_dest][t]

    # 计算未服务惩罚成本
    unserved_penalty_cost = (remaining_passengers + remaining_freights) * parameters['penalty_cost_per_unit']

    # total_cost = passenger_waiting_cost + freight_waiting_cost + modular_bus_cost + unserved_penalty_cost
    # total_cost = ω * modular_bus_cost + φ * (passenger_waiting_cost + freight_waiting_cost) + unserved_penalty_cost
    total_cost = ω * modular_bus_cost + φ * (passenger_waiting_cost + freight_waiting_cost + unserved_penalty_cost)

    cost_components = {
        "mav_transport_cost": float(modular_bus_cost),
        "passenger_waiting_cost": float(passenger_waiting_cost),
        "freight_waiting_cost": float(freight_waiting_cost),
        "waiting_cost": float(passenger_waiting_cost) + float(freight_waiting_cost),
        "unserved_penalty_cost": float(unserved_penalty_cost),
        "unserved_passengers": float(remaining_passengers),
        "unserved_freights": float(remaining_freights),
    }

    # print('individual.cost_components:', cost_components)
    # 统计未服务的需求
    remaining_passengers = 0
    remaining_freights = 0

    for s in a_matrix_p_up:
        for s_dest in a_matrix_p_up[s]:
            for t in a_matrix_p_up[s][s_dest]:
                remaining_passengers += a_matrix_p_up[s][s_dest][t]

    remaining_passengers_up = remaining_passengers

    for s in a_matrix_f_up:
        for s_dest in a_matrix_f_up[s]:
            for t in a_matrix_f_up[s][s_dest]:
                remaining_freights += a_matrix_f_up[s][s_dest][t]

    remaining_freights_up = remaining_freights

    for s in a_matrix_p_down:
        for s_dest in a_matrix_p_down[s]:
            for t in a_matrix_p_down[s][s_dest]:
                remaining_passengers += a_matrix_p_down[s][s_dest][t]

    remaining_passengers_down = remaining_passengers - remaining_passengers_up

    for s in a_matrix_f_down:
        for s_dest in a_matrix_f_down[s]:
            for t in a_matrix_f_down[s][s_dest]:
                remaining_freights += a_matrix_f_down[s][s_dest][t]

    remaining_freights_down = remaining_freights - remaining_freights_up

    print(f"✅ 评估仿真完成 - 总成本: {total_cost:.2f}")
    print('cost_components:', cost_components)
    # print(f"   系统服务乘客: {total_served_passengers}, 系统服务货物: {total_served_freight}")
    print(f"   up剩余乘客: {remaining_passengers_up}, up剩余货物: {remaining_freights_up}")
    print(f"   down剩余乘客: {remaining_passengers_down}, down剩余货物: {remaining_freights_down}")
    print(f"   系统剩余乘客: {remaining_passengers}, 系统剩余货物: {remaining_freights}")
    df_enriched = pd.DataFrame(df_enriched)

    return vehicle_schedule, total_cost, remaining_passengers, remaining_freights, failure_records, df_enriched, module_analysis_records, cost_components


def collect_vehicle_info(individual, parameters,
                         passenger_demand_up, passenger_demand_down,
                         freight_demand_up, freight_demand_down):
    vehicle_schedule = {}
    all_vehicles = []

    for direction in ["up", "down"]:
        df = reconstruct_schedule_dataframe(individual, parameters, direction)
        offset = 0 if direction == "up" else parameters["up_station_count"]
        num_stations = parameters["up_station_count"]
        vid_offset = 0 if direction == "up" else 0

        for vid in df["车辆ID"].unique():
            vid = int(vid)
            global_vid = vid + vid_offset
            vehicle_data = df[df["车辆ID"] == vid].sort_values(by="到达时间")
            vehicle_schedule[global_vid] = {"stations": {}}

            all_vehicles.append({
                "vid": vid,
                "global_vid": global_vid,
                "direction": direction,
                "dispatch": individual[direction]["vehicle_dispatch"][vid],
                "initial_alloc": individual[direction]["initial_allocation"][vid],
                "station_offset": offset,
                "num_stations": num_stations,
                "passenger_demand": passenger_demand_up if direction == "up" else passenger_demand_down,
                "freight_demand": freight_demand_up if direction == "up" else freight_demand_down
            })

    return all_vehicles, vehicle_schedule, None