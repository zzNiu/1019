import copy
import math
import pandas as pd
from df_schedule_construct import reconstruct_schedule_dataframe

def simulate_and_validate_feasibility(individual, parameters, global_demand_data, passenger_demand_up,
                                      passenger_demand_down, freight_demand_up, freight_demand_down):
    """
    仿真函数，验证调度方案的可行性并计算成本

    Returns:
        vehicle_schedule: 车辆调度方案
        total_cost: 总成本
        remaining_passengers: 剩余乘客
        remaining_freights: 剩余货物
        failure_records: 失败记录
        df_enriched: 丰富的数据框
    """
    # 解包
    df_enriched = []
    failure_records = []  # <== 添加此初始化
    module_adjustment_ranges = {}  # 新增：存储模块调整范围信息

    # 获取需求矩阵的副本，避免修改全局变量
    a_matrix_p_up = copy.deepcopy(global_demand_data["a_matrix_p_up"])
    a_matrix_f_up = copy.deepcopy(global_demand_data["a_matrix_f_up"])
    a_matrix_p_down = copy.deepcopy(global_demand_data["a_matrix_p_down"])
    a_matrix_f_down = copy.deepcopy(global_demand_data["a_matrix_f_down"])

    vehicle_dispatch_up = individual["up"]["vehicle_dispatch"]
    vehicle_dispatch_down = individual["down"]["vehicle_dispatch"]

    vehicle_module_adjustments_up = individual["up"]["module_adjustments"]
    vehicle_module_adjustments_down = individual["down"]["module_adjustments"]

    # 合并上下行的 vehicle_dispatch
    combined_vehicle_dispatch = {**vehicle_dispatch_up, **vehicle_dispatch_down}
    vehicle_module_adjustments = {**vehicle_module_adjustments_up, **vehicle_module_adjustments_down}

    infeasible = False  # 标志变量
    total_cost = 0
    total_passenger_waiting_time_cost = 0
    total_freight_waiting_time_cost = 0

    # 新增：未服务乘客和货物的等待时间成本
    unserved_passenger_waiting_time_cost = 0
    unserved_freight_waiting_time_cost = 0

    # 调用提取的函数获取 all_vehicles、vehicle_schedule
    all_vehicles, vehicle_schedule, _ = collect_vehicle_info(individual, parameters, passenger_demand_up,
                                                             passenger_demand_down, freight_demand_up,
                                                             freight_demand_down)

    # 初始化站点模块存储数量
    station_module_stock = {sid: {"modules": 0} for sid in range(parameters["up_station_count"] * 2)}

    # 新增：初始化模块调整范围记录
    for direction in ["up", "down"]:
        module_adjustment_ranges[direction] = {}

    # 记录最大仿真时间（最后一班车离开最后一个站点的时间）
    max_simulation_time = 0

    # 仿真校验逻辑
    for direction in ["up", "down"]:
        print('方向：', direction)

        if direction == "up":
            num_stations = parameters["up_station_count"]
        else:
            offset = parameters["up_station_count"]
            # 对齐上下行站点数: 下行也只跑 up_station_count 个站
            num_stations = parameters["up_station_count"]

        vehicle_dispatch = individual[direction]["vehicle_dispatch"]
        vehicle_initial_allocation = individual[direction]["initial_allocation"]

        print('根据方向 匹配相应的需求矩阵')
        a_matrix_p = a_matrix_p_up if direction == "up" else a_matrix_p_down
        a_matrix_f = a_matrix_f_up if direction == "up" else a_matrix_f_down

        for vehicle in all_vehicles:
            if vehicle["direction"] != direction:
                continue  # 跳过不属于当前方向的车辆

            vid = vehicle["global_vid"]
            offset = vehicle["station_offset"]  # 站点偏移
            arrival_time = vehicle["dispatch"]["arrival_time"]

            # 计算该车辆的最后离站时间
            last_departure_time = arrival_time + (vehicle["num_stations"] - 1) * parameters["t_s_s1"]
            # 更新最大仿真时间
            max_simulation_time = max(max_simulation_time, last_departure_time)

            onboard_passengers = {}
            onboard_freight = {}

            # 新增：初始化该车辆的调整范围记录
            if vid not in module_adjustment_ranges[direction]:
                module_adjustment_ranges[direction][vid] = {}

            for sid in range(vehicle["num_stations"]):
                station_id = sid + offset

                # 记录进站前库存
                station_module_stock_before = station_module_stock[station_id]["modules"]

                # 检查站点库存范围
                if station_module_stock_before > parameters["max_modules_stock"] or station_module_stock_before < \
                        parameters["min_modules_stock"]:
                    print(f"❌ 站点 {station_id} 库存超范围 {station_module_stock_before}")
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), {}

                # 后面的站点添加站点间运行时间
                if sid > 0:
                    arrival_time += parameters["t_s_s1"]

                # 获取当前站点的模块数量
                current_p_modules = vehicle_schedule[vid]["stations"].get(station_id, {}).get("passenger_modules", 0)
                current_f_modules = vehicle_schedule[vid]["stations"].get(station_id, {}).get("freight_modules", 0)

                total_p_capacity = current_p_modules * parameters["passenger_per_module"]
                total_f_capacity = current_f_modules * parameters["freight_per_module"]

                # 计算在车乘客数量（下车前）
                onboard_p_before = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_before = sum(sum(f.values()) for f in onboard_freight.values())

                # 乘客下车
                alighted_p = sum(sum(p_dict.values()) for dest, p_dict in onboard_passengers.items() if dest == station_id)
                onboard_passengers.pop(station_id, None)

                # 货物下车
                alighted_f = sum(sum(f_dict.values()) for dest, f_dict in onboard_freight.items() if dest == station_id)
                onboard_freight.pop(station_id, None)

                # 计算下车后的在车数量
                onboard_p_after_alighting = sum(sum(p.values()) for p in onboard_passengers.values())
                onboard_f_after_alighting = sum(sum(f.values()) for f in onboard_freight.values())

                # 容量检查
                if onboard_p_after_alighting > total_p_capacity:
                    # 添加详细的打印信息
                    print(f"===== 乘客模块容量超出详情 =====")
                    print(f"车辆ID: {vid}, 站点ID: {station_id}, 时间: {arrival_time}")
                    print(f"当前乘客模块数量: {current_p_modules}, 模块容量: {total_p_capacity}")
                    print(f"下车前在车乘客数量: {onboard_p_before}")
                    print(f"下车乘客数量: {alighted_p}")
                    print(f"下车后在车乘客数量: {onboard_p_after_alighting}")
                    print('onboard_passengers:', onboard_passengers)
                    print(f"超出容量: {onboard_p_after_alighting - total_p_capacity}")
                    print("===== 详情结束 =====")

                    missing = math.ceil((onboard_p_after_alighting - total_p_capacity) / parameters['passenger_per_module'])
                    failure_records.append({
                        "station_id": station_id,
                        "车辆编号": vid,
                        "时间": arrival_time,
                        "type": "passenger",
                        "missing": missing
                    })
                    infeasible = True
                    print(f"❌ 乘客模块容量超出限制：车辆 {vid} 在时间 {arrival_time} 在站点 {station_id}")
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), {}

                if onboard_f_after_alighting > total_f_capacity:
                    # 添加详细的打印信息
                    print(f"===== 货物模块容量超出详情 =====")
                    print(f"车辆ID: {vid}, 站点ID: {station_id}, 时间: {arrival_time}")
                    print(f"当前货物模块数量: {current_f_modules}, 模块容量: {total_f_capacity}")
                    print(f"下车前在车货物数量: {onboard_f_before}")
                    print(f"下车货物数量: {alighted_f}")
                    print(f"下车后在车货物数量: {onboard_f_after_alighting}")
                    print('onboard_freight:', onboard_freight)
                    print(f"超出容量: {onboard_f_after_alighting - total_f_capacity}")
                    print("===== 详情结束 =====")

                    missing = math.ceil((onboard_f_after_alighting - total_f_capacity) / parameters['freight_per_module'])
                    failure_records.append({
                        "station_id": station_id,
                        "车辆编号": vid,
                        "时间": arrival_time,
                        "type": "freight",
                        "missing": missing
                    })
                    infeasible = True
                    print(f"❌ 货物模块容量超出限制：车辆 {vid} 在时间 {arrival_time} 在站点 {station_id}")
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), {}

                # 计算可用容量
                available_p_capacity = total_p_capacity - onboard_p_after_alighting
                available_f_capacity = total_f_capacity - onboard_f_after_alighting

                # 乘客上车逻辑
                boarded_p = 0
                served_passenger_waiting_time = 0
                potential_passenger_demand = 0  # 新增：潜在乘客需求

                if available_p_capacity > 0:
                    for s_p in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_p < available_p_capacity:
                                if (station_id in a_matrix_p and s_p in a_matrix_p[station_id] and
                                        t in a_matrix_p[station_id][s_p]):

                                    demand_p = a_matrix_p[station_id][s_p][t]
                                    potential_passenger_demand += demand_p  # 累计潜在需求

                                    board_now_p = min(demand_p, available_p_capacity - boarded_p)

                                    if board_now_p > 0:
                                        waiting_time = arrival_time - t
                                        served_passenger_waiting_time += board_now_p * waiting_time

                                        if s_p not in onboard_passengers:
                                            onboard_passengers[s_p] = {}
                                        if arrival_time not in onboard_passengers[s_p]:
                                            onboard_passengers[s_p][arrival_time] = 0
                                        onboard_passengers[s_p][arrival_time] += board_now_p

                                        boarded_p += board_now_p
                                        a_matrix_p[station_id][s_p][t] -= board_now_p

                # 货物上车逻辑
                boarded_f = 0
                served_freight_waiting_time = 0
                potential_freight_demand = 0  # 新增：潜在货物需求

                if available_f_capacity > 0:
                    for s_p in range(station_id + 1, num_stations):
                        for t in range(arrival_time + 1):
                            if boarded_f < available_f_capacity:
                                if (station_id in a_matrix_f and s_p in a_matrix_f[station_id] and
                                        t in a_matrix_f[station_id][s_p]):

                                    demand_f = a_matrix_f[station_id][s_p][t]
                                    potential_freight_demand += demand_f  # 累计潜在需求

                                    board_now_f = min(demand_f, available_f_capacity - boarded_f)

                                    if board_now_f > 0:
                                        waiting_time = arrival_time - t
                                        served_freight_waiting_time += board_now_f * waiting_time

                                        if s_p not in onboard_freight:
                                            onboard_freight[s_p] = {}
                                        if arrival_time not in onboard_freight[s_p]:
                                            onboard_freight[s_p][arrival_time] = 0
                                        onboard_freight[s_p][arrival_time] += board_now_f

                                        boarded_f += board_now_f
                                        a_matrix_f[station_id][s_p][t] -= board_now_f

                # 累加等待时间成本
                total_passenger_waiting_time_cost += served_passenger_waiting_time
                total_freight_waiting_time_cost += served_freight_waiting_time

                # 新增：计算模块调整范围
                adjustment_range = calculate_module_adjustment_range(
                    current_p_modules, current_f_modules,
                    onboard_p_after_alighting, onboard_f_after_alighting,
                    potential_passenger_demand, potential_freight_demand,
                    parameters
                )

                module_adjustment_ranges[direction][vid][station_id] = adjustment_range

                # 更新站点库存
                station_module_stock_after = station_module_stock[station_id]["modules"] - (
                        vehicle_schedule[vid]["stations"][station_id]["delta_p"] +
                        vehicle_schedule[vid]["stations"][station_id]["delta_f"])

                station_module_stock[station_id]["modules"] = station_module_stock_after

                if station_module_stock_after > parameters["max_modules_stock"] or station_module_stock_after < parameters["min_modules_stock"]:
                    print(f"❌ 站点 {station_id} 库存超范围 {station_module_stock_after}")
                    return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), {}



                # 记录详细信息到 df_enriched
                df_enriched.append({
                    "车辆ID": vid,
                    "站点ID": station_id,
                    "到达时间": arrival_time,
                    "乘客模块": current_p_modules,
                    "货物模块": current_f_modules,
                    "总模块数": current_p_modules + current_f_modules,
                    "乘客模块增量": vehicle_schedule[vid]["stations"][station_id]["delta_p"],
                    "货物模块增量": vehicle_schedule[vid]["stations"][station_id]["delta_f"],

                    "下车前在车乘客": onboard_p_before,
                    "下车前在车货物": onboard_f_before,
                    "下车乘客": alighted_p,
                    "下车货物": alighted_f,
                    "下车后在车乘客": onboard_p_after_alighting,
                    "下车后在车货物": onboard_f_after_alighting,

                    "乘客总容量": total_p_capacity,
                    "货物总容量": total_f_capacity,
                    "可用乘客容量": available_p_capacity,
                    "可用货物容量": available_f_capacity,

                    "上车乘客": boarded_p,
                    "上车货物": boarded_f,
                    "潜在乘客需求": potential_passenger_demand,
                    "潜在货物需求": potential_freight_demand,

                    "乘客等待时间": served_passenger_waiting_time,
                    "货物等待时间": served_freight_waiting_time,

                    "站点进站前库存": station_module_stock_before,
                    "站点离站后库存": station_module_stock_after,

                    # 新增：模块调整范围信息
                    "乘客模块可减少": adjustment_range["passenger_modules"]["max_decrease"],
                    "乘客模块可增加": adjustment_range["passenger_modules"]["max_increase"],
                    "货物模块可减少": adjustment_range["freight_modules"]["max_decrease"],
                    "货物模块可增加": adjustment_range["freight_modules"]["max_increase"],
                })

    # 计算成本等后续逻辑
    if infeasible:
        print("❌ 当前个体方案不合法")
        return {}, float("inf"), 1e9, 1e9, failure_records, pd.DataFrame([]), {}

    # 获取仿真最大时间
    print("最大仿真时间：", max_simulation_time)

    # 计算未服务乘客的等待时间
    for a_matrix_p in [a_matrix_p_up, a_matrix_p_down]:
        for s in a_matrix_p:
            for s_p in a_matrix_p[s]:
                for t in a_matrix_p[s][s_p]:
                    remaining_p = a_matrix_p[s][s_p][t]
                    if remaining_p > 0:
                        # 等待时间 = 最大时间 - 需求产生时间
                        waiting_time = max_simulation_time - t
                        unserved_passenger_waiting_time_cost += remaining_p * waiting_time

    # 计算未服务货物的等待时间
    for a_matrix_f in [a_matrix_f_up, a_matrix_f_down]:
        for s in a_matrix_f:
            for s_p in a_matrix_f[s]:
                for t in a_matrix_f[s][s_p]:
                    remaining_f = a_matrix_f[s][s_p][t]
                    if remaining_f > 0:
                        # 等待时间 = 最大时间 - 需求产生时间
                        waiting_time = max_simulation_time - t
                        unserved_freight_waiting_time_cost += remaining_f * waiting_time

    # 将未服务的等待时间成本加入总等待时间成本
    total_passenger_waiting_time_cost += unserved_passenger_waiting_time_cost
    total_freight_waiting_time_cost += unserved_freight_waiting_time_cost

    passenger_waiting_cost = total_passenger_waiting_time_cost * parameters["passenger_waiting_cost"]
    freight_waiting_cost = total_freight_waiting_time_cost * parameters["freight_waiting_cost"]

    # 计算公交能量消耗成本
    modular_bus_cost = 0
    for direction in ["up", "down"]:
        vehicle_dispatch = individual[direction]["vehicle_dispatch"]
        vehicle_initial_allocation = individual[direction]["initial_allocation"]

        for vid, vehicle in vehicle_dispatch.items():
            p_modules = vehicle_initial_allocation[vid]["passenger_modules"]
            f_modules = vehicle_initial_allocation[vid]["freight_modules"]
            modular_bus_cost += parameters["t_s_s1"] * (parameters["C_F"] + parameters["C_V"] * (p_modules + f_modules) ** parameters["alpha"])

    total_cost = passenger_waiting_cost + freight_waiting_cost + modular_bus_cost

    # 检查剩余需求
    remaining_passengers = sum(
        a_matrix_p[s][s_p][t]
        for s in a_matrix_p
        for s_p in a_matrix_p[s]
        for t in a_matrix_p[s][s_p]
    )

    remaining_freights = sum(
        a_matrix_f[s][s_p][t]
        for s in a_matrix_f
        for s_p in a_matrix_f[s]
        for t in a_matrix_f[s][s_p]
    )

    if remaining_passengers > 0 or remaining_freights > 0:
        print(f"⭕️ 存在未完成需求：剩余乘客 {remaining_passengers}，剩余货物 {remaining_freights}")
        # failure_records.append({
        #     "type": "global_unserved_demand",
        #     "remaining_passengers": remaining_passengers,
        #     "remaining_freights": remaining_freights
        # })
        # return {}, float('inf'), 1e9, 1e9, failure_records, pd.DataFrame([]), {}

    print("✅ 当前个体方案验证通过")
    df_enriched = pd.DataFrame(df_enriched)

    return vehicle_schedule, total_cost, remaining_passengers, remaining_freights, failure_records, df_enriched, module_adjustment_ranges


def calculate_module_adjustment_range(current_p_modules, current_f_modules, onboard_p, onboard_f,
                                      potential_p_demand, potential_f_demand, parameters):
    """
    计算单个站点的模块调整范围，考虑总模块数硬约束

    Args:
        current_p_modules: 当前乘客模块数
        current_f_modules: 当前货物模块数
        onboard_p: 下车后在车乘客数
        onboard_f: 下车后在车货物数
        potential_p_demand: 潜在乘客需求
        potential_f_demand: 潜在货物需求
        parameters: 系统参数

    Returns:
        dict: 包含各模块类型调整范围的字典
    """

    # 计算最少需要的模块数（保证在车乘客/货物有位置）
    min_required_p_modules = math.ceil(onboard_p / parameters["passenger_per_module"]) if onboard_p > 0 else 0
    min_required_f_modules = math.ceil(onboard_f / parameters["freight_per_module"]) if onboard_f > 0 else 0

    # 计算最大有用的模块数（考虑潜在需求）
    max_useful_p_modules = math.ceil((onboard_p + potential_p_demand) / parameters["passenger_per_module"])
    max_useful_f_modules = math.ceil((onboard_f + potential_f_demand) / parameters["freight_per_module"])

    # 考虑系统约束
    max_total_modules = parameters.get("max_modules", 5)  # 使用配置中的max_modules
    min_total_modules = 1  # 硬约束：至少1个模块

    current_total = current_p_modules + current_f_modules

    # 计算乘客模块调整范围
    p_min_decrease = max(0, current_p_modules - max_useful_p_modules)  # 可以减少到有用的最大值
    p_max_decrease = max(0, current_p_modules - min_required_p_modules)  # 最多减少到最少需要的模块数

    # 考虑总模块数下限约束
    if current_total - p_max_decrease < min_total_modules:
        p_max_decrease = max(0, current_total - min_total_modules)

    p_max_increase = max(0, min(
        max_total_modules - current_total,  # 总模块数上限约束
        max_useful_p_modules - current_p_modules  # 有用性限制
    ))

    # 计算货物模块调整范围
    f_min_decrease = max(0, current_f_modules - max_useful_f_modules)
    f_max_decrease = max(0, current_f_modules - min_required_f_modules)

    # 考虑总模块数下限约束
    if current_total - f_max_decrease < min_total_modules:
        f_max_decrease = max(0, current_total - min_total_modules)

    f_max_increase = max(0, min(
        max_total_modules - current_total,  # 总模块数上限约束
        max_useful_f_modules - current_f_modules  # 有用性限制
    ))

    # 确保调整后的总模块数在允许范围内
    p_suggested_min = max(0, current_p_modules - p_max_decrease)
    p_suggested_max = min(max_total_modules - min_required_f_modules, current_p_modules + p_max_increase)

    f_suggested_min = max(0, current_f_modules - f_max_decrease)
    f_suggested_max = min(max_total_modules - min_required_p_modules, current_f_modules + f_max_increase)

    return {
        "passenger_modules": {
            "current": current_p_modules,
            "min_required": min_required_p_modules,
            "max_useful": max_useful_p_modules,
            "min_decrease": p_min_decrease,
            "max_decrease": p_max_decrease,
            "max_increase": p_max_increase,
            "suggested_range": (p_suggested_min, p_suggested_max)
        },
        "freight_modules": {
            "current": current_f_modules,
            "min_required": min_required_f_modules,
            "max_useful": max_useful_f_modules,
            "min_decrease": f_min_decrease,
            "max_decrease": f_max_decrease,
            "max_increase": f_max_increase,
            "suggested_range": (f_suggested_min, f_suggested_max)
        },
        "total_modules": {
            "current": current_total,
            "min_constraint": min_total_modules,
            "max_constraint": max_total_modules,
            "valid_range": (min_total_modules, max_total_modules)
        },
        "capacity_utilization": {
            "passenger_utilization": onboard_p / (current_p_modules * parameters["passenger_per_module"]) if current_p_modules > 0 else 0,
            "freight_utilization": onboard_f / (current_f_modules * parameters["freight_per_module"]) if current_f_modules > 0 else 0,
        },
        "demand_coverage": {
            "passenger_coverage": min(1.0, (current_p_modules * parameters["passenger_per_module"] - onboard_p) / potential_p_demand) if potential_p_demand > 0 else 1.0,
            "freight_coverage": min(1.0, (current_f_modules * parameters["freight_per_module"] - onboard_f) / potential_f_demand) if potential_f_demand > 0 else 1.0,
        }
    }

def convert_demand_matrix_to_dataframe(demand_matrix):
    data = []
    for origin in demand_matrix:
        for destination in demand_matrix[origin]:
            for timestamp in demand_matrix[origin][destination]:
                demand_value = demand_matrix[origin][destination][timestamp]
                if demand_value != 0:
                    data.append({
                        "起点": origin,
                        "终点": destination,
                        "时间戳": timestamp,
                        "需求量": demand_value
                    })
    return pd.DataFrame(data)


def collect_vehicle_info(individual, parameters, passenger_demand_up, passenger_demand_down, freight_demand_up,
                         freight_demand_down):
    vehicle_schedule = {}
    all_vehicles = []

    for direction in ["up", "down"]:
        df = reconstruct_schedule_dataframe(individual, parameters, direction)
        # df = individual[direction]["df_schedule"]
        offset = 0 if direction == "up" else parameters["up_station_count"]
        num_stations = parameters["up_station_count"]
        vid_offset = 0 if direction == "up" else 0

        for vid in df["车辆ID"].unique():
            # print('vid_type:', type(vid))
            vid = int(vid)
            # print('vid_type:', type(vid))
            # print('vid_offset_type:', type(vid_offset))
            global_vid = vid + vid_offset
            vehicle_data = df[df["车辆ID"] == vid].sort_values(by="到达时间")
            vehicle_schedule[global_vid] = {"stations": {}}

            for _, row in vehicle_data.iterrows():
                station_id = row["站点ID"]
                vehicle_schedule[global_vid]["stations"][station_id] = {
                    "timestamp": row["到达时间"],
                    "passenger_modules": row["乘客模块"],
                    "freight_modules": row["货物模块"],
                    "delta_p": row["乘客增量"],
                    "delta_f": row["货物增量"],
                    "total_modules": row["总模块数量"]
                }

            all_vehicles.append({
                "vid": vid,
                "global_vid": global_vid,
                "direction": direction,
                "dispatch": individual[direction]["vehicle_dispatch"][vid],
                "initial_alloc": individual[direction]["initial_allocation"][vid],
                "adjustments": individual[direction]["module_adjustments"].get(vid, {}),
                "station_offset": offset,
                "num_stations": num_stations,
                "passenger_demand": passenger_demand_up if direction == "up" else passenger_demand_down,
                "freight_demand": freight_demand_up if direction == "up" else freight_demand_down
            })

    return all_vehicles, vehicle_schedule, None