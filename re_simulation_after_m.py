# re_simulation_after_mutate.py
from simulation_generate import IntegratedBusModuleSystem, collect_vehicle_info, simulate_and_evaluate_individual
import copy
import pandas as pd
import uuid

# 基于simulate_with_integrated_module_system的改进变异重仿真
def simulate_after_module_mutation_v2(individual, parameters, global_demand_data,
                                      passenger_demand_up, passenger_demand_down,
                                      freight_demand_up, freight_demand_down,
                                      direction, vehicle_id, mutated_station_id):
    """
    基于simulate_with_integrated_module_system的改进变异重仿真

    核心改进：
    1. 复用完整仿真函数的核心逻辑
    2. 智能分割：变异前保持原方案，变异后重新优化
    3. 状态传递：正确处理跨班次的模块和乘客状态

    Args:
        individual: 个体染色体
        parameters: 系统参数
        global_demand_data: 全局需求数据
        passenger_demand_up/down: 乘客需求数据
        freight_demand_up/down: 货物需求数据
        direction: 变异方向 ("up" 或 "down")
        vehicle_id: 变异车辆ID
        mutated_station_id: 变异站点ID

    Returns:
        updated_individual: 更新后的个体
        simulation_results: 详细的仿真结果
    """

    # 初始化集成系统
    module_system = IntegratedBusModuleSystem(
        alpha=parameters.get('alpha', 0),
        beta=parameters.get('beta', 5),
        C_p=parameters.get('passenger_per_module', 10),
        C_f=parameters.get('freight_per_module', 8)
    )

    # 准备需求矩阵的独立副本
    a_matrix_p_up = copy.deepcopy(global_demand_data["a_matrix_p_up"])
    a_matrix_f_up = copy.deepcopy(global_demand_data["a_matrix_f_up"])
    a_matrix_p_down = copy.deepcopy(global_demand_data["a_matrix_p_down"])
    a_matrix_f_down = copy.deepcopy(global_demand_data["a_matrix_f_down"])

    # 获取车辆信息
    all_vehicles, vehicle_schedule, _ = collect_vehicle_info(
        individual, parameters, passenger_demand_up, passenger_demand_down, freight_demand_up, freight_demand_down
    )

    # 按到达时间排序所有车辆
    all_vehicles.sort(key=lambda x: x["dispatch"]["arrival_time"])

    # 找到变异车辆的位置
    mutated_vehicle_index = None
    for idx, vehicle in enumerate(all_vehicles):
        if vehicle["direction"] == direction and vehicle["vid"] == vehicle_id:
            mutated_vehicle_index = idx
            break

    if mutated_vehicle_index is None:
        raise ValueError(f"未找到变异车辆: {direction}-{vehicle_id}")

    # 初始化结果记录
    simulation_results = {
        "pre_mutation_cost": 0,
        "post_mutation_cost": 0,
        "total_cost_increment": 0,
        "failure_records": [],
        "module_analysis_records": {},
        "df_enriched": [],
        "station_states": {}
    }

    # 创建更新后的个体副本
    updated_individual = copy.deepcopy(individual)

    # 初始化全局状态
    station_module_stock = {sid: {"modules": 0} for sid in range(parameters["up_station_count"] * 2)}
    max_simulation_time = 0

    print(f"🚌 开始智能重仿真: 变异车辆{vehicle_id}({direction}), 变异站点{mutated_station_id}")
    print(f"📊 总车辆数: {len(all_vehicles)}, 变异车辆位置: {mutated_vehicle_index}")

    # === 第一阶段：仿真变异车辆之前的所有车辆 ===
    print("\n📍 第一阶段：保持原方案仿真变异前车辆")

    pre_mutation_cost = 0

    for vehicle_idx in range(mutated_vehicle_index):
        vehicle = all_vehicles[vehicle_idx]

        cost, updated_states = simulate_vehicle_with_original_plan(
            vehicle, updated_individual, parameters,
            a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
            station_module_stock, module_system, simulation_results
        )

        pre_mutation_cost += cost
        max_simulation_time = max(max_simulation_time, updated_states["last_departure_time"])

        print(f"  ✅ 车辆{vehicle['global_vid']}({vehicle['direction']}) 完成，成本: {cost:.2f}")

    simulation_results["pre_mutation_cost"] = pre_mutation_cost

    # === 第二阶段：变异车辆的智能重仿真 ===
    print("\n📍 第二阶段：变异车辆智能重仿真")

    mutated_vehicle = all_vehicles[mutated_vehicle_index]

    mutation_cost, updated_states = simulate_mutated_vehicle_intelligent(
        mutated_vehicle, updated_individual, parameters,
        a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
        station_module_stock, module_system, simulation_results,
        mutated_station_id
    )

    max_simulation_time = max(max_simulation_time, updated_states["last_departure_time"])

    print(f"  ✅ 变异车辆{mutated_vehicle['global_vid']} 完成，成本: {mutation_cost:.2f}")

    # === 第三阶段：重新优化后续车辆 ===
    print("\n📍 第三阶段：重新优化后续车辆")

    post_mutation_cost = mutation_cost

    for vehicle_idx in range(mutated_vehicle_index + 1, len(all_vehicles)):
        vehicle = all_vehicles[vehicle_idx]

        cost, updated_states = simulate_vehicle_with_reoptimization(
            vehicle, updated_individual, parameters,
            a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
            station_module_stock, module_system, simulation_results
        )

        post_mutation_cost += cost
        max_simulation_time = max(max_simulation_time, updated_states["last_departure_time"])

        print(f"  ✅ 车辆{vehicle['global_vid']}({vehicle['direction']}) 重新优化完成，成本: {cost:.2f}")

    # === 计算未服务需求成本 ===
    unserved_cost = calculate_unserved_demand_cost(
        a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
        max_simulation_time, parameters
    )

    # post_mutation_cost += unserved_cost

    # ==================== 修改/新增逻辑：开始 ====================
    print("🔄 正在对更新后的个体进行最终的确定性评估以获取准确成本构成...")

    # 对局部变异后最终确定的 updated_individual，调用一次完整的、确定性的评估函数
    # 这将确保我们获得与新基因完全匹配的总成本和详细成本构成
    (
        _,  # vehicle_schedule
        final_total_cost,
        _,  # remaining_passengers
        _,  # remaining_freights
        failure_records,  # failure_records
        _,  # df_enriched
        final_module_analysis_records,
        final_cost_components
    ) = simulate_and_evaluate_individual(
        updated_individual, parameters, global_demand_data,
        passenger_demand_up, passenger_demand_down,
        freight_demand_up, freight_demand_down
    )

    # 使用这次评估得到的、最准确的结果来更新 simulation_results
    simulation_results["post_mutation_cost"] = final_total_cost  # 使用更精确的总成本
    simulation_results["module_analysis_records"] = final_module_analysis_records
    simulation_results["failure_records"].extend(failure_records) # 合并失败记录
    simulation_results["cost_components"] = final_cost_components # <--- 最关键的新增返回数据
    print("cost_components：", simulation_results["cost_components"])

    print(f"\n ✅ 智能重仿真及最终评估完成")
    print(f"   💰 变异前成本: {pre_mutation_cost:.2f}")
    print(f"   💰 变异后精确成本: {final_total_cost:.2f}")

    # 返回更新后的个体和包含了详细成本的仿真结果
    return updated_individual, simulation_results
    # ==================== 修改/新增逻辑：结束 ====================

    # # 更新结果
    # simulation_results["post_mutation_cost"] = post_mutation_cost
    # simulation_results["total_cost_increment"] = post_mutation_cost - pre_mutation_cost
    # simulation_results["df_enriched"] = pd.DataFrame(simulation_results["df_enriched"])
    #
    # print(f"\n ✅ 智能重仿真完成")
    # print(f"   💰 变异前成本: {pre_mutation_cost:.2f}")
    # print(f"   💰 变异后成本: {post_mutation_cost:.2f}")
    # print(f"   📊 成本变化: {simulation_results['total_cost_increment']:.2f}")
    #
    # return updated_individual, simulation_results


# 使用原始调度计划仿真车辆（变异前车辆使用此方法）
def simulate_vehicle_with_original_plan(vehicle, individual, parameters,
                                        a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
                                        station_module_stock, module_system, simulation_results):
    """
    使用原始调度计划仿真车辆（变异前车辆使用此方法）
    """

    direction = vehicle["direction"]
    vid = vehicle["vid"]
    global_vid = vehicle["global_vid"]
    offset = vehicle["station_offset"]
    num_stations = vehicle["num_stations"]

    # 选择对应的需求矩阵
    a_matrix_p = a_matrix_p_up if direction == "up" else a_matrix_p_down
    a_matrix_f = a_matrix_f_up if direction == "up" else a_matrix_f_down

    # 获取原始调度方案
    original_adjustments = individual[direction].get("module_adjustments", {}).get(vid, {})

    # 初始化车辆状态
    initial_allocation = individual[direction]["initial_allocation"][vid]
    current_p_modules = initial_allocation["passenger_modules"]
    current_f_modules = initial_allocation["freight_modules"]

    onboard_passengers = {}
    onboard_freight = {}
    total_cost = 0
    arrival_time = vehicle["dispatch"]["arrival_time"]

    # 逐站点仿真
    for sid in range(num_stations):
        station_id = sid + offset

        # 更新到达时间
        if sid > 0:
            arrival_time += parameters["t_s_s1"]

        # 应用原始模块调整方案
        delta_p = original_adjustments.get(station_id, {}).get("delta_p", 0)
        delta_f = original_adjustments.get(station_id, {}).get("delta_f", 0)

        adjusted_p_modules = current_p_modules + delta_p
        adjusted_f_modules = current_f_modules + delta_f

        # 验证调整方案
        if not validate_module_adjustment(
                onboard_passengers, onboard_freight, station_id,
                adjusted_p_modules, adjusted_f_modules, parameters, station_module_stock
        ):
            simulation_results["failure_records"].append({
                "station_id": station_id,
                "vehicle_id": global_vid,
                "timestamp": arrival_time,
                "type": "infeasible_original_plan",
                "message": f"原始调度方案不可行"
            })
            return float('inf'), {"last_departure_time": arrival_time}

        # 执行站点仿真
        station_cost, station_state = execute_station_simulation_core(
            station_id, arrival_time, onboard_passengers, onboard_freight,
            adjusted_p_modules, adjusted_f_modules,
            a_matrix_p, a_matrix_f, num_stations + offset, parameters,
            station_module_stock, delta_p, delta_f
        )

        total_cost += station_cost
        current_p_modules = adjusted_p_modules
        current_f_modules = adjusted_f_modules

        # 记录详细信息
        record_station_details(simulation_results, global_vid, station_id, direction,
                               arrival_time, station_state, "original_plan")

    return total_cost, {"last_departure_time": arrival_time}


# 智能仿真变异车辆：变异站点前保持原方案，变异站点后重新优化
def simulate_mutated_vehicle_intelligent(vehicle, updated_individual, parameters,
                                         a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
                                         station_module_stock, module_system, simulation_results,
                                         mutated_station_id):
    """
    智能仿真变异车辆：变异站点前保持原方案，变异站点后重新优化
    """

    direction = vehicle["direction"]
    vid = vehicle["vid"]
    global_vid = vehicle["global_vid"]
    offset = vehicle["station_offset"]
    num_stations = vehicle["num_stations"]

    # 选择对应的需求矩阵
    a_matrix_p = a_matrix_p_up if direction == "up" else a_matrix_p_down
    a_matrix_f = a_matrix_f_up if direction == "up" else a_matrix_f_down

    # 获取原始和更新后的调度方案
    original_adjustments = updated_individual[direction].get("module_adjustments", {}).get(vid, {})

    # 初始化车辆状态
    initial_allocation = updated_individual[direction]["initial_allocation"][vid]
    current_p_modules = initial_allocation["passenger_modules"]
    current_f_modules = initial_allocation["freight_modules"]

    onboard_passengers = {}
    onboard_freight = {}
    total_cost = 0
    arrival_time = vehicle["dispatch"]["arrival_time"]

    # 确保module_adjustments结构存在
    if vid not in updated_individual[direction]["module_adjustments"]:
        updated_individual[direction]["module_adjustments"][vid] = {}

    # 逐站点智能仿真
    for sid in range(num_stations):
        station_id = sid + offset

        # 更新到达时间
        if sid > 0:
            arrival_time += parameters["t_s_s1"]

        if station_id < mutated_station_id:
            # === 变异站点之前：使用原始调度方案 ===
            delta_p = original_adjustments.get(station_id, {}).get("delta_p", 0)
            delta_f = original_adjustments.get(station_id, {}).get("delta_f", 0)

            adjusted_p_modules = current_p_modules + delta_p
            adjusted_f_modules = current_f_modules + delta_f

            # 验证调整方案
            if not validate_module_adjustment(
                    onboard_passengers, onboard_freight, station_id,
                    adjusted_p_modules, adjusted_f_modules, parameters, station_module_stock
            ):
                simulation_results["failure_records"].append({
                    "station_id": station_id,
                    "vehicle_id": global_vid,
                    "timestamp": arrival_time,
                    "type": "infeasible_pre_mutation",
                    "message": f"变异前调度方案不可行"
                })
                return float('inf'), {"last_departure_time": arrival_time}

            # 执行站点仿真
            station_cost, station_state = execute_station_simulation_core(
                station_id, arrival_time, onboard_passengers, onboard_freight,
                adjusted_p_modules, adjusted_f_modules,
                a_matrix_p, a_matrix_f, num_stations + offset, parameters,
                station_module_stock, delta_p, delta_f
            )

            record_type = "pre_mutation"

        else:
            # === 变异站点及之后：重新优化调度方案 ===

            # 分析当前状态并重新计算模块需求
            module_analysis = analyze_station_requirements(
                station_id, arrival_time, onboard_passengers, onboard_freight,
                current_p_modules, current_f_modules, station_module_stock,
                a_matrix_p, a_matrix_f, num_stations + offset, parameters, module_system
            )

            if station_id == mutated_station_id:
                # 变异站点：使用变异后的调整值
                delta_p = updated_individual[direction]["module_adjustments"][vid][station_id].get("delta_p", 0)
                delta_f = updated_individual[direction]["module_adjustments"][vid][station_id].get("delta_f", 0)
            else:
                # 其他站点：基于分析结果重新生成最优调整
                _, _, delta_p, delta_f = module_system.generate_feasible_module_allocation(module_analysis)

                # 更新染色体
                updated_individual[direction]["module_adjustments"][vid][station_id] = {
                    "delta_p": delta_p,
                    "delta_f": delta_f
                }

            adjusted_p_modules = current_p_modules + delta_p
            adjusted_f_modules = current_f_modules + delta_f

            # 验证调整方案
            if not validate_module_adjustment(
                    onboard_passengers, onboard_freight, station_id,
                    adjusted_p_modules, adjusted_f_modules, parameters, station_module_stock
            ):
                simulation_results["failure_records"].append({
                    "station_id": station_id,
                    "vehicle_id": global_vid,
                    "timestamp": arrival_time,
                    "type": "infeasible_after_mutation",
                    "message": f"变异后调整方案不可行"
                })
                return float('inf'), {"last_departure_time": arrival_time}

            # 执行站点仿真
            station_cost, station_state = execute_station_simulation_core(
                station_id, arrival_time, onboard_passengers, onboard_freight,
                adjusted_p_modules, adjusted_f_modules,
                a_matrix_p, a_matrix_f, num_stations + offset, parameters,
                station_module_stock, delta_p, delta_f
            )

            # 记录模块分析结果
            simulation_results["module_analysis_records"][f"{global_vid}_{station_id}"] = module_analysis

            record_type = "post_mutation" if station_id == mutated_station_id else "reoptimized"

        total_cost += station_cost
        current_p_modules = adjusted_p_modules
        current_f_modules = adjusted_f_modules

        # 记录详细信息
        record_station_details(simulation_results, global_vid, station_id, direction,
                               arrival_time, station_state, record_type)

    return total_cost, {"last_departure_time": arrival_time}


# 完全重新优化车辆调度（变异后车辆使用此方法）
def simulate_vehicle_with_reoptimization(vehicle, updated_individual, parameters,
                                         a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
                                         station_module_stock, module_system, simulation_results):
    """
    完全重新优化车辆调度（变异后车辆使用此方法）
    """

    direction = vehicle["direction"]
    vid = vehicle["vid"]
    global_vid = vehicle["global_vid"]
    offset = vehicle["station_offset"]
    num_stations = vehicle["num_stations"]

    # 选择对应的需求矩阵
    a_matrix_p = a_matrix_p_up if direction == "up" else a_matrix_p_down
    a_matrix_f = a_matrix_f_up if direction == "up" else a_matrix_f_down

    # 初始化车辆状态
    initial_allocation = updated_individual[direction]["initial_allocation"][vid]
    current_p_modules = initial_allocation["passenger_modules"]
    current_f_modules = initial_allocation["freight_modules"]

    onboard_passengers = {}
    onboard_freight = {}
    total_cost = 0
    arrival_time = vehicle["dispatch"]["arrival_time"]

    # 确保module_adjustments结构存在
    if vid not in updated_individual[direction]["module_adjustments"]:
        updated_individual[direction]["module_adjustments"][vid] = {}

    # 全站点重新优化
    for sid in range(num_stations):
        station_id = sid + offset

        # 更新到达时间
        if sid > 0:
            arrival_time += parameters["t_s_s1"]

        # 分析当前状态并重新计算模块需求
        module_analysis = analyze_station_requirements(
            station_id, arrival_time, onboard_passengers, onboard_freight,
            current_p_modules, current_f_modules, station_module_stock,
            a_matrix_p, a_matrix_f, num_stations + offset, parameters, module_system
        )

        # 生成最优调整方案（优化：优先满足更多等待需求）
        _, _, delta_p, delta_f = optimize_module_allocation(module_analysis, parameters)

        # 更新染色体
        updated_individual[direction]["module_adjustments"][vid][station_id] = {
            "delta_p": delta_p,
            "delta_f": delta_f
        }

        adjusted_p_modules = current_p_modules + delta_p
        adjusted_f_modules = current_f_modules + delta_f

        # 验证调整方案
        if not validate_module_adjustment(
                onboard_passengers, onboard_freight, station_id,
                adjusted_p_modules, adjusted_f_modules, parameters, station_module_stock
        ):
            simulation_results["failure_records"].append({
                "station_id": station_id,
                "vehicle_id": global_vid,
                "timestamp": arrival_time,
                "type": "infeasible_reoptimized",
                "message": f"重新优化调度方案不可行"
            })
            return float('inf'), {"last_departure_time": arrival_time}

        # 执行站点仿真
        station_cost, station_state = execute_station_simulation_core(
            station_id, arrival_time, onboard_passengers, onboard_freight,
            adjusted_p_modules, adjusted_f_modules,
            a_matrix_p, a_matrix_f, num_stations + offset, parameters,
            station_module_stock, delta_p, delta_f
        )

        total_cost += station_cost
        current_p_modules = adjusted_p_modules
        current_f_modules = adjusted_f_modules

        # 记录模块分析和详细信息
        simulation_results["module_analysis_records"][f"{global_vid}_{station_id}"] = module_analysis
        record_station_details(simulation_results, global_vid, station_id, direction,
                               arrival_time, station_state, "fully_reoptimized")

    return total_cost, {"last_departure_time": arrival_time}


# 核心站点仿真逻辑（复用主仿真函数的核心部分）
def execute_station_simulation_core(station_id, arrival_time, onboard_passengers, onboard_freight,
                                    p_modules, f_modules, a_matrix_p, a_matrix_f, max_station_id,
                                    parameters, station_module_stock, delta_p, delta_f):
    """
    核心站点仿真逻辑（复用主仿真函数的核心部分）
    """

    # 记录调整前状态
    onboard_p_before = sum(sum(p.values()) for p in onboard_passengers.values())
    onboard_f_before = sum(sum(f.values()) for f in onboard_freight.values())

    # 1. 下车操作
    alighted_p = sum(sum(p_dict.values()) for dest, p_dict in onboard_passengers.items() if dest == station_id)
    alighted_f = sum(sum(f_dict.values()) for dest, f_dict in onboard_freight.items() if dest == station_id)

    onboard_passengers.pop(station_id, None)
    onboard_freight.pop(station_id, None)

    # 下车后状态
    onboard_p_after = sum(sum(p.values()) for p in onboard_passengers.values())
    onboard_f_after = sum(sum(f.values()) for f in onboard_freight.values())

    # 2. 更新站点模块库存
    station_module_stock_before = station_module_stock[station_id]["modules"]
    station_module_stock[station_id]["modules"] -= (delta_p + delta_f)
    station_module_stock_after = station_module_stock[station_id]["modules"]

    # 3. 上车操作（复用主仿真的上车逻辑）
    adjusted_p_capacity = p_modules * parameters["passenger_per_module"]
    adjusted_f_capacity = f_modules * parameters["freight_per_module"]

    available_p_capacity = adjusted_p_capacity - onboard_p_after
    available_f_capacity = adjusted_f_capacity - onboard_f_after

    boarded_p = 0
    boarded_f = 0
    served_passenger_waiting_time = 0
    served_freight_waiting_time = 0

    # 乘客上车
    if available_p_capacity > 0:
        for s_dest in range(station_id + 1, max_station_id):
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

    # 货物上车
    if available_f_capacity > 0:
        for s_dest in range(station_id + 1, max_station_id):
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

    # 计算站点成本
    station_cost = (served_passenger_waiting_time * parameters["passenger_waiting_cost"] +
                    served_freight_waiting_time * parameters["freight_waiting_cost"])

    # 返回成本和状态信息
    station_state = {
        "onboard_p_before": onboard_p_before,
        "onboard_f_before": onboard_f_before,
        "alighted_p": alighted_p,
        "alighted_f": alighted_f,
        "onboard_p_after": onboard_p_after,
        "onboard_f_after": onboard_f_after,
        "boarded_p": boarded_p,
        "boarded_f": boarded_f,
        "adjusted_p_modules": p_modules,
        "adjusted_f_modules": f_modules,
        "delta_p": delta_p,
        "delta_f": delta_f,
        "station_cost": station_cost,
        "station_module_stock_before": station_module_stock_before,
        "station_module_stock_after": station_module_stock_after
    }

    return station_cost, station_state


# 分析站点的模块需求（复用主仿真的分析逻辑）
def analyze_station_requirements(station_id, arrival_time, onboard_passengers, onboard_freight,
                                 current_p_modules, current_f_modules, station_module_stock,
                                 a_matrix_p, a_matrix_f, max_station_id, parameters, module_system):
    """
    分析站点的模块需求（复用主仿真的分析逻辑）
    """

    # 计算车辆状态
    onboard_p_before = sum(sum(p.values()) for p in onboard_passengers.values())
    onboard_f_before = sum(sum(f.values()) for f in onboard_freight.values())

    # 计算下车需求
    alighted_p = sum(sum(p_dict.values()) for dest, p_dict in onboard_passengers.items() if dest == station_id)
    alighted_f = sum(sum(f_dict.values()) for dest, f_dict in onboard_freight.items() if dest == station_id)

    # 计算下车后状态
    onboard_p_after = onboard_p_before - alighted_p
    onboard_f_after = onboard_f_before - alighted_f

    # 计算等待需求
    waiting_p = 0
    waiting_f = 0

    for s_dest in range(station_id + 1, max_station_id):
        for t in range(arrival_time + 1):
            if (station_id in a_matrix_p and s_dest in a_matrix_p[station_id] and
                    t in a_matrix_p[station_id][s_dest]):
                waiting_p += a_matrix_p[station_id][s_dest][t]

            if (station_id in a_matrix_f and s_dest in a_matrix_f[station_id] and
                    t in a_matrix_f[station_id][s_dest]):
                waiting_f += a_matrix_f[station_id][s_dest][t]

    # 获取站点模块库存
    store_modules = station_module_stock[station_id]["modules"]

    # 调用主仿真的模块需求分析函数
    module_analysis = module_system.calculate_station_module_requirements(
        n=current_p_modules + current_f_modules,
        k=station_id,
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

    return module_analysis


# 验证模块调整后是否能满足基本的容量约束和库存约束
def validate_module_adjustment(onboard_passengers, onboard_freight, station_id,
                               adjusted_p_modules, adjusted_f_modules, parameters, station_module_stock):
    """
    验证模块调整后是否能满足基本的容量约束和库存约束
    """

    # 计算下车后在车需求
    onboard_p_after = sum(sum(p.values()) for p in onboard_passengers.values() if p != station_id)
    onboard_f_after = sum(sum(f.values()) for f in onboard_freight.values() if f != station_id)

    # 计算调整后容量
    adjusted_p_capacity = adjusted_p_modules * parameters["passenger_per_module"]
    adjusted_f_capacity = adjusted_f_modules * parameters["freight_per_module"]

    # 检查容量约束
    if onboard_p_after > adjusted_p_capacity or onboard_f_after > adjusted_f_capacity:
        return False

    # 检查模块总数约束
    total_modules = adjusted_p_modules + adjusted_f_modules
    if total_modules < parameters.get("alpha", 0) or total_modules > parameters.get("beta", 5):
        return False

    # 检查站点库存约束
    station_stock = station_module_stock[station_id]["modules"]
    if station_stock < parameters.get("min_modules_stock", 0) or station_stock > parameters.get("max_modules_stock", float('inf')):
        return False

    return True


# 计算未服务需求的等待时间成本
def calculate_unserved_demand_cost(a_matrix_p_up, a_matrix_f_up, a_matrix_p_down, a_matrix_f_down,
                                   max_simulation_time, parameters):
    """
    计算未服务需求的等待时间成本
    """
    unserved_passenger_waiting_cost = 0
    unserved_freight_waiting_cost = 0

    for a_matrix_p in [a_matrix_p_up, a_matrix_p_down]:
        for s in a_matrix_p:
            for s_dest in a_matrix_p[s]:
                for t in a_matrix_p[s][s_dest]:
                    remaining_p = a_matrix_p[s][s_dest][t]
                    if remaining_p > 0:
                        waiting_time = max_simulation_time - t
                        unserved_passenger_waiting_cost += remaining_p * waiting_time

    for a_matrix_f in [a_matrix_f_up, a_matrix_f_down]:
        for s in a_matrix_f:
            for s_dest in a_matrix_f[s]:
                for t in a_matrix_f[s][s_dest]:
                    remaining_f = a_matrix_f[s][s_dest][t]
                    if remaining_f > 0:
                        waiting_time = max_simulation_time - t
                        unserved_freight_waiting_cost += remaining_f * waiting_time

    total_unserved_cost = (unserved_passenger_waiting_cost * parameters["passenger_waiting_cost"] +
                           unserved_freight_waiting_cost * parameters["freight_waiting_cost"])

    return total_unserved_cost


# 记录站点仿真详细信息
def record_station_details(simulation_results, vehicle_id, station_id, direction,
                           arrival_time, station_state, record_type):
    """
    记录站点仿真详细信息
    """
    simulation_results["df_enriched"].append({
        "车辆ID": vehicle_id,
        "站点ID": station_id,
        "方向": direction,
        "到达时间": arrival_time,
        "记录类型": record_type,
        "调整前乘客模块": station_state["adjusted_p_modules"] - station_state["delta_p"],
        "调整前货物模块": station_state["adjusted_f_modules"] - station_state["delta_f"],
        "调整后乘客模块": station_state["adjusted_p_modules"],
        "调整后货物模块": station_state["adjusted_f_modules"],
        "模块增量_乘客": station_state["delta_p"],
        "模块增量_货物": station_state["delta_f"],
        "下车前在车乘客": station_state["onboard_p_before"],
        "下车前在车货物": station_state["onboard_f_before"],
        "下车乘客": station_state["alighted_p"],
        "下车货物": station_state["alighted_f"],
        "下车后在车乘客": station_state["onboard_p_after"],
        "下车后在车货物": station_state["onboard_f_after"],
        "上车乘客": station_state["boarded_p"],
        "上车货物": station_state["boarded_f"],
        "站点成本": station_state["station_cost"],
        "站点进站前库存": station_state["station_module_stock_before"],
        "站点出站后库存": station_state["station_module_stock_after"]
    })


# 优化模块分配，优先满足更多等待需求
def optimize_module_allocation(module_analysis, parameters):
    """
    优化模块分配，优先满足更多等待需求
    """
    current_p_modules = module_analysis['station_info']['current_p_modules']
    current_f_modules = module_analysis['station_info']['current_f_modules']
    delta_p_min, delta_p_max = module_analysis['adjustment_ranges']['passenger_modules']['delta_range']
    delta_f_min, delta_f_max = module_analysis['adjustment_ranges']['freight_modules']['delta_range']
    waiting_pass = module_analysis['passenger_analysis']['waiting']
    waiting_cargo = module_analysis['freight_analysis']['waiting']

    # 优先满足等待需求较大的模块
    if waiting_pass / parameters["passenger_per_module"] > waiting_cargo / parameters["freight_per_module"]:
        delta_p = min(delta_p_max, module_analysis['passenger_analysis']['optimal_modules'] - current_p_modules)
        delta_f = min(delta_f_max - delta_p, module_analysis['freight_analysis']['optimal_modules'] - current_f_modules)
    else:
        delta_f = min(delta_f_max, module_analysis['freight_analysis']['optimal_modules'] - current_f_modules)
        delta_p = min(delta_p_max, module_analysis['passenger_analysis']['optimal_modules'] - current_p_modules)

    p_n_k_1 = current_p_modules + delta_p
    f_n_k_1 = current_f_modules + delta_f

    return p_n_k_1, f_n_k_1, delta_p, delta_f