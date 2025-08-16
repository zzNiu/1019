# DEAP工具箱设置模块
from deap import base, creator, tools

import random

# 导入重构后的遗传算法函数
from generate_individual_with_simulation import generate_individual_with_simulation
# 【修改】从 simulation_generate 导入新的评估函数
from simulation_generate import simulate_and_evaluate_individual, simulate_with_integrated_module_system
from re_simulation_after_m import simulate_after_module_mutation_v2

# ===== 在 deap_toolbox_setup.py 顶部或合适位置 =====
cost_cache = {}  # 仅保存“本代被评估过的个体”的成本分解

def setup_deap_toolbox(parameters, global_demand_data):
    """
    设置DEAP工具箱

    Args:
        parameters: 系统参数
        global_demand_data: 全局需求数据

    Returns:
        toolbox: 配置好的DEAP工具箱
    """
    # 创建适应度类和个体类（如果还没有创建）
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", dict, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 注册个体生成函数
    def create_individual():
        """创建个体（通过仿真生成module_adjustments）"""
        individual_data = generate_individual_with_simulation(
            num_vehicles=parameters['NUM_VEHICLES'],
            max_modules=parameters['max_modules'],
            headway_range=(parameters['min_headway'], parameters['max_headway']),
            parameters=parameters,
            global_demand_data=global_demand_data
        )

        individual = creator.Individual(individual_data)

        return individual


    # 注册评估函数
    def evaluate_individual(individual):
        """评估个体适应度"""
        try:
            # 使用集成仿真系统评估个体
            (vehicle_schedule, total_cost, remaining_passengers, remaining_freights,
             failure_records, df_enriched, module_analysis_records, cost_components) = simulate_and_evaluate_individual(
                individual, parameters, global_demand_data,
                global_demand_data["passenger_demand_up"],
                global_demand_data["passenger_demand_down"],
                global_demand_data["freight_demand_up"],
                global_demand_data["freight_demand_down"]
            )

            # —— 不改染色体，只把三项成本放进缓存 ——
            cost_cache[id(individual)] = {
                "passenger_waiting_cost": float(cost_components["passenger_waiting_cost"]),
                "freight_waiting_cost": float(cost_components["freight_waiting_cost"]),
                "mav_transport_cost": float(cost_components["mav_transport_cost"]),
            }

            # 已经添加了未上车的等待时间成本计算，考虑是否添加更大的比例

            # # 如果有未完成的需求，增加惩罚
            # penalty = 0
            # if remaining_passengers > 0:
            #     penalty += remaining_passengers * parameters.get('passenger_waiting_cost', 10.0)
            # if remaining_freights > 0:
            #     penalty += remaining_freights * parameters.get('freight_waiting_cost', 5.0)

            fitness = total_cost

            return (fitness,), failure_records, module_analysis_records

        except Exception as e:
            print(f"评估个体时出错: {e}")
            return (float('inf'),), [], {}


    # 变异操作
    def intelligent_mutate(individual, parameters, global_demand_data, adjustment_ranges=None):
        """
        基于adjustment_ranges的智能变异函数

        Args:
            individual: 要变异的个体
            parameters: 系统参数
            global_demand_data: 全局需求数据
            adjustment_ranges: 模块调整范围信息（可选）

        Returns:
            tuple: (变异后的个体,)
        """

        headway_changed = False
        initial_allocation_changed = False
        module_adjustment_changed = False
        mutated_direction = None
        mutated_vehicle_id = None
        mutated_station_id = None

        # 1. 随机选择一种变异类型：0=初始模块配置，1=车头时距，2=模块调整
        mutate_type = random.randint(0, 2)

        if mutate_type == 0:
            # === 初始模块配置变异 ===
            direction = random.choice(["up", "down"])
            vehicle_ids = list(individual[direction]["initial_allocation"].keys())
            if vehicle_ids:
                # 随机选择一班车辆
                vehicle_id = random.choice(vehicle_ids)
                max_modules = parameters['max_modules']
                total_modules = random.randint(1, max_modules)
                passenger_modules = random.randint(0, total_modules)
                freight_modules = total_modules - passenger_modules

                # 更新到染色体上
                individual[direction]["initial_allocation"][vehicle_id] = {
                    "passenger_modules": passenger_modules,
                    "freight_modules": freight_modules
                }

                # 已经更新了修改的部分 需要完整更新染色体
                initial_allocation_changed = True

        # === 车头时距变异 ===
        elif mutate_type == 1:

            # === 车头时距变异 ===
            direction = random.choice(["up", "down"])
            vehicle_ids = list(individual[direction]["vehicle_dispatch"].keys())

            if vehicle_ids:
                vehicle_id = random.choice(vehicle_ids)
                old_hw = individual[direction]["vehicle_dispatch"][vehicle_id]["headway"]
                delta_hw = random.randint(-3, 3)
                new_hw = max(1, old_hw + delta_hw)
                individual[direction]["vehicle_dispatch"][vehicle_id]["headway"] = new_hw
                recalculate_arrival_times(individual, direction)
                headway_changed = True

        # === 模块调整变异 ===
        else:
            # === 模块调整变异 ===
            if adjustment_ranges:
                direction = random.choice(["up", "down"])
                if direction in adjustment_ranges:
                    vehicle_ids = list(adjustment_ranges[direction].keys())
                    if vehicle_ids:
                        # 选择一班车辆
                        vehicle_id = random.choice(vehicle_ids)
                        station_ids = list(adjustment_ranges[direction][vehicle_id].keys())
                        if station_ids:
                            # 选择一个站点
                            station_id = random.choice(station_ids)
                            p_range = adjustment_ranges[direction][vehicle_id][station_id].get("passenger_modules", {})
                            f_range = adjustment_ranges[direction][vehicle_id][station_id].get("freight_modules", {})

                            mutated = False
                            if p_range:
                                new_delta_p = mutate_within_bounds(p_range)
                                individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_p"] = new_delta_p
                                mutated = True

                            if f_range:
                                new_delta_f = mutate_within_bounds(f_range)
                                individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_f"] = new_delta_f
                                mutated = True

                            if mutated:
                                module_adjustment_changed = True
                                mutated_direction = direction
                                mutated_vehicle_id = vehicle_id
                                mutated_station_id = station_id

        # === 在变异结束后统一判断和更新染色体 ===
        if headway_changed or initial_allocation_changed:
            print('车头时距 or 初始模块配置 变异')
            print("\U0001f501 开始仿真以更新变异后个体的适应度与调整范围...")

            try:
                (vehicle_schedule, total_cost, remaining_passengers, remaining_freights,
                 failure_records, df_enriched, module_analysis_records, cost_components) = simulate_with_integrated_module_system(
                    individual, parameters, global_demand_data,
                    global_demand_data["passenger_demand_up"],
                    global_demand_data["passenger_demand_down"],
                    global_demand_data["freight_demand_up"],
                    global_demand_data["freight_demand_down"]
                )

                print("🧬 变异后染色体更新：正在从仿真结果中提取 module_adjustments 和 adjustment_ranges...")

                # 1. 初始化用于存储新计划的字典
                module_adjustments = {"up": {}, "down": {}}
                adjustment_ranges = {"up": {}, "down": {}}

                # 2. 遍历仿真记录，提取模块调整计划和范围
                for record in module_analysis_records:
                    vehicle_id = record['vehicle_id']
                    station_id = record['station_id']
                    direction = record['direction']
                    analysis = record['analysis']

                    # 初始化车辆记录的字典结构
                    if vehicle_id not in module_adjustments[direction]:
                        module_adjustments[direction][vehicle_id] = {}
                        adjustment_ranges[direction][vehicle_id] = {}

                    # 提取模块调整量 (delta)
                    # 'suggested_next_allocation' 是仿真中为下一站实际决定的模块数
                    if 'suggested_next_allocation' in analysis:
                        suggested = analysis['suggested_next_allocation']
                        current_p = analysis['station_info']['current_p_modules']
                        current_f = analysis['station_info']['current_f_modules']

                        # 计算并记录实际发生的模块数量变化
                        delta_p = suggested['passenger_modules'] - current_p
                        delta_f = suggested['freight_modules'] - current_f

                        module_adjustments[direction][vehicle_id][station_id] = {
                            "delta_p": delta_p,
                            "delta_f": delta_f
                        }

                    # 提取模块调整范围 (供下一次变异使用)
                    if 'adjustment_ranges' in analysis:
                        adjustment_ranges[direction][vehicle_id][station_id] = {
                            "passenger_modules": analysis['adjustment_ranges']['passenger_modules'],
                            "freight_modules": analysis['adjustment_ranges']['freight_modules']
                        }

                # 3. 将新生成的调整策略和范围完整更新到个体(染色体)中
                individual["up"]["module_adjustments"] = module_adjustments.get("up", {})
                individual["down"]["module_adjustments"] = module_adjustments.get("down", {})
                individual["adjustment_ranges"] = adjustment_ranges

                # 4. 更新适应度和失败记录
                individual.fitness.values = (total_cost,)
                individual["adjustment_ranges"] = module_analysis_records
                individual["failure_records"] = failure_records

                print(f"✅ 个体仿真成功，适应度: {total_cost}")

            except Exception as e:
                print(f"❌ 个体仿真失败: {e}")
                individual.fitness.values = (float("inf"),)

        elif module_adjustment_changed:
            print('中间站点模块调整 变异')
            print("\U0001f501 开始部分重仿真以更新变异后个体的适应度与调整范围...")

            try:
                updated_individual, simulation_results = simulate_after_module_mutation_v2(
                    individual, parameters, global_demand_data,
                    global_demand_data["passenger_demand_up"],
                    global_demand_data["passenger_demand_down"],
                    global_demand_data["freight_demand_up"],
                    global_demand_data["freight_demand_down"],
                    mutated_direction, mutated_vehicle_id, mutated_station_id
                )

                individual = updated_individual
                failure_records = simulation_results["failure_records"]
                module_analysis_records = simulation_results["module_analysis_records"]
                total_cost = simulation_results["pre_mutation_cost"] + simulation_results["post_mutation_cost"]

                individual.fitness.values = (total_cost,)
                individual["adjustment_ranges"] = module_analysis_records
                individual["failure_records"] = failure_records

                print(f"✅ 个体部分重仿真成功，适应度: {total_cost}")

            except Exception as e:
                print(f"❌ 个体部分重仿真失败: {e}")
                individual.fitness.values = (float("inf"),)

        return (individual,)

    # 变异后更新发车时间
    def recalculate_arrival_times(individual, direction):
        """重新计算发车时间"""
        current_time = 0

        # 按车辆ID排序，确保顺序正确
        vehicle_ids = sorted(individual[direction]["vehicle_dispatch"].keys())

        for vehicle_id in vehicle_ids:
            # 更新发车时间
            individual[direction]["vehicle_dispatch"][vehicle_id]["arrival_time"] = current_time
            # 累加车头时距
            headway = individual[direction]["vehicle_dispatch"][vehicle_id]["headway"]
            current_time += headway

    # 在指定范围内随机生成模块调整量
    def mutate_within_bounds(range_info):
        """
        在指定范围内随机生成模块调整量

        Args:
            range_info: 包含min和max的范围信息字典

        Returns:
            int: 在[min, max]范围内的随机调整量
        """

        min_val = range_info["min"]
        max_val = range_info["max"]

        # 直接在范围内随机生成调整量
        return random.randint(min_val, max_val)

    toolbox.register("individual", create_individual)

    # 注册种群生成函数
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)

    # 注册基本的DEAP操作
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 不考虑交叉了，只考虑变异操作
    # toolbox.register("mate", lambda ind1, ind2, params, global_data: (ind1, ind2))  # 占位符

    toolbox.register("mutate", intelligent_mutate)  # 占位符
    toolbox.register("clone", lambda ind: creator.Individual(ind.copy()) if hasattr(creator, 'Individual') else ind.copy())

    return toolbox