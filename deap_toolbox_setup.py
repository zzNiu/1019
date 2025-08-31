# DEAP工具箱设置模块
from deap import base, creator, tools

import random
import copy

# 导入重构后的遗传算法函数
from generate_individual_with_simulation import generate_individual_with_simulation
# 【修改】从 simulation_generate 导入新的评估函数
from simulation_generate import simulate_and_evaluate_individual, simulate_with_integrated_module_system
from re_simulation_after_m import simulate_after_module_mutation_v2

# ===== 在 deap_toolbox_setup.py 顶部或合适位置 =====

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

            # ==================== 修改逻辑：开始 ====================
            # 不再写入 cost_cache，而是直接将成本数据作为个体的一个属性
            print('成本写入染色体')
            individual.cost_components = {
                "passenger_waiting_cost": float(cost_components["passenger_waiting_cost"]),
                "freight_waiting_cost": float(cost_components["freight_waiting_cost"]),
                "mav_transport_cost": float(cost_components["mav_transport_cost"]),
            }
            print('individual.cost_components:', individual.cost_components)
            # ==================== 修改逻辑：结束 ====================

            # 已经添加了未上车的等待时间成本计算，考虑是否添加更大的比例

            # # 如果有未完成的需求，增加惩罚
            # penalty = 0
            # if remaining_passengers > 0:
            #     penalty += remaining_passengers * parameters.get('passenger_waiting_cost', 10.0)
            # if remaining_freights > 0:
            #     penalty += remaining_freights * parameters.get('freight_waiting_cost', 5.0)

            fitness = total_cost

            # ==================== 在这里添加转换逻辑 (开始) ====================
            # 这段代码与您在初始种群生成时使用的逻辑完全相同

            print("✅ 评估函数内部：正在将分析记录列表转换为字典结构...")

            # 从仿真结果中提取module_adjustments和adjustment_ranges
            module_adjustments = {"up": {}, "down": {}}
            adjustment_ranges = {"up": {}, "down": {}}

            # 处理仿真记录，提取模块调整信息
            for record in module_analysis_records:
                vehicle_id = record['vehicle_id']
                station_id = record['station_id']
                direction = record['direction']
                analysis = record['analysis']

                # 初始化车辆记录
                if vehicle_id not in module_adjustments[direction]:
                    module_adjustments[direction][vehicle_id] = {}
                    adjustment_ranges[direction][vehicle_id] = {}

                # 提取建议的模块分配 (这部分逻辑用于 individual["..."]["module_adjustments"])
                if 'suggested_next_allocation' in analysis:
                    suggested = analysis['suggested_next_allocation']
                    current_p = analysis['station_info']['current_p_modules']
                    current_f = analysis['station_info']['current_f_modules']

                    delta_p = suggested['passenger_modules'] - current_p
                    delta_f = suggested['freight_modules'] - current_f

                    module_adjustments[direction][vehicle_id][station_id] = {
                        "delta_p": delta_p,
                        "delta_f": delta_f
                    }

                # 将完整的分析结果字典存储起来
                adjustment_ranges[direction][vehicle_id][station_id] = analysis

            # 将生成的调整策略更新到个体内部 (这一步可选，取决于您的设计)
            # 如果您希望评估函数也更新个体的 module_adjustments，可以保留这两行
            individual["up"]["module_adjustments"] = module_adjustments["up"]
            individual["down"]["module_adjustments"] = module_adjustments["down"]

            # ==================== 添加转换逻辑 (结束) ====================

            return (fitness,), failure_records, module_analysis_records

        except Exception as e:
            print(f"评估个体时出错: {e}")
            # 对于评估失败的个体，也附上一个空的成本字典
            individual.cost_components = {}
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

        # ==================== 核心修正：开始 ====================
        # 从个体自身获取 adjustment_ranges，而不是依赖函数参数传递
        # 使用 .get() 方法可以安全地处理个体可能没有此属性的情况
        # adjustment_ranges = individual.get("adjustment_ranges")
        # ==================== 核心修正：结束 ====================

        headway_changed = False
        initial_allocation_changed = False
        module_adjustment_changed = False
        mutated_direction = None
        mutated_vehicle_id = None
        mutated_station_id = None

        # 1. 随机选择一种变异类型：0=初始模块配置，1=车头时距，2=模块调整
        mutate_type = random.randint(0, 1)

        print('mutate_type:', mutate_type)

        if mutate_type == 0:
            # === 初始模块配置变异 ===
            direction = random.choice(["up", "down"])
            vehicle_ids = list(individual[direction]["initial_allocation"].keys())
            if vehicle_ids:
                # 随机选择一班车辆
                vehicle_id = random.choice(vehicle_ids)

                print('初始模块配置变异前:' )
                print('初始模块配置变异前(乘客模块):', individual[direction]["initial_allocation"][vehicle_id]['passenger_modules'])
                print('初始模块配置变异前(货物模块):', individual[direction]["initial_allocation"][vehicle_id]['freight_modules'])

                max_modules = parameters['max_modules']
                total_modules = random.randint(1, max_modules)
                passenger_modules = random.randint(0, total_modules)
                freight_modules = total_modules - passenger_modules

                # 更新到染色体上
                individual[direction]["initial_allocation"][vehicle_id] = {
                    "passenger_modules": passenger_modules,
                    "freight_modules": freight_modules
                }

                print('初始模块配置变异后更新到染色体上:')
                print('初始模块配置变异前(乘客模块):', individual[direction]["initial_allocation"][vehicle_id]['passenger_modules'])
                print('初始模块配置变异前(货物模块):', individual[direction]["initial_allocation"][vehicle_id]['freight_modules'])

                # 已经更新了修改的部分 需要完整更新染色体
                initial_allocation_changed = True

                print('初始模块配置变异')

        # === 车头时距变异 ===
        elif mutate_type == 1:

            # === 车头时距变异 ===
            direction = random.choice(["up", "down"])
            vehicle_ids = list(individual[direction]["vehicle_dispatch"].keys())

            if vehicle_ids:
                vehicle_id = random.choice(vehicle_ids)
                old_hw = individual[direction]["vehicle_dispatch"][vehicle_id]["headway"]

                print('车头时距变异前:', old_hw)
                delta_hw = random.randint(-3, 3)
                new_hw = max(1, old_hw + delta_hw)
                individual[direction]["vehicle_dispatch"][vehicle_id]["headway"] = new_hw
                recalculate_arrival_times(individual, direction)
                headway_changed = True

                print('车头时距变异后:', new_hw)

                print('车头时距变异')

        # === 模块调整变异 ===
        # === 模块调整变异 (已修正) ===
        elif mutate_type == 2:
            if individual.get("adjustment_ranges"):
                direction = random.choice(["up", "down"])
                adjustment_ranges = individual["adjustment_ranges"]

                print('individual["adjustment_ranges"]:', individual["adjustment_ranges"])

                if direction in adjustment_ranges and adjustment_ranges[direction]:
                    vehicle_id = random.choice(list(adjustment_ranges[direction].keys()))

                    if vehicle_id in adjustment_ranges[direction] and adjustment_ranges[direction][vehicle_id]:
                        station_id = random.choice(list(adjustment_ranges[direction][vehicle_id].keys()))

                        # 1. 获取包含所有原始参数的“决策工具箱”
                        analysis_data = adjustment_ranges[direction][vehicle_id][station_id]

                        print('vehicle_id:', vehicle_id)
                        print('station_id:', station_id)
                        print('analysis_data:', analysis_data)

                        # 2. 提取原始参数用于重新计算
                        p_n_k = analysis_data['station_info']['current_p_modules']
                        f_n_k = analysis_data['station_info']['current_f_modules']
                        total_max = analysis_data['module_constraints']['total_max']
                        p_min = analysis_data['add']['passenger_modules_min']
                        f_min = analysis_data['add']['freight_modules_min']

                        # 3. 重新计算 p 的范围并生成新值
                        delta_p_min = p_min - p_n_k
                        delta_p_max = total_max - p_n_k - f_min
                        new_delta_p = random.randint(delta_p_min, delta_p_max) if delta_p_min <= delta_p_max else delta_p_min

                        # 4. 【核心联动逻辑】基于 new_delta_p，动态计算 f 的新范围
                        delta_f_min = f_min - f_n_k
                        new_delta_f_max = total_max - f_n_k - (p_n_k + new_delta_p)

                        # 5. 在新的联动范围内生成新值
                        new_delta_f = random.randint(delta_f_min, new_delta_f_max) if delta_f_min <= new_delta_f_max else delta_f_min

                        # 6. 更新个体染色体
                        # 确保路径存在
                        if vehicle_id not in individual[direction]["module_adjustments"]:
                            individual[direction]["module_adjustments"][vehicle_id] = {}
                        if station_id not in individual[direction]["module_adjustments"][vehicle_id]:
                            individual[direction]["module_adjustments"][vehicle_id][station_id] = {}

                        individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_p"] = new_delta_p
                        individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_f"] = new_delta_f

                        # 标记变异已发生，以便进行部分重仿真
                        module_adjustment_changed = True
                        mutated_direction = direction
                        mutated_vehicle_id = vehicle_id
                        mutated_station_id = station_id
                        print(f'模块调整联动变异: V:{vehicle_id}, S:{station_id}, new_delta_p:{new_delta_p}, new_delta_f:{new_delta_f}')
        # elif mutate_type == 2:
        #     # === 模块调整变异 ===
        #     if adjustment_ranges:
        #         direction = random.choice(["up", "down"])
        #         if direction in adjustment_ranges:
        #             vehicle_ids = list(adjustment_ranges[direction].keys())
        #             if vehicle_ids:
        #                 # 选择一班车辆
        #                 vehicle_id = random.choice(vehicle_ids)
        #                 station_ids = list(adjustment_ranges[direction][vehicle_id].keys())
        #                 if station_ids:
        #                     # 选择一个站点
        #                     station_id = random.choice(station_ids)
        #                     p_range = adjustment_ranges[direction][vehicle_id][station_id].get("passenger_modules", {})
        #                     f_range = adjustment_ranges[direction][vehicle_id][station_id].get("freight_modules", {})
        #
        #                     # ==================== 核心修正 ====================
        #                     # 在赋值前，确保该站点的字典路径存在
        #                     # 如果车辆的调整计划中没有这个站点（无论是最后一个还是中间的），
        #                     # 就为它创建一个空的字典档案。
        #                     if station_id not in individual[direction]["module_adjustments"][vehicle_id]:
        #                         individual[direction]["module_adjustments"][vehicle_id][station_id] = {}
        #                     # =================================================
        #
        #                     mutated = False
        #                     if p_range:
        #                         new_delta_p = mutate_within_bounds(p_range)
        #                         print('new_delta_p:', new_delta_p)
        #                         individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_p"] = new_delta_p
        #                         mutated = True
        #
        #                     if f_range:
        #                         new_delta_f = mutate_within_bounds(f_range)
        #                         print('new_delta_f:', new_delta_f)
        #                         individual[direction]["module_adjustments"][vehicle_id][station_id]["delta_f"] = new_delta_f
        #                         mutated = True
        #
        #                     if mutated:
        #                         module_adjustment_changed = True
        #                         mutated_direction = direction
        #                         mutated_vehicle_id = vehicle_id
        #                         mutated_station_id = station_id
        #
        #                         print('模块调整变异', '变异车辆：', vehicle_id, '变异站点：', station_id)

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

                        module_adjustments[direction][vehicle_id][station_id] = {
                            "delta_p": suggested['delta_p'],
                            "delta_f": suggested['delta_f']
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
                individual.cost_components = cost_components
                # individual["adjustment_ranges"] = module_analysis_records
                individual["failure_records"] = failure_records

                print(f"✅ 个体仿真成功，适应度: {total_cost}")

            except Exception as e:
                print(f"❌ 个体仿真失败: {e}")
                individual.fitness.values = (float("inf"),)

        elif module_adjustment_changed:

            print('中间站点模块调整 变异')
            print("\U0001f501 开始部分重仿真以更新变异后个体的适应度与调整范围...")

            try:
                # 1. 接收从重仿真函数返回的、包含详细成本的 simulation_results
                updated_individual, simulation_results = simulate_after_module_mutation_v2(
                    individual, parameters, global_demand_data,
                    global_demand_data["passenger_demand_up"],
                    global_demand_data["passenger_demand_down"],
                    global_demand_data["freight_demand_up"],
                    global_demand_data["freight_demand_down"],
                    mutated_direction, mutated_vehicle_id, mutated_station_id
                )

                # ==================== 修改/新增逻辑：开始 ====================
                # 2. 从 simulation_results 中提取最准确的总成本和详细成本
                total_cost = simulation_results["post_mutation_cost"]
                cost_components = simulation_results["cost_components"]
                module_analysis_records = simulation_results["module_analysis_records"]
                failure_records = simulation_results["failure_records"]

                # ==================== 新增的修正逻辑：开始 ====================
                # 创建一个新的空字典来存储正确结构的 adjustment_ranges
                new_adjustment_ranges = {"up": {}, "down": {}}

                # 遍历返回的 records 列表，重新构建嵌套字典
                for record in module_analysis_records:
                    vehicle_id = record['vehicle_id']
                    station_id = record['station_id']
                    direction = record['direction']
                    analysis = record['analysis']

                    # 初始化车辆记录的字典结构
                    if vehicle_id not in new_adjustment_ranges[direction]:
                        new_adjustment_ranges[direction][vehicle_id] = {}

                    # 将完整的 analysis 字典存入正确的位置
                    new_adjustment_ranges[direction][vehicle_id][station_id] = analysis
                # ==================== 新增的修正逻辑：结束 ====================

                # ==================== 解决方案核心逻辑：开始 ====================
                # 为了实现“就地修改”，我们不能直接用 individual = updated_individual
                # 而是要清空原始 individual 的内容，然后用新个体的内容填充它。

                # 1. 清空原始 individual 字典的内容
                individual.clear()
                # 2. 将优化后的克隆体 updated_individual 的所有内容复制过来
                individual.update(updated_individual)

                # 3. 现在，在原始 individual 对象上附加新的属性
                individual.fitness.values = (total_cost,)
                individual.cost_components = cost_components
                # individual.adjustment_ranges = module_analysis_records
                # 正确的代码
                individual.adjustment_ranges = new_adjustment_ranges
                individual.failure_records = failure_records
                # ==================== 解决方案核心逻辑：结束 ====================

                print(f"✅ 个体部分重仿真及评估成功，新适应度: {total_cost}")
                # ==================== 修改/新增逻辑：结束 ====================

            except Exception as e:
                print(f"❌ 个体部分重仿真失败: {e}")
                individual.fitness.values = (float("inf"),)
                # 评估失败时，也附上一个空的成本字典
                individual.cost_components = {}

        return (individual,)

        # elif module_adjustment_changed:
        #     print('中间站点模块调整 变异')
        #     print("\U0001f501 开始部分重仿真以更新变异后个体的适应度与调整范围...")
        #
        #     try:
        #         updated_individual, simulation_results = simulate_after_module_mutation_v2(
        #             individual, parameters, global_demand_data,
        #             global_demand_data["passenger_demand_up"],
        #             global_demand_data["passenger_demand_down"],
        #             global_demand_data["freight_demand_up"],
        #             global_demand_data["freight_demand_down"],
        #             mutated_direction, mutated_vehicle_id, mutated_station_id
        #         )
        #
        #         individual = updated_individual
        #         failure_records = simulation_results["failure_records"]
        #         module_analysis_records = simulation_results["module_analysis_records"]
        #         total_cost = simulation_results["pre_mutation_cost"] + simulation_results["post_mutation_cost"]
        #
        #         individual.fitness.values = (total_cost,)
        #         individual["adjustment_ranges"] = module_analysis_records
        #         individual["failure_records"] = failure_records
        #
        #         print(f"✅ 个体部分重仿真成功，适应度: {total_cost}")
        #
        #     except Exception as e:
        #         print(f"❌ 个体部分重仿真失败: {e}")
        #         individual.fitness.values = (float("inf"),)
        #
        # return (individual,)

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

        print("delta_range:", range_info["delta_range"])

        # 从 'delta_range' 元组中提取变化的下限和上限
        min_val = range_info["delta_range"].start
        max_val = range_info["delta_range"].stop
        # min_val, max_val = range_info["delta_range"]

        print('min_val:', min_val)
        print('max_val:', max_val)

        # min_val = range_info["min"]
        # max_val = range_info["max"]

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
    # toolbox.register("clone", lambda ind: creator.Individual(ind.copy()) if hasattr(creator, 'Individual') else ind.copy())
    # 改为深拷贝
    toolbox.register("clone", lambda ind: copy.deepcopy(ind))

    return toolbox