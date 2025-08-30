import random

# 【修改】导入新的确定性评估函数
from simulation_generate import simulate_with_integrated_module_system

def generate_individual_with_simulation(num_vehicles, max_modules, headway_range,
                                      parameters=None, global_demand_data=None):
    """
    通过仿真生成包含完整module_adjustments的个体

    Args:
        num_vehicles: 车辆数量
        max_modules: 最大模块数
        headway_range: 车头时距范围
        parameters: 系统参数
        global_demand_data: 全局需求数据

    Returns:
        individual: 包含module_adjustments和adjustment_ranges的完整个体
    """
    def generate_one_direction(direction):

        vid_offset = 0 if direction == "up" else 100
        vehicle_dispatch = {}
        current_time = 0

        # 生成车头时距
        for vid in range(num_vehicles):
            global_vid = vid + vid_offset
            headway = random.randint(headway_range[0], headway_range[1])
            vehicle_dispatch[global_vid] = {"headway": headway, "arrival_time": current_time}
            current_time += headway

        # 生成发车模块构成
        vehicle_initial_allocation = {}
        for vid in range(num_vehicles):
            global_vid = vid + vid_offset
            total = random.randint(1, max_modules)
            p = random.randint(0, total)
            f = total - p
            vehicle_initial_allocation[global_vid] = {"passenger_modules": p, "freight_modules": f}

        return {
            "vehicle_dispatch": vehicle_dispatch,  # 发车时间
            "initial_allocation": vehicle_initial_allocation,  # 初始模块配置
        }

    # 首先生成基础个体
    individual_up = generate_one_direction("up")
    individual_down = generate_one_direction("down")

    individual = {}
    individual["up"] = individual_up
    individual["down"] = individual_down


    # 首先为个体添加完整的module_adjustments结构，避免仿真系统报错
    print("🔄 为个体添加module_adjustments结构...")
    for direction in ["up", "down"]:
        individual[direction]["module_adjustments"] = {}
        for vid in individual[direction]["vehicle_dispatch"].keys():
            individual[direction]["module_adjustments"][vid] = {}
            # 为每个站点添加默认的调整值（0调整）
            num_stations = parameters.get('up_station_count', 10) if parameters else 10
            for station_id in range(num_stations):
                individual[direction]["module_adjustments"][vid][station_id] = {
                    "delta_p": 0,
                    "delta_f": 0
                }
    print("✅ module_adjustments结构添加完成")

    # 如果没有提供参数或数据，返回基础个体
    if parameters is None or global_demand_data is None:
        print('没有提供参数或数据，返回基础个体')
        return individual

    try:
        # 使用仿真系统生成真实的module_adjustments

        print("🔄 开始运行仿真以生成module_adjustments...")

        # 运行仿真以获取模块调整策略和范围
        (vehicle_schedule, total_cost, remaining_passengers, remaining_freights,
         failure_records, df_enriched, module_analysis_records, cost_components) = simulate_with_integrated_module_system(
            individual, parameters, global_demand_data,
            global_demand_data["passenger_demand_up"],
            global_demand_data["passenger_demand_down"],
            global_demand_data["freight_demand_up"],
            global_demand_data["freight_demand_down"]
        )

        print("✅ 仿真运行成功 生成module_adjustments...")

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

            # 提取建议的模块分配
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

            # ******** 核心修改逻辑 ********
            # 将完整的分析结果字典存储起来，为后续的智能变异提供所有原始参数
            adjustment_ranges[direction][vehicle_id][station_id] = analysis
            # ****************************

            # # 提取调整范围信息
            # # print('提取调整范围信息')
            # if 'adjustment_ranges' in analysis:
            #     adjustment_ranges[direction][vehicle_id][station_id] = {
            #         "passenger_modules": analysis['adjustment_ranges']['passenger_modules'],
            #         "freight_modules": analysis['adjustment_ranges']['freight_modules']
            #     }

        # 将生成的调整策略和范围添加到个体中
        individual["up"]["module_adjustments"] = module_adjustments["up"]
        individual["down"]["module_adjustments"] = module_adjustments["down"]
        individual["adjustment_ranges"] = adjustment_ranges

        # print(individual)

        print('模块调整相关计算完毕')
        return individual

    except Exception as e:
        print(f"⚠️ 仿真生成module_adjustments失败: {e}")
        # 如果仿真失败，添加空的module_adjustments
        for direction in ["up", "down"]:
            individual[direction]["module_adjustments"] = {}
            for vid in individual[direction]["vehicle_dispatch"].keys():
                individual[direction]["module_adjustments"][vid] = {}

        return individual
