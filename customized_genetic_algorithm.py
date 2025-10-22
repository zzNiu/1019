from deepdiff import DeepDiff
import json
import copy
import pandas as pd
import math
import random
import numpy as np
from deap import tools
import os
import matplotlib

from plot_cost_stack import plot_cost_stack_from_history


def customized_genetic_algorithm(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None,
                                 parameters=None, global_demand_data=None, max_regeneration_attempts=5, verbose=True):
    """
    Hybrid Genetic Algorithm with Regeneration Strategy for infeasible individuals.
    Uses module adjustment ranges from simulation to guide mutation and crossover.
    """

    # 成本历史（按每代最优个体记录）
    cost_history = {
        "mav_transport": [],
        "passenger_waiting": [],
        "freight_waiting": [],
        "unserved_penalty_cost": [],
        "unserved_passenger": [],
        "unserved_freight": [],
    }

    # 新增：每代所有个体的完整信息
    all_individuals_history = []  # 每一项是某一代所有个体的数据
    # 新增：每代各项指标的平均值
    generation_averages = {
        "mav_transport": [],
        "passenger_waiting": [],
        "freight_waiting": [],
        "unserved_penalty_cost": [],
        "unserved_passenger": [],
        "unserved_freight": [],
        "fitness": []
    }

    # 记录当前种群所有个体的信息和计算平均值
    def record_generation_data(pop):
        # 收集当前代所有个体的数据
        current_gen_data = []
        valid_costs = {
            "mav_transport": [],
            "passenger_waiting": [],
            "freight_waiting": [],
            "unserved_penalty_cost": [],
            "unserved_passenger": [],
            "unserved_freight": [],
            "fitness": []
        }

        # 遍历种群中的每个个体
        for idx, ind in enumerate(pop):
            # 基础信息
            ind_data = {
                "index": idx,
                "fitness": ind.fitness.values[0] if (
                            ind.fitness.valid and math.isfinite(ind.fitness.values[0])) else None,
                "cost_components": {}
            }

            # 获取成本组件
            cc = getattr(ind, 'cost_components', None)
            if cc and isinstance(cc, dict):
                for key in cost_history.keys():
                    cost_key = {
                        "mav_transport": "mav_transport_cost",
                        "passenger_waiting": "passenger_waiting_cost",
                        "freight_waiting": "freight_waiting_cost",
                        "unserved_penalty_cost": "unserved_penalty_cost",
                        "unserved_passenger": "unserved_passengers",
                        "unserved_freight": "unserved_freights"
                    }[key]
                    ind_data["cost_components"][key] = float(cc.get(cost_key, 0.0))
            current_gen_data.append(ind_data)

            # 收集有效数据用于计算平均值
            if ind_data["fitness"] is not None:
                valid_costs["fitness"].append(ind_data["fitness"])
                for key in cost_history.keys():
                    valid_costs[key].append(ind_data["cost_components"].get(key, 0.0))

        # 计算并记录平均值
        for key in generation_averages.keys():
            if valid_costs[key]:
                generation_averages[key].append(sum(valid_costs[key]) / len(valid_costs[key]))
            else:
                generation_averages[key].append(None)  # 无有效数据时记为None

        # 保存当前代所有个体数据
        all_individuals_history.append(current_gen_data)

    # 记录当前种群最优个体的三项成本（保持原有功能）
    def record_best_cost(pop):
        valid = [x for x in pop if x.fitness.valid and math.isfinite(x.fitness.values[0])]
        if not valid:
            for k in cost_history:
                cost_history[k].append(0.0)
            return

        best = min(valid, key=lambda x: x.fitness.values[0])
        cc = getattr(best, 'cost_components', None)

        if cc is None or not isinstance(cc, dict):
            print(f"❌ 严重错误：在第 {len(cost_history['passenger_waiting'])} 代的最优个体身上缺少成本数据，将记为0。")
            for k in cost_history:
                cost_history[k].append(0.0)
            return

        cost_history["mav_transport"].append(float(cc.get("mav_transport_cost", 0.0)))
        cost_history["passenger_waiting"].append(float(cc.get("passenger_waiting_cost", 0.0)))
        cost_history["freight_waiting"].append(float(cc.get("freight_waiting_cost", 0.0)))
        cost_history["unserved_penalty_cost"].append(float(cc.get("unserved_penalty_cost", 0.0)))
        cost_history["unserved_passenger"].append(float(cc.get("unserved_passengers", 0.0)))
        cost_history["unserved_freight"].append(float(cc.get("unserved_freights", 0.0)))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 初始种群评估
    print('----进入遗传算法 步骤3: 初始种群评估----')
    initial_population_before = population
    i = 1
    for ind in population:
        print(f'第 {i} 个个体')
        fit, failure_records, module_adjustment_ranges = toolbox.evaluate(ind)
        print('fit_value:', fit)
        ind.fitness.values = fit
        i += 1
    # <<< 在这里添加下面这行 >>>
    initial_population_after = population  # 捕获评估完成后的种群状态

    # 比较初始种群变化
    diff = DeepDiff(initial_population_before, initial_population_after, ignore_order=True)
    if not diff:
        print("✅ 初始种群数据未发生变化")
    else:
        print("⚠️ 初始种群数据发生变化：")

    # 记录初始种群评估结果
    feasible = [ind.fitness.values[0] for ind in population if math.isfinite(ind.fitness.values[0])]
    if feasible:
        gen_min = min(feasible)
        gen_avg = sum(feasible) / len(feasible)
        gen_max = max(feasible)
    else:
        gen_min = gen_avg = gen_max = float('nan')

    print('初始种群评估完成')
    check_station_info_existence(population, 0)

    # 新增：记录第0代所有个体信息和平均值
    record_generation_data(population)
    # 记录最优个体成本
    record_best_cost(population)

    logbook.record(gen=0, nevals=len(population), avg=gen_avg, min=gen_min, max=gen_max)
    if verbose:
        print(logbook.stream)

    # 种群进化
    print('----进入遗传算法 步骤4 种群开始进化----')
    for gen in range(1, ngen + 1):
        print(' 第(', gen, ')代 ')
        # 选择操作
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 变异操作（保持原有逻辑）
        for idx, mutant in enumerate(offspring):
            if random.random() < mutpb:
                mutant_before = copy.deepcopy(mutant)
                print(f"第{idx}个个体变异了")
                print(f"第{idx}个个体原适应度{mutant.fitness.values}")
                toolbox.mutate(mutant, parameters, global_demand_data)
                mutant_after = mutant
                mutant.mutated = True

                diff__ = DeepDiff(mutant_before, mutant_after, ignore_order=True)
                if not diff__:
                    print("✅ 个体变异未更新")
                else:
                    print("⚠️ 个体变异已经更新 确保真的发生变异")

        # 评估和处理不可行个体（保持原有逻辑）
        for i, ind in enumerate(offspring):
            if not ind.fitness.valid:
                print(f"➡️ 代数 {gen}，个体 {i + 1}/{len(offspring)}：开始评估")

                fit, failure_records, module_adjustment_ranges = toolbox.evaluate(ind)
                print(f"评估结果: {fit}")

                if not math.isfinite(fit[0]):
                    print(f"❌ 个体 {i + 1} 不可行，尝试重新生成")

                    best_ind = toolbox.clone(ind)
                    best_fit = fit

                    for attempt in range(max_regeneration_attempts):
                        feasible_parents = [p for p in population if math.isfinite(p.fitness.values[0])]

                        if len(feasible_parents) >= 2:
                            parent1, parent2 = random.sample(feasible_parents, 2)
                            new_ind = toolbox.clone(parent1)
                            toolbox.mutate(new_ind, parameters, global_demand_data)
                        else:
                            new_ind = toolbox.individual()

                        new_fit, new_failures, new_ranges = toolbox.evaluate(new_ind)
                        print(f"🔄 重生成尝试 {attempt + 1}，fit: {new_fit}")

                        if math.isfinite(new_fit[0]):
                            best_ind = new_ind
                            best_fit = new_fit
                            best_ind.adjustment_ranges = new_ranges
                            print(f"✅ 生成成功，个体 {i + 1} 现在可行")
                            break

                    ind = best_ind
                    fit = best_fit
                    offspring[i] = best_ind
                else:
                    print(f"✅ 评估成功，个体 {i + 1} 可行")
                    ind.adjustment_ranges = module_adjustment_ranges

                ind.fitness.values = fit

            else:
                if hasattr(ind, 'mutated') and ind.mutated:
                    print(f"个体 {i + 1} 已在变异中更新并评估")
                    print('individual.cost_components:', ind.cost_components)
                    print('fit_value:', ind.fitness)
                    del ind.mutated
                else:
                    print(f"个体 {i + 1} 直接继承母代")
                    print('individual.cost_components:', ind.cost_components)
                    print('fit_value:', ind.fitness)

        print('子代生成完毕')
        check_station_info_existence(offspring, gen)

        # 更新名人堂
        if halloffame is not None:
            halloffame.update(offspring)

        # 精英保留策略
        elite_size = max(1, int(len(population) * 0.02))
        elites = tools.selBest(population, elite_size)
        offspring_size = len(population) - elite_size
        offspring = tools.selBest(offspring, offspring_size)
        population[:] = elites + offspring

        # 统计当前种群
        feasible = [ind.fitness.values[0]
                    for ind in population
                    if ind.fitness.valid
                    and len(ind.fitness.values) > 0
                    and math.isfinite(ind.fitness.values[0])]

        if feasible:
            gen_min = min(feasible)
            gen_avg = sum(feasible) / len(feasible)
            gen_max = max(feasible)
        else:
            gen_min = gen_avg = gen_max = float('nan')

        logbook.record(gen=gen, nevals=len(offspring), avg=gen_avg, min=gen_min, max=gen_max)

        # 新增：记录当前代所有个体信息和平均值
        record_generation_data(population)
        # 记录最优个体成本
        record_best_cost(population)

        if verbose:
            print(logbook.stream)
            last_costs = {k: v[-1] for k, v in cost_history.items() if v}
            if last_costs:
                print(f"  \n--- 第 {gen} 代最优成本构成 ---")
                print(f"  MAV运输成本: {last_costs.get('mav_transport', 0.0):.4f}")
                print(f"  乘客等待成本: {last_costs.get('passenger_waiting', 0.0):.4f}")
                print(f"  货物等待成本: {last_costs.get('freight_waiting', 0.0):.4f}")
                print(f"  未服务需求惩罚: {last_costs.get('unserved_penalty_cost', 0.0):.4f}")
                print(f"  未服务乘客惩罚: {last_costs.get('unserved_passenger', 0.0):.4f}")
                print(f"  未服务货物惩罚: {last_costs.get('unserved_freight', 0.0):.4f}")
                print("-" * 30)

    print('进化完成')
    # 返回新增的两个历史记录
    return population, logbook, cost_history, all_individuals_history, generation_averages


def run_genetic_algorithm_with_initialization(population_size, num_vehicles, max_modules,
                                              toolbox, cxpb, mutpb, ngen,
                                              headway_range=(3, 20), stats=None, halloffame=None,
                                              parameters=None, global_demand_data=None, verbose=True,
                                              results_dir=None):
    """运行完整的遗传算法，包括初始种群生成"""
    if verbose:
        print("=== 开始运行遗传算法 ===")
        print(f"种群大小: {population_size}")
        print(f"车辆数量: {num_vehicles}")
        print(f"最大模块数: {max_modules}")
        print(f"车头时距范围: {headway_range}")
        print(f"交叉概率: {cxpb}")
        print(f"变异概率: {mutpb}")
        print(f"进化代数: {ngen}")

    # 生成初始种群
    if verbose:
        print("\n--- 进入遗传算法 步骤1: 生成初始种群 ---")

    population = []
    for i in range(population_size):
        individual = toolbox.individual()
        population.append(individual)
        print(f"已初始化 {i + 1}/{population_size} 个个体")

    if verbose:
        print(f"种群初始化完成，共 {len(population)} 个个体")

    # 运行遗传算法（接收新增的返回值）
    if verbose:
        print("\n--- 进入遗传算法 步骤2: 运行遗传算法 ---")

    final_population, logbook, cost_history, all_individuals_history, generation_averages = customized_genetic_algorithm(
        population=population,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=halloffame,
        parameters=parameters,
        global_demand_data=global_demand_data,
        verbose=verbose
    )

    print("cost_history:", cost_history)

    # 保存新增的历史记录
    if results_dir:
        # 保存成本历史到Excel
        print("\n--- 正在保存成本进化历史到 Excel 文件 ---")
        try:
            df_cost_history = pd.DataFrame(cost_history)
            df_cost_history.insert(0, 'generation', range(len(df_cost_history)))
            excel_save_path = os.path.join(results_dir, "cost_evolution_history.xlsx")
            df_cost_history.to_excel(excel_save_path, index=False)
            print(f"✅ 成本进化历史已成功保存到: {excel_save_path}")
        except Exception as e:
            print(f"❌ 保存成本历史到 Excel 时发生错误: {e}")

        # 新增：保存每代平均值到Excel
        try:
            df_avg = pd.DataFrame(generation_averages)
            df_avg.insert(0, 'generation', range(len(df_avg)))
            avg_save_path = os.path.join(results_dir, "generation_averages.xlsx")
            df_avg.to_excel(avg_save_path, index=False)
            print(f"✅ 每代平均值已成功保存到: {avg_save_path}")
        except Exception as e:
            print(f"❌ 保存每代平均值到 Excel 时发生错误: {e}")

        # 新增：保存所有个体信息到Excel（按代拆分）
        try:
            all_individuals_df = []
            for gen_idx, gen_data in enumerate(all_individuals_history):
                for ind_data in gen_data:
                    row = {
                        "generation": gen_idx,
                        "individual_index": ind_data["index"],
                        "fitness": ind_data["fitness"]
                    }
                    # 添加各项成本组件
                    row.update(ind_data["cost_components"])
                    all_individuals_df.append(row)

            df_all = pd.DataFrame(all_individuals_df)
            all_ind_save_path = os.path.join(results_dir, "all_individuals_history.xlsx")
            df_all.to_excel(all_ind_save_path, index=False)
            print(f"✅ 所有个体历史信息已成功保存到: {all_ind_save_path}")
        except Exception as e:
            print(f"❌ 保存所有个体信息到 Excel 时发生错误: {e}")

        # 绘制成本构成堆叠图
        print("\n--- 正在绘制成本构成进化堆叠图 ---")
        try:
            matplotlib.rcParams['font.family'] = 'SimHei'
            matplotlib.rcParams['axes.unicode_minus'] = False
            save_path = os.path.join(results_dir, "成本构成堆叠图.png")
            plot_cost_stack_from_history(cost_history, title="成本构成进化堆叠图", save_path=save_path)
            print(f"✅ 成本构成堆叠图已成功保存到: {save_path}")
        except Exception as e:
            print(f"❌ 绘制成本构成堆叠图时发生错误: {e}")
    else:
        print("\n--- 未提供结果目录 (results_dir)，跳过绘制成本构成堆叠图 ---")

    if verbose:
        print("\n=== 遗传算法运行完成 ===")

    # 返回新增的历史记录
    return final_population, logbook, cost_history, all_individuals_history, generation_averages


def check_station_info_existence(offspring_population, current_gen):
    """检查种群中所有个体的 'station_info' 完整性"""
    print(f"\n--- [第 {current_gen} 代] 开始检查子代 'station_info' 完整性 ---")
    is_fully_valid = True

    for idx, individual in enumerate(offspring_population):
        if 'adjustment_ranges' not in individual:
            print(f"⚠️ 警告: 个体 {idx + 1} 缺少 'adjustment_ranges' 键，无法检查。")
            is_fully_valid = False
            continue

        for direction in ['up', 'down']:
            if direction not in individual['adjustment_ranges']:
                print(f"⚠️ 警告: 个体 {idx + 1} 的 'adjustment_ranges' 中缺少 '{direction}' 方向的数据。")
                is_fully_valid = False
                continue

            for vehicle_id, vehicle_data in individual['adjustment_ranges'][direction].items():
                for station_id, station_data in vehicle_data.items():
                    if 'station_info' not in station_data:
                        print(f"❌ 错误: 在个体 {idx + 1} 的 'adjustment_ranges' -> '{direction}' -> "
                              f"车辆 '{vehicle_id}' -> 站点 '{station_id}' 中, 未找到 'station_info' 键。")
                        is_fully_valid = False

    if is_fully_valid:
        print(f"✅ 检查通过: 第 {current_gen} 代所有子代个体的 'station_info' 均存在。")
    else:
        print(f"❌ 检查未通过: 第 {current_gen} 代存在数据不完整的个体。")

    print("--- 完整性检查结束 ---\n")
    return is_fully_valid

# from deepdiff import DeepDiff
# import json
#
# import copy
# import pandas as pd
# import math
# import random
# import numpy as np
# from deap import tools
#
# import os  # <--- 新增导入
# import matplotlib # <--- 新增导入
#
# # === 新增：读评估阶段写入的成本缓存 & 绘图函数 ===
# from plot_cost_stack import plot_cost_stack_from_history
#
#
# def customized_genetic_algorithm(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None,
#                            parameters=None, global_demand_data=None, max_regeneration_attempts=5, verbose=True):
#     """
#     Hybrid Genetic Algorithm with Regeneration Strategy for infeasible individuals.
#     Uses module adjustment ranges from simulation to guide mutation and crossover.
#
#     :param population: Initial population
#     :param toolbox: DEAP toolbox (with evaluate, mate, mutate, select)
#     :param cxpb: Crossover probability
#     :param mutpb: Mutation probability
#     :param ngen: Number of generations
#     :param stats: DEAP Statistics object
#     :param halloffame: DEAP HallOfFame object
#     :param parameters: Custom parameters passed to evaluate
#     :param global_demand_data: Custom demand data passed to evaluate
#     :param max_regeneration_attempts: Maximum times to attempt regenerating an infeasible individual
#     :param verbose: Whether to print log each generation
#     :return: (final population, logbook)
#     """
#
#     # 在算法开始时清空缓存，确保每次运行都是干净的
#     # cost_cache.clear()
#
#     # ===== 在 customized_genetic_algorithm.py 中（遗传主循环外侧）=====
#     # === 新增：成本历史（按每代最优个体记录） ===
#     # cost_history = {"passenger": [], "freight": [], "mav": []}
#     cost_history = {
#         "mav_transport": [],
#         "passenger_waiting": [],
#         "freight_waiting": [],
#         "unserved_penalty_cost": [],
#         "unserved_passenger": [],
#         "unserved_freight": [],
#         # "total_cost": [],
#     }
#
#     # 新增：每代所有个体的完整信息
#     all_individuals_history = []  # 每一项是某一代所有个体的数据
#     # 新增：每代各项指标的平均值
#     generation_averages = {
#         "mav_transport": [],
#         "passenger_waiting": [],
#         "freight_waiting": [],
#         "unserved_penalty_cost": [],
#         "unserved_passenger": [],
#         "unserved_freight": [],
#         "fitness": []
#     }
#
#     # 记录当前种群所有个体的信息和计算平均值
#     def record_generation_data(pop):
#         # 收集当前代所有个体的数据
#         current_gen_data = []
#         valid_costs = {
#             "mav_transport": [],
#             "passenger_waiting": [],
#             "freight_waiting": [],
#             "unserved_penalty_cost": [],
#             "unserved_passenger": [],
#             "unserved_freight": [],
#             "fitness": []
#         }
#
#         # 遍历种群中的每个个体
#         for idx, ind in enumerate(pop):
#             # 基础信息
#             ind_data = {
#                 "index": idx,
#                 "fitness": ind.fitness.values[0] if (
#                             ind.fitness.valid and math.isfinite(ind.fitness.values[0])) else None,
#                 "cost_components": {}
#             }
#
#             # 获取成本组件
#             cc = getattr(ind, 'cost_components', None)
#             if cc and isinstance(cc, dict):
#                 for key in cost_history.keys():
#                     cost_key = {
#                         "mav_transport": "mav_transport_cost",
#                         "passenger_waiting": "passenger_waiting_cost",
#                         "freight_waiting": "freight_waiting_cost",
#                         "unserved_penalty_cost": "unserved_penalty_cost",
#                         "unserved_passenger": "unserved_passengers",
#                         "unserved_freight": "unserved_freights"
#                     }[key]
#                     ind_data["cost_components"][key] = float(cc.get(cost_key, 0.0))
#             current_gen_data.append(ind_data)
#
#             # 收集有效数据用于计算平均值
#             if ind_data["fitness"] is not None:
#                 valid_costs["fitness"].append(ind_data["fitness"])
#                 for key in cost_history.keys():
#                     valid_costs[key].append(ind_data["cost_components"].get(key, 0.0))
#
#         # 计算并记录平均值
#         for key in generation_averages.keys():
#             if valid_costs[key]:
#                 generation_averages[key].append(sum(valid_costs[key]) / len(valid_costs[key]))
#             else:
#                 generation_averages[key].append(None)  # 无有效数据时记为None
#
#         # 保存当前代所有个体数据
#         all_individuals_history.append(current_gen_data)
#
#     # === 新增：记录当前种群最优个体的三项成本 ===
#     def record_best_cost(pop):
#         # 过滤出已赋值适应度且有限的个体
#         valid = [x for x in pop if x.fitness.valid and math.isfinite(x.fitness.values[0])]
#         if not valid:
#             # 如果没有可用的个体（例如全部不可行），记 0 占位，保证代数对齐
#             for k in cost_history:
#                 cost_history[k].append(0.0)
#             return
#
#         # 取适应度最小（更优）的个体
#         best = min(valid, key=lambda x: x.fitness.values[0])
#
#         # ==================== 解决方案核心逻辑 ====================
#         # 直接从最优个体 best 身上读取在评估时附加的 cost_components 属性。
#         # 使用 getattr 函数可以安全地获取属性，如果属性不存在，则返回 None。
#         cc = getattr(best, 'cost_components', None)
#
#         if cc is None or not isinstance(cc, dict):
#             # 这个后备逻辑用于处理极端情况，例如某个体因未知错误而缺少成本数据。
#             # 在正常情况下，由于所有被评估的个体都会被附加 .cost_components 属性，
#             # 所以这个分支理论上不应该被执行。
#             print(f"❌ 严重错误：在第 {len(cost_history['passenger'])} 代的最优个体身上缺少成本数据，将记为0。")
#             for k in cost_history:
#                 cost_history[k].append(0.0)
#             return
#         # ==========================================================
#
#         # 使用从个体身上获取到的成本数据更新历史记录
#         cost_history["mav_transport"].append(float(cc.get("mav_transport_cost", 0.0)))
#         cost_history["passenger_waiting"].append(float(cc.get("passenger_waiting_cost", 0.0)))
#         cost_history["freight_waiting"].append(float(cc.get("freight_waiting_cost", 0.0)))
#         cost_history["unserved_penalty_cost"].append(float(cc.get("unserved_penalty_cost", 0.0)))
#         cost_history["unserved_passenger"].append(float(cc.get("unserved_passengers", 0.0)))
#         cost_history["unserved_freight"].append(float(cc.get("unserved_freights", 0.0)))
#
#     logbook = tools.Logbook()
#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
#
#
#     print('----进入遗传算法 步骤3: 初始种群评估----')
#     initial_population_before = population
#     i = 1
#     for ind in population:
#         print(f'第 {i} 个个体')
#         # print('ind:', ind)
#         # print('初始种群评估')
#         i += 1
#         fit, failure_records, module_adjustment_ranges = toolbox.evaluate(ind)
#         print('fit_value:', fit)
#         ind.fitness.values = fit
#         # 存储模块调整范围信息到个体中，供后续变异使用
#         # ind.adjustment_ranges = module_adjustment_ranges
#     initial_population_after = population
#
#     # 比较
#     diff = DeepDiff(initial_population_before, initial_population_after, ignore_order=True)
#
#     # 打印结果
#     if not diff:
#         print("✅ 初始种群数据未发生变化")
#     else:
#         print("⚠️ 初始种群数据发生变化：")
#
#     # 记录初始种群评估结果
#     feasible = [ind.fitness.values[0] for ind in population if math.isfinite(ind.fitness.values[0])]
#
#     if feasible:
#         gen_min = min(feasible)
#         gen_avg = sum(feasible) / len(feasible)
#         gen_max = max(feasible)
#     else:
#         gen_min = gen_avg = gen_max = float('nan')
#
#     print('初始种群评估完成')
#
#     # ==================== 在这里新增函数调用 ====================
#     # 调用封装好的函数，对生成的子代种群进行 'station_info' 完整性检查
#     check_station_info_existence(population, 0)
#     # ==========================================================
#
#     # === 新增：记录第 0 代最优个体的成本构成 ===
#     record_best_cost(population)
#
#     logbook.record(gen=0, nevals=len(population),avg=gen_avg, min=gen_min, max=gen_max)
#     if verbose:
#         print(logbook.stream)
#
#
#     # 种群进化Evolution loop
#     print('----进入遗传算法 步骤4 种群开始进化----')
#     # print('----第2步：种群开始进化----')
#     for gen in range(1, ngen + 1):
#         print(' 第(', gen, ')代 ')
#         # 选择操作
#         offspring = toolbox.select(population, len(population))
#         offspring = list(map(toolbox.clone, offspring))
#
#         # 变异
#         for idx, mutant in enumerate(offspring):
#             if random.random() < mutpb:
#                 # print('mutant["adjustment_ranges"]:', mutant["adjustment_ranges"])
#                 mutant_before = copy.deepcopy(mutant)
#                 print(f"第{idx}个个体变异了")
#                 print(f"第{idx}个个体原适应度{mutant.fitness.values}")
#                 # 如果个体有调整范围信息，传递给变异操作
#                 # if hasattr(mutant, 'adjustment_ranges'):
#                 #     toolbox.mutate(mutant, parameters, global_demand_data)
#                 # else:
#                 toolbox.mutate(mutant, parameters, global_demand_data)
#                 mutant_after = mutant
#                 # ==================== 修改/新增逻辑：开始 ====================
#                 mutant.mutated = True  # 为变异后的个体打上标记
#
#                 diff__ = DeepDiff(mutant_before, mutant_after, ignore_order=True)
#
#                 # 打印结果
#                 if not diff__:
#                     print("✅ 个体变异未更新")
#                 else:
#                     print("⚠️ 个体变异已经更新 确保真的发生变异")
#                     # print(json.dumps(diff__, indent=2, ensure_ascii=False))
#                     # print(diff__)  # 可以正常打印
#                     # print(json.dumps(diff__.to_dict(), indent=2, ensure_ascii=False))
#                 # ==================== 修改/新增逻辑：结束 ====================
#
#                 # del mutant.fitness.values
#                 # 清除调整范围信息，因为个体已经改变（
#                 # 在变异之后已经调用函数进行了计算和更新，所以不需要删除
#                 # if 'adjustment_ranges' in mutant:
#                 #     del mutant['adjustment_ranges']
#
#         # 评估和处理不可行个体
#         for i, ind in enumerate(offspring):
#             if not ind.fitness.valid:
#                 print(f"➡️ 代数 {gen}，个体 {i + 1}/{len(offspring)}：开始评估")
#
#                 # 尝试评估个体
#                 fit, failure_records, module_adjustment_ranges = toolbox.evaluate(ind)
#                 print(f"评估结果: {fit}")
#
#                 # 处理不可行个体
#                 if not math.isfinite(fit[0]):
#                     print(f"❌ 个体 {i + 1} 不可行，尝试重新生成")
#
#                     # 存储最佳尝试结果
#                     best_ind = toolbox.clone(ind)
#                     best_fit = fit
#
#                     # 尝试重新生成个体
#                     for attempt in range(max_regeneration_attempts):
#                         # 从可行个体中随机选择两个父本
#                         feasible_parents = [p for p in population if math.isfinite(p.fitness.values[0])]
#
#                         if len(feasible_parents) >= 2:
#                             # 有足够的可行父本，进行交叉和变异
#                             parent1, parent2 = random.sample(feasible_parents, 2)
#                             new_ind = toolbox.clone(parent1)
#
#                             # 应用交叉
#                             # if random.random() < cxpb:  # 高概率交叉
#                             #     toolbox.mate(new_ind, toolbox.clone(parent2), parameters, global_demand_data)
#
#                             # 应用变异
#                             # if hasattr(parent1, 'adjustment_ranges'):
#                             #     toolbox.mutate(new_ind, parameters, global_demand_data)
#                             # else:
#                             toolbox.mutate(new_ind, parameters, global_demand_data)
#                         else:
#                             # 没有足够的可行父本，生成新个体
#                             new_ind = toolbox.individual()
#
#                         # 评估新个体
#                         new_fit, new_failures, new_ranges = toolbox.evaluate(new_ind)
#                         print(f"🔄 重生成尝试 {attempt + 1}，fit: {new_fit}")
#
#                         # 如果新个体可行或比之前的更好，则保留
#                         if math.isfinite(new_fit[0]):
#                         # if math.isfinite(new_fit[0]) or (not math.isfinite(best_fit[0]) and new_fit[0] < best_fit[0]):
#                             best_ind = new_ind
#                             best_fit = new_fit
#                             best_ind.adjustment_ranges = new_ranges
#
#                             if math.isfinite(new_fit[0]):
#                                 print(f"✅ 生成成功，个体 {i + 1} 现在可行")
#                                 break
#
#                     # 使用最佳尝试结果替换当前个体
#                     ind = best_ind
#                     fit = best_fit
#                     offspring[i] = best_ind
#                 else:
#                     print(f"✅ 评估成功，个体 {i + 1} 可行")
#                     # 存储模块调整范围信息到个体中
#                     ind.adjustment_ranges = module_adjustment_ranges
#
#                 ind.fitness.values = fit
#
#             # ==================== 修改/新增逻辑：开始 ====================
#             else:
#                 # 检查个体是否有 'mutated' 标记
#                 if hasattr(ind, 'mutated') and ind.mutated:
#                     print(f"个体 {i + 1} 已在变异中更新并评估")
#                     print('individual.cost_components:', ind.cost_components)
#                     print('fit_value:', ind.fitness)
#
#                     # 清除标记，以免影响下一代
#                     del ind.mutated
#                 else:
#                     print(f"个体 {i + 1} 直接继承母代")
#                     print('individual.cost_components:', ind.cost_components)
#                     print('fit_value:', ind.fitness)
#
#         print('子代生成完毕')
#
#         # ==================== 在这里新增函数调用 ====================
#         # 调用封装好的函数，对生成的子代种群进行 'station_info' 完整性检查
#         check_station_info_existence(offspring, gen)
#         # ==========================================================
#
#         # 更新名人堂
#         if halloffame is not None:
#             halloffame.update(offspring)
#
#         # 精英保留策略：保留一部分最好的父代个体
#         elite_size = max(1, int(len(population) * 0.02))  # 保留10%的精英
#         elites = tools.selBest(population, elite_size)
#
#         # 替换种群，但保留精英
#         offspring_size = len(population) - elite_size
#         offspring = tools.selBest(offspring, offspring_size)  # 选择最好的后代
#         population[:] = elites + offspring  # 精英 + 后代
#
#         # # 替换种群
#         # population[:] = offspring
#
#         # 统计当前种群中所有已评估且有效的个体
#         feasible = [ind.fitness.values[0]
#                     for ind in population
#                     if ind.fitness.valid
#                     and len(ind.fitness.values) > 0
#                     and math.isfinite(ind.fitness.values[0])]
#
#         if feasible:
#             gen_min = min(feasible)
#             gen_avg = sum(feasible) / len(feasible)
#             gen_max = max(feasible)
#         else:
#             gen_min = gen_avg = gen_max = float('nan')
#
#         logbook.record(gen=gen, nevals=len(offspring), avg=gen_avg, min=gen_min, max=gen_max)
#
#         # === 新增：记录本代最优个体的成本构成 ===
#         record_best_cost(population)
#
#         if verbose:
#             print(logbook.stream)
#             # 增加打印成本构成的逻辑
#             last_costs = {k: v[-1] for k, v in cost_history.items() if v}
#             if last_costs:
#                 print(f"  \n--- 第 {gen} 代最优成本构成 ---")
#                 print(f"  MAV运输成本: {last_costs.get('mav_transport', 0.0):.4f}")
#                 print(f"  乘客等待成本: {last_costs.get('passenger_waiting', 0.0):.4f}")
#                 print(f"  货物等待成本: {last_costs.get('freight_waiting', 0.0):.4f}")
#                 print(f"  未服务需求惩罚: {last_costs.get('unserved_penalty_cost', 0.0):.4f}")
#                 print(f"  未服务乘客惩罚: {last_costs.get('unserved_passenger', 0.0):.4f}")
#                 print(f"  未服务货物惩罚: {last_costs.get('unserved_freight', 0.0):.4f}")
#                 print("-" * 30)
#
#     print('进化完成')
#
#     return population, logbook, cost_history
#
#
# def run_genetic_algorithm_with_initialization(population_size, num_vehicles, max_modules,
#                                             toolbox, cxpb, mutpb, ngen,
#                                             headway_range=(3, 20), stats=None, halloffame=None,
#                                             parameters=None, global_demand_data=None, verbose=True,
#                                             results_dir=None): # <--- 1. 新增 results_dir 参数):
#     """
#     运行完整的遗传算法，包括初始种群生成
#
#     Args:
#         population_size: 种群大小
#         num_vehicles: 车辆数量
#         max_modules: 最大模块数
#         toolbox: DEAP工具箱
#         cxpb: 交叉概率
#         mutpb: 变异概率
#         ngen: 进化代数
#         headway_range: 车头时距范围
#         stats: DEAP统计对象
#         halloffame: DEAP名人堂对象
#         parameters: 自定义参数
#         global_demand_data: 全局需求数据
#         verbose: 是否打印详细信息
#
#     Returns:
#         tuple: (final_population, logbook)
#     """
#     if verbose:
#         print("=== 开始运行遗传算法 ===")
#         print(f"种群大小: {population_size}")
#         print(f"车辆数量: {num_vehicles}")
#         print(f"最大模块数: {max_modules}")
#         print(f"车头时距范围: {headway_range}")
#         print(f"交叉概率: {cxpb}")
#         print(f"变异概率: {mutpb}")
#         print(f"进化代数: {ngen}")
#
#     # 生成初始种群
#     if verbose:
#         print("\n--- 进入遗传算法 步骤1: 生成初始种群 ---")
#
#     population = []
#     for i in range(population_size):
#         individual = toolbox.individual()
#         population.append(individual)
#         # if verbose and (i + 1) % 10 == 0:
#         #     print(f"已初始化 {i + 1}/{population_size} 个个体")
#         print(f"已初始化 {i + 1}/{population_size} 个个体")
#
#     if verbose:
#         print(f"种群初始化完成，共 {len(population)} 个个体")
#
#     # 运行遗传算法
#     if verbose:
#         print("\n--- 进入遗传算法 步骤2: 运行遗传算法 ---")
#
#     final_population, logbook, cost_history = customized_genetic_algorithm(
#         population=population,
#         toolbox=toolbox,
#         cxpb=cxpb,
#         mutpb=mutpb,
#         ngen=ngen,
#         stats=stats,
#         halloffame=halloffame,
#         parameters=parameters,
#         global_demand_data=global_demand_data,
#         verbose=verbose
#     )
#
#     print("cost_history:", cost_history)
#
#     # ==================== 新增逻辑：开始 ====================
#     # 在遗传算法运行结束后，直接绘制成本构成堆叠图
#     if results_dir:
#
#         # --- 2. 新增：保存成本历史到 Excel ---
#         print("\n--- 正在保存成本进化历史到 Excel 文件 ---")
#         try:
#             df_cost_history = pd.DataFrame(cost_history)
#             # 增加 'generation' 列，方便查看
#             df_cost_history.insert(0, 'generation', range(len(df_cost_history)))
#             excel_save_path = os.path.join(results_dir, "cost_evolution_history.xlsx")
#             df_cost_history.to_excel(excel_save_path, index=False)
#             print(f"✅ 成本进化历史已成功保存到: {excel_save_path}")
#         except Exception as e:
#             print(f"❌ 保存成本历史到 Excel 时发生错误: {e}")
#
#         print("\n--- 正在绘制成本构成进化堆叠图 ---")
#         try:
#             # 确保在绘图前设置中文字体，以防乱码
#             matplotlib.rcParams['font.family'] = 'SimHei'
#             matplotlib.rcParams['axes.unicode_minus'] = False
#
#             # 定义图表的完整保存路径
#             save_path = os.path.join(results_dir, "成本构成堆叠图.png")
#
#             # 调用绘图函数
#             plot_cost_stack_from_history(cost_history, title="成本构成进化堆叠图", save_path=save_path)
#
#             print(f"✅ 成本构成堆叠图已成功保存到: {save_path}")
#         except Exception as e:
#             print(f"❌ 绘制成本构成堆叠图时发生错误: {e}")
#     else:
#         print("\n--- 未提供结果目录 (results_dir)，跳过绘制成本构成堆叠图 ---")
#     # ==================== 新增逻辑：结束 ====================
#
#     if verbose:
#         print("\n=== 遗传算法运行完成 ===")
#
#     return final_population, logbook, cost_history, all_individuals_history, generation_averages
#
#
# def check_station_info_existence(offspring_population, current_gen):
#     """
#     封装的检查函数，用于验证一个种群中所有个体的内部数据
#     是否都完整地包含了 'adjustment_ranges' 键。
#
#     Args:
#         offspring_population (list): 需要被检查的子代种群。
#         current_gen (int): 当前的进化代数，用于在日志中清晰地报告问题。
#
#     Returns:
#         bool: 如果所有个体都通过检查，返回 True；否则返回 False。
#     """
#     print(f"\n--- [第 {current_gen} 代] 开始检查子代 'station_info' 完整性 ---")
#     is_fully_valid = True  # 初始化标志位，假设所有个体都是有效的
#
#     # 遍历种群中的每一个个体
#     for idx, individual in enumerate(offspring_population):
#         # 核心修正：使用 'in' 关键字检查 'adjustment_ranges' 是否是 'individual' 的一个键
#         if 'adjustment_ranges' not in individual:
#             # 这种情况可能发生在个体未经评估就进入了下一代，是潜在的数据问题
#             print(f"⚠️ 警告: 个体 {idx + 1} 缺少 'adjustment_ranges' 键，无法检查。")
#             is_fully_valid = False
#             continue  # 跳过此个体的后续检查
#
#         # 遍历 'up' 和 'down' 两个方向
#         for direction in ['up', 'down']:
#             # 检查方向数据是否存在
#             if direction not in individual['adjustment_ranges']: # 注意这里也要用键访问
#                 print(f"⚠️ 警告: 个体 {idx + 1} 的 'adjustment_ranges' 中缺少 '{direction}' 方向的数据。")
#                 is_fully_valid = False
#                 continue
#
#             # 遍历该方向下的所有车辆
#             for vehicle_id, vehicle_data in individual['adjustment_ranges'][direction].items():
#                 # 遍历该车辆的所有站点记录
#                 for station_id, station_data in vehicle_data.items():
#                     # 核心检查：判断 'station_info' 键是否存在
#                     if 'station_info' not in station_data:
#                         print(f"❌ 错误: 在个体 {idx + 1} 的 'adjustment_ranges' -> '{direction}' -> "
#                               f"车辆 '{vehicle_id}' -> 站点 '{station_id}' 中, 未找到 'station_info' 键。")
#                         is_fully_valid = False  # 发现问题，将标志位置为 False
#
#     # 循环结束后，根据标志位的最终状态打印总结信息
#     if is_fully_valid:
#         print(f"✅ 检查通过: 第 {current_gen} 代所有子代个体的 'station_info' 均存在。")
#     else:
#         print(f"❌ 检查未通过: 第 {current_gen} 代存在数据不完整的个体。")
#
#     print("--- 完整性检查结束 ---\n")
#     return is_fully_valid