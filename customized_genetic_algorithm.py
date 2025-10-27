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
        "fitness": [],  # <--- 新增
        "mav_transport": [],
        "waiting_time_cost": [],  # <--- 新增
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
        "fitness": [],
        "mav_transport": [],
        "waiting_time_cost": [],  # <--- 在这里添加新行
        "passenger_waiting": [],
        "freight_waiting": [],
        "unserved_penalty_cost": [],
        "unserved_passenger": [],
        "unserved_freight": []
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
            "fitness": [],
            "waiting_time_cost": []  # <--- 在这里添加新行
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
            # (请替换为这个新代码块)
            cc = getattr(ind, 'cost_components', None)
            if cc and isinstance(cc, dict):
                # ==================== 修改排序逻辑：开始 ====================
                # 1. 先从 cc (cost_components) 中获取所有原始值
                mav_cost = float(cc.get("mav_transport_cost", 0.0))
                p_wait_cost = float(cc.get("passenger_waiting_cost", 0.0))
                f_wait_cost = float(cc.get("freight_waiting_cost", 0.0))
                unserved_p_cost = float(cc.get("unserved_penalty_cost", 0.0))
                unserved_p_num = float(cc.get("unserved_passengers", 0.0))
                unserved_f_num = float(cc.get("unserved_freights", 0.0))

                # 2. 计算派生值 (waiting_time_cost)
                total_wait_cost = p_wait_cost + f_wait_cost

                # 3. 按照您要求的顺序，将键值对插入 ind_data["cost_components"] 字典
                #    (Python 3.7+ 字典会保持此插入顺序，Pandas会遵循此顺序)
                ind_data["cost_components"]["mav_transport"] = mav_cost
                ind_data["cost_components"]["waiting_time_cost"] = total_wait_cost
                ind_data["cost_components"]["passenger_waiting"] = p_wait_cost
                ind_data["cost_components"]["freight_waiting"] = f_wait_cost
                ind_data["cost_components"]["unserved_penalty_cost"] = unserved_p_cost
                ind_data["cost_components"]["unserved_passenger"] = unserved_p_num
                ind_data["cost_components"]["unserved_freight"] = unserved_f_num
                # ==================== 修改排序逻辑：结束 ====================

            current_gen_data.append(ind_data)
            # cc = getattr(ind, 'cost_components', None)
            # if cc and isinstance(cc, dict):
            #     for key in cost_history.keys():
            #         cost_key = {
            #             "mav_transport": "mav_transport_cost",
            #             "passenger_waiting": "passenger_waiting_cost",
            #             "freight_waiting": "freight_waiting_cost",
            #             "unserved_penalty_cost": "unserved_penalty_cost",
            #             "unserved_passenger": "unserved_passengers",
            #             "unserved_freight": "unserved_freights"
            #         }[key]
            #         ind_data["cost_components"][key] = float(cc.get(cost_key, 0.0))
            #
            #     # ==================== 新增逻辑：开始 ====================
            #     # 计算并添加总等待成本
            #     p_wait = ind_data["cost_components"].get("passenger_waiting", 0.0)
            #     f_wait = ind_data["cost_components"].get("freight_waiting", 0.0)
            #     ind_data["cost_components"]["waiting_time_cost"] = p_wait + f_wait
            #     # ==================== 新增逻辑：结束 ====================
            #
            # current_gen_data.append(ind_data)

            # 收集有效数据用于计算平均值
            if ind_data["fitness"] is not None:
                valid_costs["fitness"].append(ind_data["fitness"])
                for key in cost_history.keys():
                    valid_costs[key].append(ind_data["cost_components"].get(key, 0.0))

                # ==================== 新增逻辑：开始 ====================
                # 收集 waiting_time_cost 用于计算平均值
                valid_costs["waiting_time_cost"].append(ind_data["cost_components"].get("waiting_time_cost", 0.0))
                # ==================== 新增逻辑：结束 ====================

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

        # --- 新增和修改的逻辑 ---
        # 1. 获取所有需要的成本值
        mav_cost = float(cc.get("mav_transport_cost", 0.0))
        p_wait_cost = float(cc.get("passenger_waiting_cost", 0.0))
        f_wait_cost = float(cc.get("freight_waiting_cost", 0.0))
        unserved_p_cost = float(cc.get("unserved_penalty_cost", 0.0))
        unserved_p_num = float(cc.get("unserved_passengers", 0.0))
        unserved_f_num = float(cc.get("unserved_freights", 0.0))

        # 2. 计算派生值 (总等待时间成本)
        total_wait_cost = p_wait_cost + f_wait_cost

        # 3. 记录适应度 (总成本)
        cost_history["fitness"].append(best.fitness.values[0])

        # 4. 按统一顺序记录所有成本
        cost_history["mav_transport"].append(mav_cost)
        cost_history["waiting_time_cost"].append(total_wait_cost)  # <--- 记录新增字段
        cost_history["passenger_waiting"].append(p_wait_cost)
        cost_history["freight_waiting"].append(f_wait_cost)
        cost_history["unserved_penalty_cost"].append(unserved_p_cost)
        cost_history["unserved_passenger"].append(unserved_p_num)
        cost_history["unserved_freight"].append(unserved_f_num)
        # --- 逻辑结束 ---

        # cost_history["mav_transport"].append(float(cc.get("mav_transport_cost", 0.0)))
        # cost_history["passenger_waiting"].append(float(cc.get("passenger_waiting_cost", 0.0)))
        # cost_history["freight_waiting"].append(float(cc.get("freight_waiting_cost", 0.0)))
        # cost_history["unserved_penalty_cost"].append(float(cc.get("unserved_penalty_cost", 0.0)))
        # cost_history["unserved_passenger"].append(float(cc.get("unserved_passengers", 0.0)))
        # cost_history["unserved_freight"].append(float(cc.get("unserved_freights", 0.0)))

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

    # ==================== 新增：初始化收敛检查器 ====================
    # 记录到目前为止的最佳适应度
    best_fitness_so_far = gen_min if math.isfinite(gen_min) else float('inf')
    # 记录连续没有改善的代数
    generations_no_improvement = 0

    # 从 parameters 字典中获取配置，如果未提供则使用默认值
    patience = parameters.get('early_stopping_patience', 10)
    tolerance = parameters.get('early_stopping_tolerance', 1e-4)

    print(f"\n[收敛检查] 启动。耐心={patience}代, 阈值={tolerance}")
    print(f"[收敛检查] 第 0 代最优解: {best_fitness_so_far:.6f}")
    # ================================================================

    # 新增：记录第0代所有个体信息和平均值
    record_generation_data(population)
    # 记录最优个体成本
    record_best_cost(population)

    logbook.record(gen=0, nevals=len(population), avg=gen_avg, min=gen_min, max=gen_max)
    if verbose:
        print(logbook.stream)

    # ==================== 新增：初始化收敛代数变量 ====================
    convergence_generation = None
    # ================================================================

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

        # ==================== 新增：提前停止（收敛）检查 ====================
        if math.isfinite(gen_min):
            # 计算与历史最优解的差距
            improvement = best_fitness_so_far - gen_min

            if improvement > tolerance:
                # 1. 适应度有明显改善
                print(f"  [收敛检查] 第 {gen} 代发现新最优解: {gen_min:.6f} (改善: {improvement:.6f})")
                best_fitness_so_far = gen_min
                generations_no_improvement = 0
            else:
                # 2. 适应度没有明显改善
                generations_no_improvement += 1
                print(f"  [收敛检查] 第 {gen} 代未发现明显改善 (连续 {generations_no_improvement}/{patience} 代)")

            # 3. 检查是否达到停止条件
            if generations_no_improvement >= patience:
                print(f"\n--- 算法已收敛 ---")
                print(f"连续 {patience} 代最佳适应度未见明显改善 (阈值 {tolerance})。")
                print(f"在第 {gen} 代提前停止。")

                # ==================== 新增：记录收敛代数 ====================
                convergence_generation = gen
                # ==========================================================

                break  # <--- 关键：跳出 for 循环
        else:
            # 4. 如果当前代没有有效解，跳过检查
            print(f"  [收敛检查] 第 {gen} 代无有效解，跳过检查。")
        # ========================== 检查结束 ==========================

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
    return population, logbook, cost_history, all_individuals_history, generation_averages, convergence_generation


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

    # final_population, logbook, cost_history, all_individuals_history, generation_averages = customized_genetic_algorithm(
    final_population, logbook, cost_history, all_individuals_history, generation_averages, convergence_generation = customized_genetic_algorithm(
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

    print('遗传算法进化结束 ---绘制成本构成堆叠进化图---')

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

        # ==================== 新增代码块：开始 ====================
        # 新增：绘制平均成本构成堆叠图
        print("\n--- 正在绘制平均成本构成进化堆叠图 ---")
        try:
            # 确保字体设置（如果前面失败，这里可以再次尝试）
            matplotlib.rcParams['font.family'] = 'SimHei'
            matplotlib.rcParams['axes.unicode_minus'] = False

            # 1. 定义一个新的保存路径
            avg_save_path = os.path.join(results_dir, "平均成本构成堆叠图.png")

            # 2. 关键：使用 generation_averages 字典作为数据源
            # 3. 提供一个新的标题
            plot_cost_stack_from_history(
                generation_averages,
                title="平均成本构成进化堆叠图",
                save_path=avg_save_path
            )

            print(f"✅ 平均成本构成堆叠图已成功保存到: {avg_save_path}")
        except Exception as e:
            print(f"❌ 绘制平均成本构成堆叠图时发生错误: {e}")
        # ==================== 新增代码块：结束 ====================

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
    # 返回新增的历史记录和收敛代数
    return final_population, logbook, cost_history, all_individuals_history, generation_averages, convergence_generation
    # return final_population, logbook, cost_history, all_individuals_history, generation_averages


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