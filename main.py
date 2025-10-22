# 注意指定需求数据


# 主程序入口
from deap import tools

import time
import copy
from datetime import datetime
import os

# 导入配置和数据加载
from config import parameters
from demand_loader import load_global_demand_data

# 导入重构后的函数
from deap_toolbox_setup import setup_deap_toolbox
from result_analysis import (
    analyze_and_save_best_individual,
    save_best_individual_results,
    generate_summary_report,
    print_solution
)


def main():

    start_time = time.time()  # ⏱ 记录程序开始时间

    """主程序 - 使用仿真生成的module_adjustments"""
    print("🚌 公交车模块调度优化系统")
    print("=" * 60)

    try:
        # 步骤1: 加载配置和数据
        print("\n--- 步骤1: 加载配置和数据 ---")

        print("🔄 正在加载需求数据...")

        # 指定需求数据文件路径（需要先运行data_generator.py生成）
        demand_data_file = "需求数据_demand_data_20251022_154125.json"  # 请根据实际生成的文件名修改

        # print("🔄 需求数据加载成功...", demand_data_file)

        try:
            global_demand_data, raw_data = load_global_demand_data(demand_data_file, parameters)
            print("✅ 需求数据加载成功")

            # 获取需求矩阵的副本
            a_matrix_p_up = copy.deepcopy(global_demand_data["a_matrix_p_up"])
            a_matrix_f_up = copy.deepcopy(global_demand_data["a_matrix_f_up"])
            a_matrix_p_down = copy.deepcopy(global_demand_data["a_matrix_p_down"])
            a_matrix_f_down = copy.deepcopy(global_demand_data["a_matrix_f_down"])

            # 计算剩余需求（分别计算上行和下行）
            all_passengers = 0
            all_freights = 0

            all_passengers_up = 0
            all_freights_up = 0

            all_passengers_down = 0
            all_freights_down = 0

            # 计算上行剩余需求
            for s in a_matrix_p_up:
                for s_dest in a_matrix_p_up[s]:
                    for t in a_matrix_p_up[s][s_dest]:
                        all_passengers += a_matrix_p_up[s][s_dest][t]

            all_passengers_up = all_passengers

            for s in a_matrix_f_up:
                for s_dest in a_matrix_f_up[s]:
                    for t in a_matrix_f_up[s][s_dest]:
                        all_freights += a_matrix_f_up[s][s_dest][t]

            all_freights_up = all_freights

            # 计算下行剩余需求
            for s in a_matrix_p_down:
                for s_dest in a_matrix_p_down[s]:
                    for t in a_matrix_p_down[s][s_dest]:
                        all_passengers += a_matrix_p_down[s][s_dest][t]

            all_passengers_down = all_passengers - all_passengers_up

            for s in a_matrix_f_down:
                for s_dest in a_matrix_f_down[s]:
                    for t in a_matrix_f_down[s][s_dest]:
                        all_freights += a_matrix_f_down[s][s_dest][t]

            all_freights_down = all_freights - all_freights_up

            print(f"✅ 仿真之前 ")
            print(f"   up总乘客: {all_passengers_up}, up总货物: {all_freights_up}")
            print(f"   down总乘客: {all_passengers_down}, down总货物: {all_freights_down}")
            print(f"   系统总乘客: {all_passengers}, 系统总货物: {all_freights}")

        except FileNotFoundError:
            print(f"⚠️ 需求数据文件 {demand_data_file} 不存在")
            print("请先运行 python data_generator.py 生成需求数据")
            return

        except Exception as e:
            print(f"⚠️ 需求数据加载失败: {e}")
            return

        # 步骤2: 设置DEAP工具箱
        print("\n--- 步骤2: 设置评估函数 ---")
        toolbox = setup_deap_toolbox(parameters, global_demand_data)
        print("✅ 工具箱设置完成")

        # 步骤3: 设置遗传算法参数
        print("\n--- 步骤3: 设置遗传算法参数 ---")
        ga_params = {
            'population_size': parameters['NUM_POPSIZE'],  # 种群大小
            'num_vehicles': parameters['NUM_VEHICLES'],
            'max_modules': parameters['max_modules'],

            'cxpb': parameters['cxpb'],  # 交叉概率
            'mutpb': parameters['mutpb'],  # 变异概率

            'ngen': parameters['MAX_GENERATIONS'],  # 进化代数
            'headway_range': (parameters['min_headway'], parameters['max_headway']),
            'verbose': True,

            'num_HallOfFame': parameters['num_HallOfFame']
        }

        print(f"种群大小: {ga_params['population_size']}")
        print(f"交叉概率: {ga_params['cxpb']}")
        print(f"变异概率: {ga_params['mutpb']}")
        print(f"进化代数: {ga_params['ngen']}")

        # 步骤4: 创建统计和名人堂
        print("\n--- 步骤4: 设置统计和名人堂 ---")
        # DEAP框架中的统计类
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(x) / len(x) if x else float('nan'))
        stats.register("min", min)
        stats.register("max", max)

        # halloffame = tools.HallOfFame(10)  # 保存最好的10个个体
        halloffame = tools.HallOfFame(ga_params['num_HallOfFame'])  # 保存最好的10个个体
        print("✅ 统计和名人堂设置完成")

        # ==================== 1. 在这里新增创建目录的逻辑 ====================
        # 使用时间戳创建一个唯一的结果目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"best_solution_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"结果将保存到目录: {results_dir}")
        # =================================================================

        # 步骤5: 运行遗传算法
        print("\n--- 步骤5: 运行遗传算法优化 ---")
        print("🚀 调用 run_genetic_algorithm_with_initialization 函数...")

        from customized_genetic_algorithm import run_genetic_algorithm_with_initialization

        final_population, logbook, cost_history, all_individuals_history, generation_averages = run_genetic_algorithm_with_initialization(
            population_size=ga_params['population_size'],
            num_vehicles=ga_params['num_vehicles'],
            max_modules=ga_params['max_modules'],
            toolbox=toolbox,
            cxpb=ga_params['cxpb'],
            mutpb=ga_params['mutpb'],
            ngen=ga_params['ngen'],
            headway_range=ga_params['headway_range'],
            stats=stats,
            halloffame=halloffame,
            parameters=parameters,
            global_demand_data=global_demand_data,
            verbose=ga_params['verbose'],
            results_dir=results_dir  # <--- 2. 将创建的目录路径传递进去
        )

        # 步骤6: 输出结果概览
        print("\n--- 步骤6: 输出优化结果概览 ---")
        best_individual = print_solution(final_population, logbook)

        # if best_individual:
        #     # ==================== 2. 将 timestamp 和 results_dir 都传递下去 ====================
        #     analyze_and_save_best_individual(
        #         best_individual=best_individual,
        #         parameters=parameters,
        #         global_demand_data=global_demand_data,
        #         logbook=logbook,
        #         cost_history=cost_history,
        #         results_dir=results_dir,  # <-- 传递目录
        #         timestamp=timestamp  # <-- 新增：传递时间戳字符串
        #     )
        #     # ==============================================================================

        # 步骤7: 显示名人堂
        print("\n--- 步骤7: 名人堂（最佳个体） ---")
        if halloffame:
            for i, individual in enumerate(halloffame):
                print(f"第 {i+1} 名: 适应度 = {individual.fitness.values[0]:.6f}")

        # 步骤8: 详细分析和保存最佳个体
        if best_individual:
            print("\n--- 步骤8: 详细分析和保存最佳个体 ---")
            success = analyze_and_save_best_individual(best_individual, parameters, global_demand_data, logbook, cost_history, results_dir, timestamp)

            if success:
                print("✅ 最佳个体分析和保存完成")
            else:
                print("⚠️ 最佳个体分析过程中出现问题")

        elapsed_time = time.time() - start_time  # ⏱ 计算耗时
        print(f"\n🎉 优化完成！总耗时: {elapsed_time:.2f} 秒")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()