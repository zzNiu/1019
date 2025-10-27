# 结果分析模块
import json
import pandas as pd
from datetime import datetime
import os

from simulation_generate import simulate_and_evaluate_individual
from df_schedule_construct import reconstruct_schedule_dataframe
from plot_cost_stack import plot_cost_stack_from_history

# ==================== 新增：导入甘特图绘制函数 ====================
from result_gantt_plot import draw_station_bar_plot, generate_schedule_gantt_charts


# =================================================================

# def analyze_and_save_best_individual(best_individual, parameters, global_demand_data, logbook=None):
# def analyze_and_save_best_individual(best_individual, parameters, global_demand_data, logbook=None, cost_history=None, results_dir=None, timestamp=None):
def analyze_and_save_best_individual(best_individual, parameters, global_demand_data, logbook=None,
                                     cost_history=None, results_dir=None, timestamp=None,
                                     convergence_generation=None):

    """详细分析并保存最佳个体"""
    print(f"\n{'='*60}")
    print(f"🏆 最佳个体详细分析")
    print(f"{'='*60}")

    # 运行仿真获取详细结果
    print("🔄 正在运行最佳个体的详细仿真...")
    try:

        # ==================== 新增：计算初始总需求 ====================
        # (此逻辑从 main.py 复制而来，用于计算服务总量)
        a_matrix_p_up = global_demand_data["a_matrix_p_up"]
        a_matrix_f_up = global_demand_data["a_matrix_f_up"]
        a_matrix_p_down = global_demand_data["a_matrix_p_down"]
        a_matrix_f_down = global_demand_data["a_matrix_f_down"]

        all_passengers = 0
        all_freights = 0
        all_passengers_up = 0
        all_freights_up = 0
        all_passengers_down = 0
        all_freights_down = 0

        # 计算上行需求
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

        # 计算下行需求
        for s in a_matrix_p_down:
            for s_dest in a_matrix_p_down[s]:
                for t in a_matrix_p_down[s][s_dest]:
                    all_passengers += a_matrix_p_down[s][s_dest][t]
        # (注意：修复了 main.py 中的拼写错误 all_passengers_dwon)
        all_passengers_down = all_passengers - all_passengers_up

        for s in a_matrix_f_down:
            for s_dest in a_matrix_f_down[s]:
                for t in a_matrix_f_down[s][s_dest]:
                    all_freights += a_matrix_f_down[s][s_dest][t]
        all_freights_down = all_freights - all_freights_up
        # ==================== 初始需求计算结束 ====================

        vehicle_schedule, total_cost, remaining_passengers, remaining_freights, failure_records, df_enriched, module_analysis_records, cost_components = simulate_and_evaluate_individual(
            best_individual, parameters, global_demand_data,
            global_demand_data["passenger_demand_up"],
            global_demand_data["passenger_demand_down"],
            global_demand_data["freight_demand_up"],
            global_demand_data["freight_demand_down"]
        )

        print("✅ 仿真完成")

        # ==================== 新增：定义详细指标并按新格式打印 ====================

        # 1. 计算服务总量
        total_served_passengers = all_passengers - remaining_passengers
        total_served_freight = all_freights - remaining_freights

        # 2. 假设详细的剩余需求存储在 cost_components 中
        try:
            # (我们假设 cost_components 包含这些用于细分的键)
            remaining_passengers_up = cost_components.get('unserved_passengers_up', 0)
            remaining_passengers_down = cost_components.get('unserved_passengers_down', 0)
            remaining_freights_up = cost_components.get('unserved_freights_up', 0)
            remaining_freights_down = cost_components.get('unserved_freights_down', 0)
        except Exception:
            # B方案: 如果 cost_components 缺失或格式不对，则无法细分
            remaining_passengers_up = "未知"
            remaining_passengers_down = "未知"
            remaining_freights_up = "未知"
            remaining_freights_down = "未知"

        # 3. 按您要求的格式打印
        print(f"\n📊 基本性能指标:")
        print(f"  ✅ 仿真完成 - 总成本: {total_cost:.2f}")
        print(f"   系统服务乘客: {total_served_passengers}, 系统服务货物: {total_served_freight}")
        print(f"   up剩余乘客: {remaining_passengers_up}, up剩余货物: {remaining_freights_up}")
        print(f"   down剩余乘客: {remaining_passengers_down}, down剩余货物: {remaining_freights_down}")
        print(f"   系统剩余乘客: {remaining_passengers}, 系统剩余货物: {remaining_freights}")
        print(f"   失败记录数: {len(failure_records)}")  # (保留了原有的失败记录，这很重要)
        # ==================== 打印块替换完成 ====================
        # print(f"\n📊 基本性能指标:")
        # print(f"  总成本: {total_cost:.2f}")
        # print(f"  剩余乘客: {remaining_passengers}")
        # print(f"  剩余货物: {remaining_freights}")
        # print(f"  失败记录数: {len(failure_records)}")

        # 车辆调度信息
        print(f"\n🚌 车辆调度详情:")

        # 上行车辆
        print(f"\n  上行方向 ({len(best_individual['up']['vehicle_dispatch'])} 辆车):")
        total_up_modules = 0
        for vid, dispatch_info in best_individual['up']['vehicle_dispatch'].items():
            allocation = best_individual['up']['initial_allocation'][vid]
            total_modules = allocation['passenger_modules'] + allocation['freight_modules']
            total_up_modules += total_modules
            print(f"    车辆{vid}: 发车时间={dispatch_info['arrival_time']}分钟, "
                  f"车头时距={dispatch_info['headway']}分钟")
            print(f"           初始配置: 乘客模块={allocation['passenger_modules']}, "
                  f"货运模块={allocation['freight_modules']}, 总计={total_modules}")

        # 下行车辆
        print(f"\n  下行方向 ({len(best_individual['down']['vehicle_dispatch'])} 辆车):")
        total_down_modules = 0
        for vid, dispatch_info in best_individual['down']['vehicle_dispatch'].items():
            allocation = best_individual['down']['initial_allocation'][vid]
            total_modules = allocation['passenger_modules'] + allocation['freight_modules']
            total_down_modules += total_modules
            print(f"    车辆{vid}: 发车时间={dispatch_info['arrival_time']}分钟, "
                  f"车头时距={dispatch_info['headway']}分钟")
            print(f"           初始配置: 乘客模块={allocation['passenger_modules']}, "
                  f"货运模块={allocation['freight_modules']}, 总计={total_modules}")

        print(f"\n  总模块使用: 上行={total_up_modules}, 下行={total_down_modules}, 总计={total_up_modules + total_down_modules}")

        # 生成时刻表
        print(f"\n📅 生成详细时刻表...")
        schedule_data = {}

        for direction in ['up', 'down']:
            try:
                df_schedule = reconstruct_schedule_dataframe(best_individual, parameters, direction)
                schedule_data[direction] = df_schedule
                print(f"  {direction}行时刻表: {len(df_schedule)} 条记录")
            except Exception as e:
                print(f"  ⚠️ 生成{direction}行时刻表失败: {e}")
                schedule_data[direction] = pd.DataFrame()

        # 保存结果
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 2. 修改对 save_best_individual_results 的调用，将 timestamp 传递下去
        save_best_individual_results(
            best_individual=best_individual,
            simulation_results={
                'total_cost': total_cost,
                'remaining_passengers': remaining_passengers,
                'remaining_freights': remaining_freights,
                'failure_records': failure_records,
                'df_enriched': df_enriched,
                'module_analysis_records': module_analysis_records,
                'schedule_data': schedule_data,
                'logbook': logbook,
                'cost_history': cost_history,

                # ==================== 在这里添加新行：开始 ====================
                # (这些变量是您在上一请求中计算的)
                'total_served_passengers': total_served_passengers,
                'total_served_freight': total_served_freight,
                'remaining_passengers_up': remaining_passengers_up,
                'remaining_passengers_down': remaining_passengers_down,
                'remaining_freights_up': remaining_freights_up,
                'remaining_freights_down': remaining_freights_down,
                # (同时传入 cost_components 以备后用)
                'cost_components': cost_components,
                # ==================== 添加新行：结束 ====================

                # ==================== 插入新代码：开始 ====================
                # (传递总需求数据)
                'total_passengers_up': all_passengers_up,
                'total_freights_up': all_freights_up,
                'total_passengers_down': all_passengers_down,
                'total_freights_down': all_freights_down,
                # ==================== 插入新代码：结束 ====================

                'convergence_generation': convergence_generation,  # <--- 在这里添加新行

            },
            results_dir=results_dir,  # <-- 传递目录
            timestamp=timestamp  # <-- 新增：传递时间戳
        )

        return True

    except Exception as e:
        print(f"❌ 分析最佳个体时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_best_individual_results(best_individual, simulation_results, results_dir, timestamp):
    """保存最佳个体的详细结果"""
    print(f"\n💾 保存最佳个体结果...")

    # # 创建结果目录
    # results_dir = f"best_solution_{timestamp}"
    # os.makedirs(results_dir, exist_ok=True)

    try:
        # 1. 保存个体基本信息
        individual_info = {
            'timestamp': timestamp,
            'fitness': best_individual.fitness.values[0] if hasattr(best_individual, 'fitness') else None,
            'total_cost': simulation_results['total_cost'],
            'remaining_passengers': simulation_results['remaining_passengers'],
            'remaining_freights': simulation_results['remaining_freights'],
            'up_direction': {
                'vehicle_dispatch': best_individual['up']['vehicle_dispatch'],
                'initial_allocation': best_individual['up']['initial_allocation']
            },
            'down_direction': {
                'vehicle_dispatch': best_individual['down']['vehicle_dispatch'],
                'initial_allocation': best_individual['down']['initial_allocation']
            }
        }

        # 添加模块调整信息（如果存在）
        if 'module_adjustments' in best_individual['up']:
            individual_info['up_direction']['module_adjustments'] = best_individual['up']['module_adjustments']
        if 'module_adjustments' in best_individual['down']:
            individual_info['down_direction']['module_adjustments'] = best_individual['down']['module_adjustments']

        with open(f"{results_dir}/best_individual.json", 'w', encoding='utf-8') as f:
            json.dump(individual_info, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 个体信息已保存到: {results_dir}/best_individual.json")

        # 2. 保存详细仿真结果
        if not simulation_results['df_enriched'].empty:
            simulation_results['df_enriched'].to_excel(f"{results_dir}/simulation_details.xlsx", index=False)
            print(f"  ✅ 仿真详情已保存到: {results_dir}/simulation_details.xlsx")

        # 3. 保存时刻表
        for direction, df_schedule in simulation_results['schedule_data'].items():
            if not df_schedule.empty:
                df_schedule.to_excel(f"{results_dir}/schedule_{direction}.xlsx", index=False)
                print(f"  ✅ {direction}行时刻表已保存到: {results_dir}/schedule_{direction}.xlsx")

        # 4. 保存失败记录
        if simulation_results['failure_records']:
            with open(f"{results_dir}/failure_records.json", 'w', encoding='utf-8') as f:
                json.dump(simulation_results['failure_records'], f, indent=2, ensure_ascii=False)
            print(f"  ✅ 失败记录已保存到: {results_dir}/failure_records.json")

        # 5. 保存进化历史
        if simulation_results['logbook']:
            logbook_data = []
            for record in simulation_results['logbook']:
                logbook_data.append(dict(record))

            with open(f"{results_dir}/evolution_history.json", 'w', encoding='utf-8') as f:
                json.dump(logbook_data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 进化历史已保存到: {results_dir}/evolution_history.json")

            # 生成详细的成本进化曲线
            try:
                from visualization import generate_comprehensive_cost_evolution_plot
                print(f"  🎨 生成成本进化曲线...")
                generate_comprehensive_cost_evolution_plot(simulation_results['logbook'], results_dir)
                print(f"  ✅ 成本进化曲线已保存到: {results_dir}/")
            except Exception as e:
                print(f"  ⚠️ 生成成本进化曲线失败: {e}")

        # 6. 生成总结报告
        print('  生成简略的总结报告')
        generate_summary_report(best_individual, simulation_results, f"{results_dir}/summary_report.txt")
        print(f"  ✅ 总结报告已保存到: {results_dir}/summary_report.txt")

        # ==================== 新增：调用详细报告 ====================
        try:
            print('  生成详细的总结报告')
            generate_summary_detail_report(best_individual, simulation_results,f"{results_dir}/summary_detail_report.txt")
            print(f"  ✅ 详细总结报告已保存到: {results_dir}/summary_detail_report.txt")
        except Exception as e:
            print(f"  ⚠️ 生成详细总结报告失败: {e}")
        # ==========================================================

        # ==================== 新增：调用甘特图绘制 ====================
        # 确保 simulation_details.xlsx 已经保存（或至少 df_enriched 已可用）
        if not simulation_results['df_enriched'].empty:
            try:
                print(f"  🎨 生成调度甘特图...")
                generate_schedule_gantt_charts(
                    simulation_details_df=simulation_results['df_enriched'],
                    save_dir=results_dir
                )
                print(f"  ✅ 调度甘特图已保存到: {results_dir}/")
            except Exception as e:
                print(f"  ⚠️ 生成调度甘特图失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ℹ️ 跳过甘特图绘制，因为仿真详情 (df_enriched) 为空。")
        # ==========================================================

        print(f"\n🎉 所有结果已保存到目录: {results_dir}")
        return results_dir

    except Exception as e:
        print(f"❌ 保存结果时出错: {e}")
        return None


def print_solution(final_population, logbook):
    """打印解决方案（简化版本）"""
    if not final_population:
        print("❌ 没有找到有效解决方案")
        return None

    # 找到最佳个体
    best_individual = min(final_population, key=lambda x: x.fitness.values[0])

    print(f"\n=== 最优解决方案概览 ===")
    print(f"最佳适应度: {best_individual.fitness.values[0]:.6f}")

    # 简要车辆信息
    up_vehicles = len(best_individual['up']['vehicle_dispatch'])
    down_vehicles = len(best_individual['down']['vehicle_dispatch'])
    print(f"车辆配置: 上行{up_vehicles}辆, 下行{down_vehicles}辆")

    # 进化统计信息
    if logbook:
        print(f"进化代数: {len(logbook)}")
        final_stats = logbook[-1]
        print(f"最终代适应度范围: {final_stats['min']:.6f} - {final_stats['max']:.6f}")

    return best_individual


def generate_summary_detail_report(best_individual, simulation_results, filepath):
    """生成总结报告（超详细版）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("🏆 最佳调度方案总结报告 (详细版)\n")
        f.write("=" * 80 + "\n\n")

        # --- 1. 基本性能指标 ---
        f.write("📊 基本性能指标:\n")
        f.write(f"  总成本 (Fitness): {simulation_results.get('total_cost', 0):.2f}\n")
        f.write(f"  系统总服务乘客: {simulation_results.get('total_served_passengers', '未知')}\n")
        f.write(f"  系统总服务货物: {simulation_results.get('total_served_freight', '未知')}\n")
        f.write(f"  系统总剩余乘客: {simulation_results.get('remaining_passengers', '未知')}\n")
        f.write(f"  系统总剩余货物: {simulation_results.get('remaining_freights', '未知')}\n")
        failure_records = simulation_results.get('failure_records', [])
        f.write(f"  仿真失败记录数: {len(failure_records)}\n\n")

        # --- 2. 需求服务详情 (新增模块) ---
        f.write("📈 需求服务详情:\n")

        # (从 simulation_results 获取总需求数据 - 这是第1步添加的)
        total_p_up = simulation_results.get('total_passengers_up', 0)
        total_f_up = simulation_results.get('total_freights_up', 0)
        total_p_down = simulation_results.get('total_passengers_down', 0)
        total_f_down = simulation_results.get('total_freights_down', 0)

        # (获取未服务数据)
        rem_p_up = simulation_results.get('remaining_passengers_up', 0)
        rem_f_up = simulation_results.get('remaining_freights_up', 0)
        rem_p_down = simulation_results.get('remaining_passengers_down', 0)
        rem_f_down = simulation_results.get('remaining_freights_down', 0)

        # (计算服务量)
        served_p_up = total_p_up - rem_p_up
        served_f_up = total_f_up - rem_f_up
        served_p_down = total_p_down - rem_p_down
        served_f_down = total_f_down - rem_f_down

        f.write(f"  上行乘客 (Up P):   总需求={total_p_up}, 已服务={served_p_up}, 未服务={rem_p_up}\n")
        f.write(f"  上行货物 (Up F):   总需求={total_f_up}, 已服务={served_f_up}, 未服务={rem_f_up}\n")
        f.write(f"  下行乘客 (Down P): 总需求={total_p_down}, 已服务={served_p_down}, 未服务={rem_p_down}\n")
        f.write(f"  下行货物 (Down F): 总需求={total_f_down}, 已服务={served_f_down}, 未服务={rem_f_down}\n\n")

        # --- 3. 成本详细分解 (使用修正后的键名) ---
        f.write("💰 成本详细分解:\n")
        cost_comp = simulation_results.get('cost_components')
        if cost_comp:
            f.write(f"  MAV能量消耗: {cost_comp.get('mav_transport_cost', 0):.2f}\n")
            f.write(f"  总等待成本: {cost_comp.get('waiting_cost', 0):.2f}\n")
            f.write(f"    - 乘客等待成本: {cost_comp.get('passenger_waiting_cost', 0):.2f}\n")
            f.write(f"    - 货物等待成本: {cost_comp.get('freight_waiting_cost', 0):.2f}\n")
            f.write(f"  惩罚成本 (总计): {cost_comp.get('unserved_penalty_cost', 0):.2f}\n")
            f.write(f"    - 未服务乘客 (人数): {cost_comp.get('unserved_passengers', 0):.0f}\n")
            f.write(f"    - 未服务货物 (件数): {cost_comp.get('unserved_freights', 0):.0f}\n\n")
        else:
            f.write("  (未找到详细的成本分解 'cost_components')\n\n")

        # --- 4. 车辆配置统计 (初始配置) ---
        f.write("🚌 车辆配置统计 (初始配置):\n")
        # (这部分代码与上一版相同，保持不变)
        total_p_modules = 0
        total_f_modules = 0
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            f.write(f"\n  {direction_name}方向:\n")
            vehicle_dispatch = best_individual[direction]['vehicle_dispatch']
            initial_allocation = best_individual[direction]['initial_allocation']
            total_vehicles = len(vehicle_dispatch)
            dir_passenger_modules = sum(alloc['passenger_modules'] for alloc in initial_allocation.values())
            dir_freight_modules = sum(alloc['freight_modules'] for alloc in initial_allocation.values())
            total_p_modules += dir_passenger_modules
            total_f_modules += dir_freight_modules
            f.write(f"    车辆数量: {total_vehicles}\n")
            f.write(f"    总乘客模块: {dir_passenger_modules}\n")
            f.write(f"    总货运模块: {dir_freight_modules}\n")
            headways = [dispatch['headway'] for dispatch in vehicle_dispatch.values()]
            if headways:
                f.write(f"    车头时距范围: {min(headways):.1f} - {max(headways):.1f} 分钟\n")
            else:
                f.write("    (无车辆发车)\n")
        f.write(f"\n  系统总计:\n")
        f.write(f"    总乘客模块 (初始): {total_p_modules}\n")
        f.write(f"    总货运模块 (初始): {total_f_modules}\n\n")

        # --- 5. 详细车辆信息 (初始配置) ---
        f.write("🚗 详细车辆信息 (初始配置):\n")
        # (这部分代码与上一版相同，保持不变)
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            f.write(f"\n  {direction_name}方向车辆:\n")
            if not best_individual[direction]['vehicle_dispatch']:
                f.write("    (无车辆)\n")
                continue
            for vid, dispatch_info in best_individual[direction]['vehicle_dispatch'].items():
                allocation = best_individual[direction]['initial_allocation'][vid]
                f.write(f"    车辆{vid}: 发车时间={dispatch_info['arrival_time']}分钟, "
                        f"车头时距={dispatch_info['headway']}分钟, "
                        f"乘客模块={allocation['passenger_modules']}, "
                        f"货运模块={allocation['freight_modules']}\n")
        f.write("\n")

        # --- 6. 模块调整总结 ---
        f.write("🔄 模块调整总结 (如果启用):\n")
        # (这部分代码与上一版相同，保持不变)
        adjustment_found = False
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            if 'module_adjustments' in best_individual[direction] and best_individual[direction]['module_adjustments']:
                adjustment_found = True
                f.write(f"  {direction_name}方向:\n")
                for adj_rule in best_individual[direction]['module_adjustments']:
                    f.write(f"    - 规则: {adj_rule}\n")
            else:
                f.write(f"  {direction_name}方向: (无调整规则)\n")
        if not adjustment_found:
            f.write("  (未启用或未生成模块调整规则)\n")
        f.write("\n")

        # --- 7. 进化过程 ---
        f.write("🧬 进化过程:\n")
        # (这部分代码与上一版相同，保持不变)
        if simulation_results.get('logbook'):
            logbook = simulation_results['logbook']
            f.write(f"  总代数: {len(logbook)}\n")
            first_gen = logbook[0]
            last_gen = logbook[-1]
            try:
                first_min_str = first_gen.get('min', 'N/A')
                last_min_str = last_gen.get('min', 'N/A')
                initial_min = float(first_min_str)
                final_min = float(last_min_str)
                f.write(f"  初始代最佳适应度: {initial_min:.6f}\n")
                f.write(f"  最终代最佳适应度: {final_min:.6f}\n")
                if initial_min > 0:
                    improvement = ((initial_min - final_min) / initial_min * 100)
                    f.write(f"  改进幅度: {improvement:.2f}%\n")
                else:
                    f.write("  改进幅度: N/A (初始成本为0)\n")
            except (ValueError, TypeError):
                f.write("  (无法计算适应度改进)\n")
            convergence_gen = simulation_results.get('convergence_generation')
            if convergence_gen is not None:
                f.write(f"  收敛状态: 在第 {convergence_gen} 代提前停止 (已收敛)\n")
            else:
                total_run = len(logbook) - 1
                f.write(f"  收敛状态: 运行至最大代数 {total_run} (未提前停止)\n")
        else:
            f.write("  (无 Logbook 信息)\n")
        f.write("\n")

        # --- 8. 详细车辆/站点日志 (新增模块) ---
        f.write("=" * 80 + "\n")
        f.write("Detailed Vehicle & Station Log (来自 simulation_details.xlsx)\n")
        f.write("=" * 80 + "\n")

        df_log = simulation_results.get('df_enriched')

        if df_log is None or df_log.empty:
            f.write("  (无详细仿真日志 'df_enriched' 可用)\n")
        else:
            # (假设 'df_enriched' 包含这些列)
            # (您可能需要根据您的 'df_enriched' (即 simulation_details.xlsx) 中的实际列名调整这里的列名)
            REQUIRED_COLS = [
                'vid', 'direction', 'station_id', 'arrival_time', 'departure_time',
                'p_modules_onboard', 'f_modules_onboard',
                'p_modules_added', 'p_modules_removed',
                'f_modules_added', 'f_modules_removed',
                'station_p_module_stock', 'station_f_module_stock'
            ]

            # 检查列是否存在
            missing_cols = [col for col in REQUIRED_COLS if col not in df_log.columns]
            if missing_cols:
                f.write(f"  (警告: df_enriched 缺少以下关键列，无法生成详细日志: {missing_cols})\n")
            else:
                # 按车辆ID和时间排序
                df_log_sorted = df_log.sort_values(by=['vid', 'arrival_time'])

                current_vid = None
                for _, row in df_log_sorted.iterrows():
                    vid = row['vid']
                    if vid != current_vid:
                        # 打印新车辆的表头
                        current_vid = vid
                        f.write(f"\n--- 车辆 {vid} (方向: {row['direction']}) ---\n")
                        f.write(
                            f"  站点 |  到达 |  出发 |"
                            f" 车载(P/F) | 增减(P) | 增减(F) |"
                            f" 站点库存(P/F)\n"
                        )
                        f.write(f"  " + "-" * 75 + "\n")

                    # 格式化数据
                    station = f"{row['station_id']:>4}"
                    arr_time = f"{row['arrival_time']:>6.1f}"
                    dep_time = f"{row['departure_time']:>6.1f}"

                    onboard = f"{int(row['p_modules_onboard']):>2}/{int(row['f_modules_onboard']):<2}"

                    p_change = f"+{int(row['p_modules_added'])}/-{int(row['p_modules_removed'])}"
                    f_change = f"+{int(row['f_modules_added'])}/-{int(row['f_modules_removed'])}"

                    stock = f"{int(row['station_p_module_stock']):>2}/{int(row['station_f_module_stock']):<2}"

                    # 打印行
                    f.write(
                        f"  {station} | {arr_time} | {dep_time} |"
                        f" {onboard:<7} | {p_change:<7} | {f_change:<7} |"
                        f" {stock:<10}\n"
                    )

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")

def generate_summary_report(best_individual, simulation_results, filepath):
    """生成总结报告（增强版）"""
    print('生成总结报告')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("🏆 最佳调度方案总结报告\n")
        f.write("=" * 60 + "\n\n")

        # --- 1. 基本性能指标 ---
        f.write("📊 基本性能指标:\n")
        f.write(f"  总成本 (Fitness): {simulation_results.get('total_cost', 0):.2f}\n")  # 使用 .get 增加安全性

        # 从 simulation_results 中获取服务数据
        total_served_p = simulation_results.get('total_served_passengers', '未知')
        total_served_f = simulation_results.get('total_served_freight', '未知')
        # rem_p_up = simulation_results.get('remaining_passengers_up', '未知')
        # rem_p_down = simulation_results.get('remaining_passengers_down', '未知')
        # rem_f_up = simulation_results.get('remaining_freights_up', '未知')
        # rem_f_down = simulation_results.get('remaining_freights_down', '未知')

        f.write(f"  系统服务乘客: {total_served_p}\n")
        f.write(f"  系统服务货物: {total_served_f}\n")

        f.write(f"  系统总剩余乘客: {simulation_results.get('remaining_passengers', '未知')}\n")
        f.write(f"  系统总剩余货物: {simulation_results.get('remaining_freights', '未知')}\n")

        # failure_records = simulation_results.get('failure_records', [])
        # f.write(f"  仿真失败记录数: {len(failure_records)}\n\n")

        # --- 2. 成本详细分解 (!! 错误修复 !!) ---
        f.write("💰 成本详细分解:\n")
        # (错误修复：这里使用 cost_components, 而不是 cost_history)
        cost_comp = simulation_results.get('cost_components')
        if cost_comp:
            f.write(f"  MAV能量消耗: {cost_comp.get('mav_transport_cost', 0):.2f}\n")
            f.write(f"  总等待成本: {cost_comp.get('waiting_cost', 0):.2f}\n")
            f.write(f"    - 乘客等待成本: {cost_comp.get('passenger_waiting_cost', 0):.2f}\n")
            f.write(f"    - 货物等待成本: {cost_comp.get('freight_waiting_cost', 0):.2f}\n")
            f.write(f"  惩罚成本: {cost_comp.get('unserved_penalty_cost', 0):.2f}\n")
            f.write(f"    - 乘客未服务惩罚: {cost_comp.get('unserved_passengers', 0):.2f}\n")
            f.write(f"    - 货物未服务惩罚: {cost_comp.get('unserved_freights', 0):.2f}\n\n")
        else:
            f.write("  (未找到详细的成本分解 'cost_components')\n\n")

        # --- 3. 车辆配置统计 ---
        f.write("🚌 车辆配置统计:\n")
        total_p_modules = 0
        total_f_modules = 0

        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            f.write(f"\n  {direction_name}方向:\n")

            vehicle_dispatch = best_individual[direction]['vehicle_dispatch']
            initial_allocation = best_individual[direction]['initial_allocation']

            total_vehicles = len(vehicle_dispatch)
            dir_passenger_modules = sum(alloc['passenger_modules'] for alloc in initial_allocation.values())
            dir_freight_modules = sum(alloc['freight_modules'] for alloc in initial_allocation.values())
            total_p_modules += dir_passenger_modules
            total_f_modules += dir_freight_modules

            f.write(f"    车辆数量: {total_vehicles}\n")
            f.write(f"    总乘客模块: {dir_passenger_modules}\n")
            f.write(f"    总货运模块: {dir_freight_modules}\n")
            f.write(f"    总模块数: {dir_passenger_modules + dir_freight_modules}\n")

            # 计算车头时距
            headways = [dispatch['headway'] for dispatch in vehicle_dispatch.values()]
            if headways:
                f.write(f"    车头时距范围: {min(headways):.1f} - {max(headways):.1f} 分钟\n")
                f.write(f"    平均车头时距: {sum(headways) / len(headways):.1f} 分钟\n")
            else:
                f.write("    (无车辆发车)\n")

        f.write(f"\n  系统总计:\n")
        f.write(f"    总乘客模块: {total_p_modules}\n")
        f.write(f"    总货运模块: {total_f_modules}\n")
        f.write(f"    总模块数 (初始): {total_p_modules + total_f_modules}\n\n")

        # --- 4. 详细车辆信息 ---
        f.write("🚗 详细车辆信息:\n")
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            f.write(f"\n  {direction_name}方向车辆:\n")

            if not best_individual[direction]['vehicle_dispatch']:
                f.write("    (无车辆)\n")
                continue

            for vid, dispatch_info in best_individual[direction]['vehicle_dispatch'].items():
                allocation = best_individual[direction]['initial_allocation'][vid]
                f.write(f"    车辆{vid}: 发车时间={dispatch_info['arrival_time']}分钟, "
                        f"车头时距={dispatch_info['headway']}分钟, "
                        f"乘客模块={allocation['passenger_modules']}, "
                        f"货运模块={allocation['freight_modules']}\n")

        # --- 5. 模块调整总结 (新增逻辑) ---
        f.write(f"\n🔄 模块调整总结 (如果启用):\n")
        adjustment_found = False
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            # 检查 'module_adjustments' 是否存在且不为空
            if 'module_adjustments' in best_individual[direction] and best_individual[direction]['module_adjustments']:
                adjustment_found = True
                f.write(f"  {direction_name}方向:\n")
                # (假设 module_adjustments 是一个列表，内容可以被打印)
                for adj_rule in best_individual[direction]['module_adjustments']:
                    f.write(f"    - 规则: {adj_rule}\n")
            else:
                f.write(f"  {direction_name}方向: (无调整规则)\n")

        if not adjustment_found:
            f.write("  (未启用或未生成模块调整规则)\n")

        # --- 6. 进化过程 ---
        if simulation_results.get('logbook'):
            f.write(f"\n📈 进化过程:\n")
            logbook = simulation_results['logbook']
            f.write(f"  总代数: {len(logbook)}\n")  # (logbook 长度是 N+1, 包含第0代)

            first_gen = logbook[0]
            last_gen = logbook[-1]

            # 使用 .get 和类型转换来确保安全
            try:
                first_min_str = first_gen.get('min', 'N/A')
                last_min_str = last_gen.get('min', 'N/A')

                f.write(f"  初始代最佳适应度: {float(first_min_str):.6f}\n")
                f.write(f"  最终代最佳适应度: {float(last_min_str):.6f}\n")

                initial_min = float(first_min_str)
                final_min = float(last_min_str)

                if initial_min > 0:  # 避免除以零
                    improvement = ((initial_min - final_min) / initial_min * 100)
                    f.write(f"  改进幅度: {improvement:.2f}%\n")
                else:
                    f.write("  改进幅度: N/A (初始成本为0)\n")
            except (ValueError, TypeError):
                f.write(f"  初始代最佳适应度: {first_min_str}\n")
                f.write(f"  最终代最佳适应度: {last_min_str}\n")
                f.write("  改进幅度: N/A (数据格式错误)\n")

            # --- 写入收敛信息 ---
            convergence_gen = simulation_results.get('convergence_generation')
            if convergence_gen is not None:
                f.write(f"  收敛状态: 在第 {convergence_gen} 代提前停止 (已收敛)\n")
            else:
                total_run = len(logbook) - 1  # 实际运行的代数
                f.write(f"  收敛状态: 运行至最大代数 {total_run} (未提前停止)\n")
        else:
            f.write(f"\n📈 进化过程: (无 Logbook 信息)\n")

        # (清除了之前发现的重复行 "详细车辆信息")
        f.write("\n" + "=" * 60 + "\n")
        f.write("报告结束\n")
        f.write("=" * 60 + "\n")

# def generate_summary_report(best_individual, simulation_results, filepath):
#     """生成总结报告"""
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write("="*60 + "\n")
#         f.write("🏆 最佳调度方案总结报告\n")
#         f.write("="*60 + "\n\n")
#
#         # # 基本信息
#         # (这是替换后的新代码块)
#         f.write("📊 基本性能指标:\n")
#         f.write(f"  总成本: {simulation_results['total_cost']:.2f}\n")
#
#         f.write(f"  总成本: {simulation_results['cost_history']['fitness']:.2f}\n")
#
#         f.write(f"  MAV能量消耗: {simulation_results['cost_history']['mav_transport']:.2f}\n")
#         f.write(f"  总等待成本: {simulation_results['cost_history']['waiting_time_cost']:.2f}\n")
#         f.write(f"  乘客等待成本: {simulation_results['cost_history']['passenger_waiting']:.2f}\n")
#         f.write(f"  货物等待成本: {simulation_results['cost_history']['freight_waiting']:.2f}\n")
#         f.write(f"  惩罚成本: {simulation_results['cost_history']['unserved_penalty_cost']:.2f}\n")
#         f.write(f"  乘客未服务: {simulation_results['cost_history']['unserved_passenger']:.2f}\n")
#         f.write(f"  货物未服务: {simulation_results['cost_history']['unserved_freight']:.2f}\n")
#
#         # --- 从 simulation_results 中获取新数据 ---
#         # (使用 .get() 方法，如果键不存在则返回 '未知'，确保安全)
#         total_served_p = simulation_results.get('total_served_passengers', '未知')
#         total_served_f = simulation_results.get('total_served_freight', '未知')
#         rem_p_up = simulation_results.get('remaining_passengers_up', '未知')
#         rem_p_down = simulation_results.get('remaining_passengers_down', '未知')
#         rem_f_up = simulation_results.get('remaining_freights_up', '未知')
#         rem_f_down = simulation_results.get('remaining_freights_down', '未知')
#
#         # --- 按照您要求的格式写入文件 ---
#         f.write(f"   系统服务乘客: {total_served_p}, 系统服务货物: {total_served_f}\n")
#         f.write(f"   up剩余乘客: {rem_p_up}, up剩余货物: {rem_f_up}\n")
#         f.write(f"   down剩余乘客: {rem_p_down}, down剩余货物: {rem_f_down}\n")
#
#         # --- 保留原有的系统总剩余和失败记录 ---
#         f.write(f"   系统剩余乘客: {simulation_results['remaining_passengers']}\n")
#         f.write(f"   系统剩余货物: {simulation_results['remaining_freights']}\n")
#         f.write(f"   失败记录数: {len(simulation_results['failure_records'])}\n\n")
#         # f.write("📊 基本性能指标:\n")
#         # f.write(f"  总成本: {simulation_results['total_cost']:.2f}\n")
#         # f.write(f"  剩余乘客: {simulation_results['remaining_passengers']}\n")
#         # f.write(f"  剩余货物: {simulation_results['remaining_freights']}\n")
#         # f.write(f"  失败记录数: {len(simulation_results['failure_records'])}\n\n")
#
#         # 车辆配置统计
#         f.write("🚌 车辆配置统计:\n")
#
#         for direction in ['up', 'down']:
#             direction_name = "上行" if direction == "up" else "下行"
#             f.write(f"\n  {direction_name}方向:\n")
#
#             vehicle_dispatch = best_individual[direction]['vehicle_dispatch']
#             initial_allocation = best_individual[direction]['initial_allocation']
#
#             total_vehicles = len(vehicle_dispatch)
#             total_passenger_modules = sum(alloc['passenger_modules'] for alloc in initial_allocation.values())
#             total_freight_modules = sum(alloc['freight_modules'] for alloc in initial_allocation.values())
#
#             f.write(f"    车辆数量: {total_vehicles}\n")
#             f.write(f"    总乘客模块: {total_passenger_modules}\n")
#             f.write(f"    总货运模块: {total_freight_modules}\n")
#             f.write(f"    总模块数: {total_passenger_modules + total_freight_modules}\n")
#
#             # 车头时距统计
#             headways = [dispatch['headway'] for dispatch in vehicle_dispatch.values()]
#             f.write(f"    车头时距范围: {min(headways):.1f} - {max(headways):.1f} 分钟\n")
#             f.write(f"    平均车头时距: {sum(headways)/len(headways):.1f} 分钟\n")
#
#         # 详细车辆信息
#         f.write("\n🚗 详细车辆信息:\n")
#         for direction in ['up', 'down']:
#             direction_name = "上行" if direction == "up" else "下行"
#             f.write(f"\n  {direction_name}方向车辆:\n")
#
#             for vid, dispatch_info in best_individual[direction]['vehicle_dispatch'].items():
#                 allocation = best_individual[direction]['initial_allocation'][vid]
#                 f.write(f"    车辆{vid}: 发车时间={dispatch_info['arrival_time']}分钟, "
#                        f"车头时距={dispatch_info['headway']}分钟, "
#                        f"乘客模块={allocation['passenger_modules']}, "
#                        f"货运模块={allocation['freight_modules']}\n")
#
#         # 如果有进化历史，添加进化信息
#         if simulation_results['logbook']:
#             f.write(f"\n📈 进化过程:\n")
#             f.write(f"  总代数: {len(simulation_results['logbook'])}\n")
#
#             first_gen = simulation_results['logbook'][0]
#             last_gen = simulation_results['logbook'][-1]
#
#             f.write(f"  初始代最佳适应度: {first_gen['min']:.6f}\n")
#             f.write(f"  最终代最佳适应度: {last_gen['min']:.6f}\n")
#             f.write(f"  改进幅度: {((first_gen['min'] - last_gen['min']) / first_gen['min'] * 100):.2f}%\n")
#
#             # ==================== 新增：写入收敛信息 ====================
#             convergence_gen = simulation_results.get('convergence_generation')
#             if convergence_gen is not None:
#                 f.write(f"  收敛状态: 在第 {convergence_gen} 代提前停止 (已收敛)\n")
#             else:
#                 total_run = len(simulation_results['logbook']) - 1
#                 f.write(f"  收敛状态: 运行至最大代数 {total_run} (未提前停止)\n")
#             # ==========================================================
#
#             # (原有的代码)
#         f.write("\n🚗 详细车辆信息:\n")