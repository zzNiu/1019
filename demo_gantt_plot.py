#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
甘特图演示脚本 - 从现有解决方案生成详细的调度甘特图
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 导入甘特图绘制模块
from enhanced_gantt_plot import EnhancedGanttPlotter, generate_gantt_from_solution_dir


def find_latest_solution():
    """查找最新的解决方案目录"""
    solution_dirs = [d for d in os.listdir('.') if d.startswith('best_solution_')]
    if not solution_dirs:
        return None
    return max(solution_dirs)


def demo_gantt_from_existing_solution():
    """从现有解决方案演示甘特图功能"""
    print("🚌 甘特图演示 - 从现有解决方案")
    print("=" * 60)
    
    # 查找最新解决方案
    latest_solution = find_latest_solution()
    if not latest_solution:
        print("❌ 未找到解决方案目录")
        print("请先运行 python main.py 生成解决方案")
        return False
    
    print(f"📁 使用解决方案: {latest_solution}")
    
    try:
        # 创建输出目录
        output_dir = f"gantt_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 创建输出目录: {output_dir}")
        
        # 创建甘特图绘制器
        plotter = EnhancedGanttPlotter(solution_dir=latest_solution)
        
        if plotter.best_individual is None:
            print("❌ 无法加载解决方案数据")
            return False
        
        print("\n🎨 开始生成甘特图...")
        
        # 1. 生成综合甘特图
        print("  📊 生成综合甘特图...")
        try:
            fig1 = plotter.generate_comprehensive_gantt_chart(save_dir=output_dir)
            print("    ✅ 综合甘特图完成")
        except Exception as e:
            print(f"    ❌ 综合甘特图失败: {e}")
        
        # 2. 生成详细车辆甘特图
        print("  🚍 生成详细车辆甘特图...")
        try:
            fig2 = plotter.generate_detailed_vehicle_gantt(save_dir=output_dir)
            print("    ✅ 详细车辆甘特图完成")
        except Exception as e:
            print(f"    ❌ 详细车辆甘特图失败: {e}")
        
        # 3. 生成载荷分析图表
        print("  📈 生成载荷分析图表...")
        try:
            fig3 = plotter.generate_load_analysis_chart(save_dir=output_dir)
            print("    ✅ 载荷分析图表完成")
        except Exception as e:
            print(f"    ❌ 载荷分析图表失败: {e}")
        
        # 4. 生成统计报告
        print("  📋 生成统计报告...")
        try:
            generate_gantt_statistics_report(plotter, output_dir)
            print("    ✅ 统计报告完成")
        except Exception as e:
            print(f"    ❌ 统计报告失败: {e}")
        
        print(f"\n✅ 甘特图演示完成！")
        print(f"📁 所有结果已保存到: {output_dir}")
        
        # 列出生成的文件
        print(f"\n📄 生成的文件:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_gantt_statistics_report(plotter, output_dir):
    """生成甘特图统计报告"""
    report_path = os.path.join(output_dir, "gantt_statistics_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🚌 调度甘特图统计报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本信息
        if plotter.best_individual:
            f.write("📊 基本性能指标:\n")

            # 安全格式化数值
            total_cost = plotter.best_individual.get('total_cost', 'N/A')
            if isinstance(total_cost, (int, float)):
                f.write(f"  总成本: {total_cost:.2f}\n")
            else:
                f.write(f"  总成本: {total_cost}\n")

            fitness = plotter.best_individual.get('fitness', 'N/A')
            if isinstance(fitness, (int, float)):
                f.write(f"  适应度: {fitness:.6f}\n")
            else:
                f.write(f"  适应度: {fitness}\n")

            f.write(f"  剩余乘客: {plotter.best_individual.get('remaining_passengers', 'N/A')}\n")
            f.write(f"  剩余货物: {plotter.best_individual.get('remaining_freights', 'N/A')}\n\n")
        
        # 车辆统计
        f.write("🚍 车辆配置统计:\n")
        total_vehicles = 0
        total_p_modules = 0
        total_f_modules = 0
        
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"
            
            if plotter.best_individual and f'{direction}_direction' in plotter.best_individual:
                dir_data = plotter.best_individual[f'{direction}_direction']
                
                vehicle_dispatch = dir_data.get('vehicle_dispatch', {})
                initial_allocation = dir_data.get('initial_allocation', {})
                
                vehicle_count = len(vehicle_dispatch)
                p_modules = sum(alloc.get('passenger_modules', 0) for alloc in initial_allocation.values())
                f_modules = sum(alloc.get('freight_modules', 0) for alloc in initial_allocation.values())
                
                total_vehicles += vehicle_count
                total_p_modules += p_modules
                total_f_modules += f_modules
                
                f.write(f"\n  {direction_name}方向:\n")
                f.write(f"    车辆数量: {vehicle_count}\n")
                f.write(f"    乘客模块: {p_modules}\n")
                f.write(f"    货物模块: {f_modules}\n")
                f.write(f"    总模块: {p_modules + f_modules}\n")
                
                # 车头时距统计
                if vehicle_dispatch:
                    headways = [v.get('headway', 0) for v in vehicle_dispatch.values()]
                    if headways:
                        f.write(f"    平均车头时距: {np.mean(headways):.2f} 分钟\n")
                        f.write(f"    车头时距范围: {min(headways)} - {max(headways)} 分钟\n")
                
                # 详细车辆信息
                f.write(f"    详细车辆配置:\n")
                for vid, dispatch in vehicle_dispatch.items():
                    if vid in initial_allocation:
                        alloc = initial_allocation[vid]
                        f.write(f"      车辆{vid}: 发车时间={dispatch.get('arrival_time', 0)}分钟, "
                               f"车头时距={dispatch.get('headway', 0)}分钟, "
                               f"乘客模块={alloc.get('passenger_modules', 0)}, "
                               f"货物模块={alloc.get('freight_modules', 0)}\n")
        
        f.write(f"\n📈 总体统计:\n")
        f.write(f"  总车辆数: {total_vehicles}\n")
        f.write(f"  总乘客模块: {total_p_modules}\n")
        f.write(f"  总货物模块: {total_f_modules}\n")
        f.write(f"  总模块数: {total_p_modules + total_f_modules}\n")
        
        if total_vehicles > 0:
            f.write(f"  平均每车模块数: {(total_p_modules + total_f_modules) / total_vehicles:.2f}\n")
        
        # 时刻表统计
        if plotter.schedule_data:
            f.write(f"\n📅 时刻表统计:\n")
            for direction in ['up', 'down']:
                direction_name = "上行" if direction == "up" else "下行"
                if direction in plotter.schedule_data and not plotter.schedule_data[direction].empty:
                    df = plotter.schedule_data[direction]
                    f.write(f"  {direction_name}时刻表记录数: {len(df)}\n")
                    
                    if '到达时间' in df.columns:
                        times = pd.to_numeric(df['到达时间'], errors='coerce').dropna()
                        if len(times) > 0:
                            f.write(f"  {direction_name}运行时间范围: {times.min():.1f} - {times.max():.1f} 分钟\n")
                            f.write(f"  {direction_name}总运行时长: {times.max() - times.min():.1f} 分钟\n")
                    
                    if '站点ID' in df.columns:
                        stations = df['站点ID'].unique()
                        f.write(f"  {direction_name}覆盖站点数: {len(stations)}\n")
        
        f.write(f"\n📋 甘特图文件说明:\n")
        f.write(f"  - comprehensive_gantt_chart.png: 综合甘特图，显示整体调度概览\n")
        f.write(f"  - detailed_vehicle_gantt.png: 详细车辆甘特图，显示每辆车的运行时间线\n")
        f.write(f"  - load_analysis_chart.png: 载荷分析图表，显示模块使用情况\n")
        f.write(f"  - gantt_statistics_report.txt: 本统计报告\n")
        
        f.write(f"\n🎨 图表说明:\n")
        f.write(f"  - 蓝色圆圈/条形: 乘客模块数量\n")
        f.write(f"  - 红色方块/条形: 货物模块数量\n")
        f.write(f"  - 黑色圆点: 站点位置\n")
        f.write(f"  - 连线: 车辆运行轨迹\n")
        f.write(f"  - 热力图: 载荷分布密度\n")


def demo_with_sample_data():
    """使用示例数据演示甘特图功能"""
    print("🚌 甘特图演示 - 使用示例数据")
    print("=" * 60)
    
    # 创建示例数据
    sample_individual = create_sample_individual()
    sample_schedule = create_sample_schedule_data()
    
    # 创建输出目录
    output_dir = f"gantt_sample_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 创建输出目录: {output_dir}")
    
    try:
        # 创建甘特图绘制器
        plotter = EnhancedGanttPlotter(
            best_individual=sample_individual,
            schedule_data=sample_schedule
        )
        
        print("\n🎨 开始生成示例甘特图...")
        
        # 生成图表
        plotter.generate_comprehensive_gantt_chart(save_dir=output_dir)
        plotter.generate_detailed_vehicle_gantt(save_dir=output_dir)
        plotter.generate_load_analysis_chart(save_dir=output_dir)
        
        # 生成报告
        generate_gantt_statistics_report(plotter, output_dir)
        
        print(f"✅ 示例甘特图演示完成！")
        print(f"📁 结果已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_individual():
    """创建示例个体数据"""
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "fitness": 50000.0,
        "total_cost": 55000.0,
        "remaining_passengers": 10,
        "remaining_freights": 5,
        "up_direction": {
            "vehicle_dispatch": {
                "0": {"headway": 10, "arrival_time": 0},
                "1": {"headway": 8, "arrival_time": 10},
                "2": {"headway": 12, "arrival_time": 18}
            },
            "initial_allocation": {
                "0": {"passenger_modules": 2, "freight_modules": 1},
                "1": {"passenger_modules": 3, "freight_modules": 2},
                "2": {"passenger_modules": 1, "freight_modules": 3}
            }
        },
        "down_direction": {
            "vehicle_dispatch": {
                "100": {"headway": 9, "arrival_time": 0},
                "101": {"headway": 11, "arrival_time": 9},
                "102": {"headway": 7, "arrival_time": 20}
            },
            "initial_allocation": {
                "100": {"passenger_modules": 2, "freight_modules": 2},
                "101": {"passenger_modules": 1, "freight_modules": 1},
                "102": {"passenger_modules": 3, "freight_modules": 1}
            }
        }
    }


def create_sample_schedule_data():
    """创建示例时刻表数据"""
    # 上行时刻表
    up_data = []
    for vid in [0, 1, 2]:
        for sid in range(5):  # 5个站点
            up_data.append({
                "车辆ID": vid,
                "站点ID": sid,
                "到达时间": vid * 10 + sid * 2,
                "乘客模块": np.random.randint(0, 4),
                "货物模块": np.random.randint(0, 3),
                "总模块数量": 0,  # 将在后面计算
                "乘客增量": np.random.randint(-1, 2),
                "货物增量": np.random.randint(-1, 2)
            })
    
    # 下行时刻表
    down_data = []
    for vid in [100, 101, 102]:
        for sid in range(5):  # 5个站点
            down_data.append({
                "车辆ID": vid,
                "站点ID": sid,
                "到达时间": (vid - 100) * 10 + sid * 2,
                "乘客模块": np.random.randint(0, 4),
                "货物模块": np.random.randint(0, 3),
                "总模块数量": 0,  # 将在后面计算
                "乘客增量": np.random.randint(-1, 2),
                "货物增量": np.random.randint(-1, 2)
            })
    
    # 计算总模块数量
    for data in up_data + down_data:
        data["总模块数量"] = data["乘客模块"] + data["货物模块"]
    
    return {
        "up": pd.DataFrame(up_data),
        "down": pd.DataFrame(down_data)
    }


def main():
    """主函数"""
    print("🚌 甘特图演示工具")
    print("=" * 50)
    
    # 检查是否有现有解决方案
    latest_solution = find_latest_solution()
    
    if latest_solution:
        print(f"✅ 发现现有解决方案: {latest_solution}")
        choice = input("选择演示模式:\n1. 使用现有解决方案\n2. 使用示例数据\n请输入选择 (1/2): ").strip()
        
        if choice == "1":
            success = demo_gantt_from_existing_solution()
        elif choice == "2":
            success = demo_with_sample_data()
        else:
            print("❌ 无效选择")
            return
    else:
        print("⚠️ 未发现现有解决方案，使用示例数据演示")
        success = demo_with_sample_data()
    
    if success:
        print("\n🎉 甘特图演示成功完成！")
    else:
        print("\n❌ 甘特图演示失败")


if __name__ == "__main__":
    main()
