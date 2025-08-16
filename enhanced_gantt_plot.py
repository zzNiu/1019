#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版甘特图绘制模块 - 显示最优调度计划，包括在车乘客和货物数量
支持从最优解决方案数据生成详细的时空图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
plt.style.use('default')

# 设置全局绘图参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class EnhancedGanttPlotter:
    """增强版甘特图绘制器"""

    def __init__(self, solution_dir=None, best_individual=None, schedule_data=None):
        """
        初始化甘特图绘制器

        Args:
            solution_dir: 解决方案目录路径
            best_individual: 最佳个体数据
            schedule_data: 时刻表数据
        """
        self.solution_dir = solution_dir
        self.best_individual = best_individual
        self.schedule_data = schedule_data

        if solution_dir and not best_individual:
            self._load_solution_data()

    def _load_solution_data(self):
        """从解决方案目录加载数据"""
        try:
            # 加载最佳个体数据
            individual_file = os.path.join(self.solution_dir, 'best_individual.json')
            with open(individual_file, 'r', encoding='utf-8') as f:
                self.best_individual = json.load(f)

            # 加载时刻表数据
            self.schedule_data = {}
            for direction in ['up', 'down']:
                schedule_file = os.path.join(self.solution_dir, f'schedule_{direction}.xlsx')
                if os.path.exists(schedule_file):
                    self.schedule_data[direction] = pd.read_excel(schedule_file)
                else:
                    print(f"⚠️ 未找到{direction}行时刻表文件")
                    self.schedule_data[direction] = pd.DataFrame()

            print(f"✅ 成功加载解决方案数据: {self.solution_dir}")

        except Exception as e:
            print(f"❌ 加载解决方案数据失败: {e}")
            self.best_individual = None
            self.schedule_data = {}

    def generate_comprehensive_gantt_chart(self, save_dir=None, figsize=(20, 16)):
        """生成综合甘特图"""
        if not self.best_individual or not self.schedule_data:
            print("❌ 缺少必要数据，无法生成甘特图")
            return

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, height_ratios=[2, 2, 1], width_ratios=[1, 1])

        # 上行甘特图
        ax_up = plt.subplot(gs[0, 0])
        self._plot_direction_gantt(ax_up, 'up', '上行方向调度甘特图')

        # 下行甘特图
        ax_down = plt.subplot(gs[0, 1])
        self._plot_direction_gantt(ax_down, 'down', '下行方向调度甘特图')

        # 综合时空图
        ax_spacetime = plt.subplot(gs[1, :])
        self._plot_spacetime_diagram(ax_spacetime)

        # 统计信息
        ax_stats = plt.subplot(gs[2, :])
        self._plot_statistics_summary(ax_stats)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'comprehensive_gantt_chart.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 综合甘特图已保存到: {save_path}")

        plt.show()
        return fig

    def _plot_direction_gantt(self, ax, direction, title):
        """绘制单方向甘特图"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, f'{direction}行数据不可用', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return

        df = self.schedule_data[direction].copy()

        # 数据预处理
        df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
        df = df.dropna(subset=['到达时间'])

        if df.empty:
            ax.text(0.5, 0.5, f'{direction}行时间数据无效', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return

        # 获取车辆和站点信息
        vehicles = sorted(df['车辆ID'].unique())
        stations = sorted(df['站点ID'].unique())

        # 设置颜色映射
        vehicle_colors = plt.cm.Set3(np.linspace(0, 1, len(vehicles)))

        # 绘制每辆车的运行轨迹
        for i, vid in enumerate(vehicles):
            vehicle_data = df[df['车辆ID'] == vid].sort_values('到达时间')

            if len(vehicle_data) < 2:
                continue

            color = vehicle_colors[i]

            # 绘制车辆轨迹线
            times = vehicle_data['到达时间'].values
            station_indices = [stations.index(sid) for sid in vehicle_data['站点ID']]

            ax.plot(times, station_indices, color=color, linewidth=3, alpha=0.8,
                    label=f'车辆{vid}', marker='o', markersize=6)

            # 在每个站点添加模块信息
            for _, row in vehicle_data.iterrows():
                station_idx = stations.index(row['站点ID'])
                time = row['到达时间']

                # 乘客模块数量（蓝色圆圈）
                passenger_modules = row['乘客模块']
                if passenger_modules > 0:
                    circle_p = plt.Circle((time, station_idx), radius=passenger_modules * 0.3,
                                          color='blue', alpha=0.6)
                    ax.add_patch(circle_p)
                    ax.text(time, station_idx + 0.15, str(passenger_modules),
                            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

                # 货物模块数量（红色方块）
                freight_modules = row['货物模块']
                if freight_modules > 0:
                    square_f = plt.Rectangle((time - freight_modules * 0.15, station_idx - freight_modules * 0.15),
                                             freight_modules * 0.3, freight_modules * 0.3,
                                             color='red', alpha=0.6)
                    ax.add_patch(square_f)
                    ax.text(time, station_idx - 0.15, str(freight_modules),
                            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # 设置坐标轴
        ax.set_xlabel('时间 (分钟)', fontsize=12)
        ax.set_ylabel('站点ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels([f'站点{sid}' for sid in stations])
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 添加图例说明
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                       markersize=8, alpha=0.6, label='乘客模块'),
            plt.Line2D([0], [0], marker='s', color='red', linestyle='None',
                       markersize=8, alpha=0.6, label='货物模块')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def _plot_spacetime_diagram(self, ax):
        """绘制时空图"""
        ax.set_title('综合时空运行图', fontsize=16, fontweight='bold')

        all_stations = set()
        all_times = []

        # 收集所有数据
        for direction in ['up', 'down']:
            if direction in self.schedule_data and not self.schedule_data[direction].empty:
                df = self.schedule_data[direction]
                all_stations.update(df['站点ID'].unique())
                all_times.extend(pd.to_numeric(df['到达时间'], errors='coerce').dropna())

        if not all_stations or not all_times:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return

        stations = sorted(all_stations)
        time_range = [min(all_times), max(all_times)]

        # 绘制上行和下行轨迹
        colors = {'up': 'blue', 'down': 'red'}
        labels = {'up': '上行', 'down': '下行'}

        for direction in ['up', 'down']:
            if direction not in self.schedule_data or self.schedule_data[direction].empty:
                continue

            df = self.schedule_data[direction]
            df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
            df = df.dropna(subset=['到达时间'])

            vehicles = sorted(df['车辆ID'].unique())

            for vid in vehicles:
                vehicle_data = df[df['车辆ID'] == vid].sort_values('到达时间')

                if len(vehicle_data) < 2:
                    continue

                times = vehicle_data['到达时间'].values
                station_indices = [stations.index(sid) for sid in vehicle_data['站点ID']]

                # 绘制轨迹线
                ax.plot(times, station_indices, color=colors[direction],
                        linewidth=2, alpha=0.7, label=labels[direction] if vid == vehicles[0] else "")

                # 添加载客信息
                for _, row in vehicle_data.iterrows():
                    station_idx = stations.index(row['站点ID'])
                    time = row['到达时间']
                    total_load = row['乘客模块'] + row['货物模块']

                    if total_load > 0:
                        # 用圆圈大小表示总载荷
                        circle = plt.Circle((time, station_idx), radius=total_load * 0.2,
                                            color=colors[direction], alpha=0.4)
                        ax.add_patch(circle)

        ax.set_xlabel('时间 (分钟)', fontsize=12)
        ax.set_ylabel('站点ID', fontsize=12)
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels([f'站点{sid}' for sid in stations])
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_statistics_summary(self, ax):
        """绘制统计摘要"""
        ax.axis('off')

        # 计算统计信息
        stats_text = self._calculate_statistics()

        # 分为两列显示
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 添加性能指标图表
        self._add_performance_charts(ax)

    def _calculate_statistics(self):
        """计算统计信息"""
        stats = []
        stats.append("📊 调度方案统计摘要")
        stats.append("=" * 50)

        # 基本信息
        if self.best_individual:
            # 安全格式化数值
            total_cost = self.best_individual.get('total_cost', 'N/A')
            if isinstance(total_cost, (int, float)):
                stats.append(f"🎯 总成本: {total_cost:.2f}")
            else:
                stats.append(f"🎯 总成本: {total_cost}")

            stats.append(f"🚌 剩余乘客: {self.best_individual.get('remaining_passengers', 'N/A')}")
            stats.append(f"📦 剩余货物: {self.best_individual.get('remaining_freights', 'N/A')}")

        # 车辆统计
        for direction in ['up', 'down']:
            direction_name = "上行" if direction == "up" else "下行"

            if self.best_individual and f'{direction}_direction' in self.best_individual:
                dir_data = self.best_individual[f'{direction}_direction']
                vehicle_count = len(dir_data.get('vehicle_dispatch', {}))

                # 计算模块统计
                initial_alloc = dir_data.get('initial_allocation', {})
                total_p_modules = sum(alloc.get('passenger_modules', 0) for alloc in initial_alloc.values())
                total_f_modules = sum(alloc.get('freight_modules', 0) for alloc in initial_alloc.values())

                stats.append(f"\n🚍 {direction_name}方向:")
                stats.append(f"  车辆数量: {vehicle_count}")
                stats.append(f"  乘客模块: {total_p_modules}")
                stats.append(f"  货物模块: {total_f_modules}")
                stats.append(f"  总模块: {total_p_modules + total_f_modules}")

                # 车头时距统计
                if 'vehicle_dispatch' in dir_data:
                    headways = [v.get('headway', 0) for v in dir_data['vehicle_dispatch'].values()]
                    if headways:
                        stats.append(f"  平均车头时距: {np.mean(headways):.1f}分钟")
                        stats.append(f"  车头时距范围: {min(headways)}-{max(headways)}分钟")

        return "\n".join(stats)

    def _add_performance_charts(self, ax):
        """添加性能图表"""
        # 在右侧添加小型图表
        if not self.best_individual:
            return

        # 创建子图区域
        from matplotlib.patches import Rectangle

        # 模块分布饼图区域
        pie_rect = Rectangle((0.6, 0.6), 0.35, 0.35, transform=ax.transAxes,
                             facecolor='white', edgecolor='black', alpha=0.8)
        ax.add_patch(pie_rect)

        # 计算模块分布数据
        total_p_modules = 0
        total_f_modules = 0

        for direction in ['up', 'down']:
            if f'{direction}_direction' in self.best_individual:
                dir_data = self.best_individual[f'{direction}_direction']
                initial_alloc = dir_data.get('initial_allocation', {})
                total_p_modules += sum(alloc.get('passenger_modules', 0) for alloc in initial_alloc.values())
                total_f_modules += sum(alloc.get('freight_modules', 0) for alloc in initial_alloc.values())

        # 简化的饼图数据显示
        ax.text(0.77, 0.85, '模块分布', transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold')
        ax.text(0.77, 0.75, f'乘客: {total_p_modules}', transform=ax.transAxes,
                ha='center', fontsize=9, color='blue')
        ax.text(0.77, 0.70, f'货物: {total_f_modules}', transform=ax.transAxes,
                ha='center', fontsize=9, color='red')
        ax.text(0.77, 0.65, f'总计: {total_p_modules + total_f_modules}', transform=ax.transAxes,
                ha='center', fontsize=9, fontweight='bold')

    def generate_detailed_vehicle_gantt(self, save_dir=None, figsize=(24, 16)):
        """生成详细的车辆甘特图，显示每辆车的详细运行状态"""
        if not self.best_individual or not self.schedule_data:
            print("❌ 缺少必要数据，无法生成详细甘特图")
            return

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        for i, direction in enumerate(['up', 'down']):
            ax = axes[i]
            self._plot_detailed_vehicle_timeline(ax, direction)

        plt.tight_layout()

        if save_dir:

            save_path = os.path.join(save_dir, 'detailed_vehicle_gantt.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 详细车辆甘特图已保存到: {save_path}")

        plt.show()
        return fig

    def generate_load_analysis_chart(self, save_dir=None, figsize=(18, 12)):
        """生成载荷分析图表"""
        if not self.schedule_data:
            print("❌ 缺少时刻表数据，无法生成载荷分析图")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 第一行：上行分析
        self._plot_load_distribution(axes[0, 0], 'up', '上行载荷分布')
        self._plot_load_timeline(axes[0, 1], 'up', '上行载荷时间线')
        self._plot_station_load_heatmap(axes[0, 2], 'up', '上行站点载荷热力图')

        # 第二行：下行分析
        self._plot_load_distribution(axes[1, 0], 'down', '下行载荷分布')
        self._plot_load_timeline(axes[1, 1], 'down', '下行载荷时间线')
        self._plot_station_load_heatmap(axes[1, 2], 'down', '下行站点载荷热力图')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'load_analysis_chart.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 载荷分析图表已保存到: {save_path}")

        plt.show()
        return fig

    def _plot_detailed_vehicle_timeline(self, ax, direction):
        """绘制详细的车辆时间线"""
        direction_name = "上行" if direction == "up" else "下行"

        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, f'{direction_name}数据不可用', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{direction_name}方向详细时间线')
            return

        df = self.schedule_data[direction].copy()
        df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
        df = df.dropna(subset=['到达时间'])

        if df.empty:
            ax.text(0.5, 0.5, f'{direction_name}时间数据无效', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{direction_name}方向详细时间线')
            return

        vehicles = sorted(df['车辆ID'].unique())

        # 为每辆车分配一行
        for i, vid in enumerate(vehicles):
            vehicle_data = df[df['车辆ID'] == vid].sort_values('到达时间')

            y_pos = i

            # 绘制车辆运行时间段
            if len(vehicle_data) >= 2:
                start_time = vehicle_data['到达时间'].min()
                end_time = vehicle_data['到达时间'].max()

                # 背景时间条
                rect = patches.Rectangle((start_time, y_pos - 0.4), end_time - start_time, 0.8,
                                         linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.3)
                ax.add_patch(rect)

            # 在每个站点绘制详细信息
            for _, row in vehicle_data.iterrows():
                time = row['到达时间']
                passenger_modules = row['乘客模块']
                freight_modules = row['货物模块']
                station_id = row['站点ID']

                # 乘客模块（蓝色条）
                if passenger_modules > 0:
                    p_rect = patches.Rectangle((time - 0.5, y_pos - 0.3), 1, 0.2,
                                               facecolor='blue', alpha=0.8, edgecolor='darkblue')
                    ax.add_patch(p_rect)
                    ax.text(time, y_pos - 0.2, str(passenger_modules), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')

                # 货物模块（红色条）
                if freight_modules > 0:
                    f_rect = patches.Rectangle((time - 0.5, y_pos + 0.1), 1, 0.2,
                                               facecolor='red', alpha=0.8, edgecolor='darkred')
                    ax.add_patch(f_rect)
                    ax.text(time, y_pos + 0.2, str(freight_modules), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')

                # 站点标记
                ax.plot(time, y_pos, 'ko', markersize=6)
                ax.text(time, y_pos - 0.5, f'S{station_id}', ha='center', va='top',
                        fontsize=7, rotation=45)

        # 设置坐标轴
        ax.set_xlabel('时间 (分钟)', fontsize=12)
        ax.set_ylabel('车辆ID', fontsize=12)
        ax.set_title(f'{direction_name}方向详细车辆时间线', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(vehicles)))
        ax.set_yticklabels([f'车辆{vid}' for vid in vehicles])
        ax.grid(True, alpha=0.3, axis='x')

        # 添加图例
        legend_elements = [
            patches.Patch(color='blue', alpha=0.8, label='乘客模块'),
            patches.Patch(color='red', alpha=0.8, label='货物模块'),
            plt.Line2D([0], [0], marker='o', color='black', linestyle='None',
                       markersize=6, label='站点')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def _plot_load_distribution(self, ax, direction, title):
        """绘制载荷分布图"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction]

        # 计算载荷数据
        passenger_loads = df['乘客模块'].values
        freight_loads = df['货物模块'].values
        total_loads = passenger_loads + freight_loads

        # 绘制直方图
        ax.hist([passenger_loads, freight_loads, total_loads],
                bins=10, alpha=0.7, label=['乘客模块', '货物模块', '总载荷'],
                color=['blue', 'red', 'green'])

        ax.set_xlabel('模块数量')
        ax.set_ylabel('频次')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_load_timeline(self, ax, direction, title):
        """绘制载荷时间线"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction].copy()
        df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
        df = df.dropna(subset=['到达时间']).sort_values('到达时间')

        # 按时间聚合载荷
        time_groups = df.groupby('到达时间').agg({
            '乘客模块': 'sum',
            '货物模块': 'sum'
        }).reset_index()

        ax.plot(time_groups['到达时间'], time_groups['乘客模块'],
                'b-', linewidth=2, marker='o', label='乘客模块')
        ax.plot(time_groups['到达时间'], time_groups['货物模块'],
                'r-', linewidth=2, marker='s', label='货物模块')
        ax.plot(time_groups['到达时间'],
                time_groups['乘客模块'] + time_groups['货物模块'],
                'g--', linewidth=2, marker='^', label='总载荷')

        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('模块数量')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_station_load_heatmap(self, ax, direction, title):
        """绘制站点载荷热力图"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction]

        # 创建站点-车辆载荷矩阵
        stations = sorted(df['站点ID'].unique())
        vehicles = sorted(df['车辆ID'].unique())

        # 乘客模块热力图数据
        passenger_matrix = np.zeros((len(vehicles), len(stations)))
        freight_matrix = np.zeros((len(vehicles), len(stations)))

        for i, vid in enumerate(vehicles):
            for j, sid in enumerate(stations):
                vehicle_station_data = df[(df['车辆ID'] == vid) & (df['站点ID'] == sid)]
                if not vehicle_station_data.empty:
                    passenger_matrix[i, j] = vehicle_station_data['乘客模块'].iloc[0]
                    freight_matrix[i, j] = vehicle_station_data['货物模块'].iloc[0]

        # 绘制总载荷热力图
        total_matrix = passenger_matrix + freight_matrix
        im = ax.imshow(total_matrix, cmap='YlOrRd', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(range(len(stations)))
        ax.set_xticklabels([f'S{sid}' for sid in stations])
        ax.set_yticks(range(len(vehicles)))
        ax.set_yticklabels([f'V{vid}' for vid in vehicles])
        ax.set_xlabel('站点ID')
        ax.set_ylabel('车辆ID')
        ax.set_title(title)

        # 添加数值标注
        for i in range(len(vehicles)):
            for j in range(len(stations)):
                if total_matrix[i, j] > 0:
                    ax.text(j, i, f'{int(total_matrix[i, j])}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='总模块数')


def generate_gantt_from_solution_dir(solution_dir, save_dir=None):
    """
    从解决方案目录生成甘特图

    Args:
        solution_dir: 解决方案目录路径
        save_dir: 保存目录（如果为None，则保存到解决方案目录）
    """
    if save_dir is None:
        save_dir = solution_dir

    plotter = EnhancedGanttPlotter(solution_dir=solution_dir)

    if plotter.best_individual is None:
        print("❌ 无法加载解决方案数据")
        return None

    # 生成综合甘特图
    fig = plotter.generate_comprehensive_gantt_chart(save_dir=save_dir)

    return fig


def main():
    """主函数 - 演示甘特图绘制功能"""
    print("🚌 增强版甘特图绘制工具")
    print("=" * 50)

    # 查找最新的解决方案目录
    solution_dirs = [d for d in os.listdir('.') if d.startswith('best_solution_')]
    if not solution_dirs:
        print("❌ 未找到解决方案目录，请先运行优化程序")
        return

    # 使用最新的解决方案
    latest_dir = max(solution_dirs)
    print(f"📁 使用解决方案目录: {latest_dir}")

    try:
        # 生成甘特图
        fig = generate_gantt_from_solution_dir(latest_dir)

        if fig:
            print(f"✅ 甘特图生成完成！")
            print(f"📊 结果已保存到: {latest_dir}")
        else:
            print("❌ 甘特图生成失败")

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


class EnhancedGanttPlotterExtension:
    """甘特图绘制器扩展方法"""

    def generate_detailed_vehicle_gantt(self, save_dir=None, figsize=(24, 16)):
        """生成详细的车辆甘特图，显示每辆车的详细运行状态"""
        if not self.best_individual or not self.schedule_data:
            print("❌ 缺少必要数据，无法生成详细甘特图")
            return

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        for i, direction in enumerate(['up', 'down']):
            ax = axes[i]
            self._plot_detailed_vehicle_timeline(ax, direction)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'detailed_vehicle_gantt.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 详细车辆甘特图已保存到: {save_path}")

        plt.show()
        return fig

    def _plot_detailed_vehicle_timeline(self, ax, direction):
        """绘制详细的车辆时间线"""
        direction_name = "上行" if direction == "up" else "下行"

        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, f'{direction_name}数据不可用', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{direction_name}方向详细时间线')
            return

        df = self.schedule_data[direction].copy()
        df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
        df = df.dropna(subset=['到达时间'])

        if df.empty:
            ax.text(0.5, 0.5, f'{direction_name}时间数据无效', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{direction_name}方向详细时间线')
            return

        vehicles = sorted(df['车辆ID'].unique())

        # 为每辆车分配一行
        for i, vid in enumerate(vehicles):
            vehicle_data = df[df['车辆ID'] == vid].sort_values('到达时间')

            y_pos = i

            # 绘制车辆运行时间段
            if len(vehicle_data) >= 2:
                start_time = vehicle_data['到达时间'].min()
                end_time = vehicle_data['到达时间'].max()

                # 背景时间条
                rect = patches.Rectangle((start_time, y_pos - 0.4), end_time - start_time, 0.8,
                                         linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.3)
                ax.add_patch(rect)

            # 在每个站点绘制详细信息
            for _, row in vehicle_data.iterrows():
                time = row['到达时间']
                passenger_modules = row['乘客模块']
                freight_modules = row['货物模块']
                station_id = row['站点ID']

                # 乘客模块（蓝色条）
                if passenger_modules > 0:
                    p_rect = patches.Rectangle((time - 0.5, y_pos - 0.3), 1, 0.2,
                                               facecolor='blue', alpha=0.8, edgecolor='darkblue')
                    ax.add_patch(p_rect)
                    ax.text(time, y_pos - 0.2, str(passenger_modules), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')

                # 货物模块（红色条）
                if freight_modules > 0:
                    f_rect = patches.Rectangle((time - 0.5, y_pos + 0.1), 1, 0.2,
                                               facecolor='red', alpha=0.8, edgecolor='darkred')
                    ax.add_patch(f_rect)
                    ax.text(time, y_pos + 0.2, str(freight_modules), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')

                # 站点标记
                ax.plot(time, y_pos, 'ko', markersize=6)
                ax.text(time, y_pos - 0.5, f'S{station_id}', ha='center', va='top',
                        fontsize=7, rotation=45)

        # 设置坐标轴
        ax.set_xlabel('时间 (分钟)', fontsize=12)
        ax.set_ylabel('车辆ID', fontsize=12)
        ax.set_title(f'{direction_name}方向详细车辆时间线', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(vehicles)))
        ax.set_yticklabels([f'车辆{vid}' for vid in vehicles])
        ax.grid(True, alpha=0.3, axis='x')

        # 添加图例
        legend_elements = [
            patches.Patch(color='blue', alpha=0.8, label='乘客模块'),
            patches.Patch(color='red', alpha=0.8, label='货物模块'),
            plt.Line2D([0], [0], marker='o', color='black', linestyle='None',
                       markersize=6, label='站点')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def generate_load_analysis_chart(self, save_dir=None, figsize=(18, 12)):
        """生成载荷分析图表"""
        if not self.schedule_data:
            print("❌ 缺少时刻表数据，无法生成载荷分析图")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 第一行：上行分析
        self._plot_load_distribution(axes[0, 0], 'up', '上行载荷分布')
        self._plot_load_timeline(axes[0, 1], 'up', '上行载荷时间线')
        self._plot_station_load_heatmap(axes[0, 2], 'up', '上行站点载荷热力图')

        # 第二行：下行分析
        self._plot_load_distribution(axes[1, 0], 'down', '下行载荷分布')
        self._plot_load_timeline(axes[1, 1], 'down', '下行载荷时间线')
        self._plot_station_load_heatmap(axes[1, 2], 'down', '下行站点载荷热力图')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'load_analysis_chart.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 载荷分析图表已保存到: {save_path}")

        plt.show()
        return fig

    def _plot_load_distribution(self, ax, direction, title):
        """绘制载荷分布图"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction]

        # 计算载荷数据
        passenger_loads = df['乘客模块'].values
        freight_loads = df['货物模块'].values
        total_loads = passenger_loads + freight_loads

        # 绘制直方图
        ax.hist([passenger_loads, freight_loads, total_loads],
                bins=10, alpha=0.7, label=['乘客模块', '货物模块', '总载荷'],
                color=['blue', 'red', 'green'])

        ax.set_xlabel('模块数量')
        ax.set_ylabel('频次')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_load_timeline(self, ax, direction, title):
        """绘制载荷时间线"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction].copy()
        df['到达时间'] = pd.to_numeric(df['到达时间'], errors='coerce')
        df = df.dropna(subset=['到达时间']).sort_values('到达时间')

        # 按时间聚合载荷
        time_groups = df.groupby('到达时间').agg({
            '乘客模块': 'sum',
            '货物模块': 'sum'
        }).reset_index()

        ax.plot(time_groups['到达时间'], time_groups['乘客模块'],
                'b-', linewidth=2, marker='o', label='乘客模块')
        ax.plot(time_groups['到达时间'], time_groups['货物模块'],
                'r-', linewidth=2, marker='s', label='货物模块')
        ax.plot(time_groups['到达时间'],
                time_groups['乘客模块'] + time_groups['货物模块'],
                'g--', linewidth=2, marker='^', label='总载荷')

        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('模块数量')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_station_load_heatmap(self, ax, direction, title):
        """绘制站点载荷热力图"""
        if direction not in self.schedule_data or self.schedule_data[direction].empty:
            ax.text(0.5, 0.5, '数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        df = self.schedule_data[direction]

        # 创建站点-车辆载荷矩阵
        stations = sorted(df['站点ID'].unique())
        vehicles = sorted(df['车辆ID'].unique())

        # 乘客模块热力图数据
        passenger_matrix = np.zeros((len(vehicles), len(stations)))
        freight_matrix = np.zeros((len(vehicles), len(stations)))

        for i, vid in enumerate(vehicles):
            for j, sid in enumerate(stations):
                vehicle_station_data = df[(df['车辆ID'] == vid) & (df['站点ID'] == sid)]
                if not vehicle_station_data.empty:
                    passenger_matrix[i, j] = vehicle_station_data['乘客模块'].iloc[0]
                    freight_matrix[i, j] = vehicle_station_data['货物模块'].iloc[0]

        # 绘制总载荷热力图
        total_matrix = passenger_matrix + freight_matrix
        im = ax.imshow(total_matrix, cmap='YlOrRd', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(range(len(stations)))
        ax.set_xticklabels([f'S{sid}' for sid in stations])
        ax.set_yticks(range(len(vehicles)))
        ax.set_yticklabels([f'V{vid}' for vid in vehicles])
        ax.set_xlabel('站点ID')
        ax.set_ylabel('车辆ID')
        ax.set_title(title)

        # 添加数值标注
        for i in range(len(vehicles)):
            for j in range(len(stations)):
                if total_matrix[i, j] > 0:
                    ax.text(j, i, f'{int(total_matrix[i, j])}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='总模块数')


if __name__ == "__main__":
    main()
