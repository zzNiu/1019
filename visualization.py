# 增强版可视化模块 - 生成详细的成本进化曲线和调度计划图表
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
from scipy.signal import savgol_filter

# # 设置matplotlib中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
# matplotlib.rcParams['axes.unicode_minus'] = False
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 设置seaborn样式
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


def generate_comprehensive_cost_evolution_plot(logbook, save_dir):
    """
    生成全面的成本进化分析图表（增强版）

    Args:
        logbook: DEAP logbook对象或进化历史数据列表
        save_dir: 保存目录
    """
    if not logbook:
        print("⚠️ 没有进化历史数据，无法生成成本进化曲线")
        return

    # 数据预处理
    if hasattr(logbook, '__iter__') and hasattr(logbook[0], 'get'):
        # 如果是字典列表格式
        generations = [record.get('gen', i) for i, record in enumerate(logbook)]
        min_costs = [record.get('min', float('inf')) for record in logbook if np.isfinite(record.get('min', float('inf')))]
        avg_costs = [record.get('avg', float('inf')) for record in logbook if np.isfinite(record.get('avg', float('inf')))]
        max_costs = [record.get('max', float('inf')) for record in logbook if np.isfinite(record.get('max', float('inf')))]
    else:
        # 如果是DEAP logbook格式
        generations = [record['gen'] for record in logbook]
        min_costs = [record['min'] for record in logbook if np.isfinite(record['min'])]
        avg_costs = [record['avg'] for record in logbook if np.isfinite(record['avg'])]
        max_costs = [record['max'] for record in logbook if np.isfinite(record['max'])]

    # 确保数据长度一致
    valid_length = min(len(generations), len(min_costs), len(avg_costs), len(max_costs))
    generations = generations[:valid_length]
    min_costs = min_costs[:valid_length]
    avg_costs = avg_costs[:valid_length]
    max_costs = max_costs[:valid_length]

    if valid_length == 0:
        print("⚠️ 没有有效的进化数据")
        return

    # 生成综合分析图表
    _generate_comprehensive_analysis_plot(generations, min_costs, avg_costs, max_costs, save_dir)

    print(f"✅ 所有成本进化曲线图表已保存到: {save_dir}")

# 12个子图
def _generate_comprehensive_analysis_plot(generations, min_costs, avg_costs, max_costs, save_dir):
    """生成综合分析图表"""
    fig = plt.figure(figsize=(12, 9))

    # 子图1: 主要成本进化曲线
    plt.subplot(1, 2, 1)
    # ax1 = plt.subplot(1, 2, 1)  # 获取轴对象
    plt.plot(generations, min_costs, 'b-', linewidth=3, label='最佳成本', marker='o', markersize=5)
    plt.plot(generations, avg_costs, 'g--', linewidth=2, label='平均成本', marker='s', markersize=4)
    plt.plot(generations, max_costs, 'r:', linewidth=2, label='最差成本', marker='^', markersize=3)
    plt.xlabel('代数', fontsize=12)
    plt.ylabel('成本', fontsize=12)
    plt.title('成本进化曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加纵轴格式化设置
    import matplotlib.ticker as ticker

    # ax1 = plt.subplot(1, 2, 1)  # 获取轴对象

    # # 方法3: 设置合适的刻度数量
    # ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    #
    # plt.xlabel('代数', fontsize=12)
    # plt.ylabel('成本', fontsize=12)
    # plt.title('成本进化曲线', fontsize=14, fontweight='bold')
    # plt.legend(fontsize=10)
    # plt.grid(True, alpha=0.3)

    # # 子图2: 成本改进趋势
    # plt.subplot(3, 4, 2)
    # if len(min_costs) > 1:
    #     improvements = [0] + [min_costs[i-1] - min_costs[i] for i in range(1, len(min_costs))]
    #     colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
    #     bars = plt.bar(generations, improvements, alpha=0.7, color=colors)
    #
    #     # 添加趋势线
    #     if len(improvements) > 3:
    #         z = np.polyfit(generations, improvements, 2)
    #         p = np.poly1d(z)
    #         plt.plot(generations, p(generations), "k--", alpha=0.8, linewidth=2, label='趋势线')
    #         plt.legend()
    #
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('成本改进量', fontsize=12)
    #     plt.title('每代成本改进量', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #     plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    #
    # # 子图3: 成本分布范围
    # plt.subplot(3, 4, 3)
    # cost_ranges = [max_costs[i] - min_costs[i] for i in range(len(min_costs))]
    # plt.plot(generations, cost_ranges, 'purple', linewidth=3, marker='d', markersize=5)
    # plt.fill_between(generations, cost_ranges, alpha=0.3, color='purple')
    #
    # # 添加移动平均线
    # if len(cost_ranges) > 3:
    #     window_size = min(5, len(cost_ranges) // 2)
    #     moving_avg = pd.Series(cost_ranges).rolling(window=window_size).mean()
    #     plt.plot(generations, moving_avg, 'orange', linewidth=2, linestyle='--', label=f'{window_size}代移动平均')
    #     plt.legend()
    #
    # plt.xlabel('代数', fontsize=12)
    # plt.ylabel('成本范围 (最大-最小)', fontsize=12)
    # plt.title('种群成本分布范围', fontsize=14, fontweight='bold')
    # plt.grid(True, alpha=0.3)
    #
    # # 子图4: 收敛性分析
    # plt.subplot(3, 4, 4)
    # if len(min_costs) > 5:
    #     window_size = min(5, len(min_costs) // 3)
    #     convergence_indicator = []
    #     for i in range(window_size-1, len(min_costs)):
    #         recent_costs = min_costs[i-window_size+1:i+1]
    #         convergence_indicator.append(np.std(recent_costs))
    #
    #     conv_gens = generations[window_size-1:]
    #     plt.plot(conv_gens, convergence_indicator, 'orange', linewidth=3, marker='x', markersize=6)
    #     plt.fill_between(conv_gens, convergence_indicator, alpha=0.3, color='orange')
    #
    #     # 添加收敛阈值线
    #     threshold = np.mean(convergence_indicator) * 0.1
    #     plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'收敛阈值: {threshold:.3f}')
    #     plt.legend()
    #
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel(f'最近{window_size}代成本标准差', fontsize=12)
    #     plt.title('收敛性指标', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图5: 相对改进百分比
    # plt.subplot(3, 4, 5)
    # if len(min_costs) > 1:
    #     relative_improvements = [(min_costs[0] - cost) / min_costs[0] * 100 for cost in min_costs]
    #     plt.plot(generations, relative_improvements, 'darkgreen', linewidth=3, marker='o', markersize=5)
    #     plt.fill_between(generations, relative_improvements, alpha=0.3, color='darkgreen')
    #
    #     # 添加目标线
    #     target_improvement = 10  # 10%改进目标
    #     plt.axhline(y=target_improvement, color='red', linestyle='--', alpha=0.7, label=f'目标改进: {target_improvement}%')
    #
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('相对改进百分比 (%)', fontsize=12)
    #     plt.title('累积改进百分比', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #
    #     # 添加最终改进百分比标注
    #     final_improvement = relative_improvements[-1]
    #     plt.annotate(f'最终改进: {final_improvement:.2f}%',
    #                 xy=(generations[-1], final_improvement),
    #                 xytext=(generations[-1] - len(generations)*0.2, final_improvement),
    #                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
    #                 fontsize=10, ha='right', color='darkgreen', fontweight='bold')
    #
    # # 子图6: 成本分布箱线图
    # plt.subplot(3, 4, 6)
    # if len(generations) >= 3:
    #     # 选择关键代数进行比较
    #     n_gens = len(generations)
    #     if n_gens >= 5:
    #         key_indices = [0, n_gens//4, n_gens//2, 3*n_gens//4, n_gens-1]
    #     else:
    #         key_indices = list(range(n_gens))
    #
    #     box_data = []
    #     box_labels = []
    #
    #     for idx in key_indices:
    #         if idx < len(min_costs):
    #             # 基于min, avg, max生成模拟分布
    #             mean_val = avg_costs[idx]
    #             std_val = (max_costs[idx] - min_costs[idx]) / 4
    #             simulated_costs = np.random.normal(mean_val, std_val, 50)
    #             # 确保生成的数据在合理范围内
    #             simulated_costs = np.clip(simulated_costs, min_costs[idx], max_costs[idx])
    #             box_data.append(simulated_costs)
    #             box_labels.append(f'第{generations[idx]}代')
    #
    #     bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    #
    #     # 设置箱线图颜色
    #     colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    #     for patch, color in zip(bp['boxes'], colors):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.7)
    #
    #     plt.ylabel('成本分布', fontsize=12)
    #     plt.title('关键代数成本分布对比', fontsize=14, fontweight='bold')
    #     plt.xticks(rotation=45)
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图7: 成本变化率
    # plt.subplot(3, 4, 7)
    # if len(min_costs) > 2:
    #     change_rates = []
    #     for i in range(1, len(min_costs)):
    #         if min_costs[i-1] != 0:
    #             rate = (min_costs[i] - min_costs[i-1]) / min_costs[i-1] * 100
    #             change_rates.append(rate)
    #         else:
    #             change_rates.append(0)
    #
    #     change_gens = generations[1:]
    #     colors = ['green' if rate < 0 else 'red' for rate in change_rates]
    #     plt.bar(change_gens, change_rates, alpha=0.7, color=colors)
    #     plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('成本变化率 (%)', fontsize=12)
    #     plt.title('成本变化率', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图8: 累积改进量
    # plt.subplot(3, 4, 8)
    # if len(min_costs) > 1:
    #     cumulative_improvements = [0]
    #     for i in range(1, len(min_costs)):
    #         cumulative_improvements.append(min_costs[0] - min_costs[i])
    #
    #     plt.plot(generations, cumulative_improvements, 'purple', linewidth=3, marker='o', markersize=5)
    #     plt.fill_between(generations, cumulative_improvements, alpha=0.3, color='purple')
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('累积改进量', fontsize=12)
    #     plt.title('累积改进量', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图9: 成本稳定性分析
    # plt.subplot(3, 4, 9)
    # if len(min_costs) > 3:
    #     stability_window = min(3, len(min_costs) // 4)
    #     stability_scores = []
    #     for i in range(stability_window, len(min_costs)):
    #         recent_costs = min_costs[i-stability_window:i]
    #         stability = 1 / (1 + np.std(recent_costs))  # 稳定性得分
    #         stability_scores.append(stability)
    #
    #     stab_gens = generations[stability_window:]
    #     plt.plot(stab_gens, stability_scores, 'brown', linewidth=3, marker='s', markersize=5)
    #     plt.fill_between(stab_gens, stability_scores, alpha=0.3, color='brown')
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('稳定性得分', fontsize=12)
    #     plt.title('成本稳定性分析', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图10: 效率指标
    # plt.subplot(3, 4, 10)
    # if len(min_costs) > 1:
    #     efficiency_scores = []
    #     for i in range(len(min_costs)):
    #         if i == 0:
    #             efficiency_scores.append(0)
    #         else:
    #             improvement = min_costs[0] - min_costs[i]
    #             efficiency = improvement / (i + 1)  # 每代平均改进
    #             efficiency_scores.append(efficiency)
    #
    #     plt.plot(generations, efficiency_scores, 'teal', linewidth=3, marker='d', markersize=5)
    #     plt.fill_between(generations, efficiency_scores, alpha=0.3, color='teal')
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('效率得分', fontsize=12)
    #     plt.title('进化效率分析', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # # 子图11: 成本梯度分析
    # plt.subplot(3, 4, 11)
    # if len(min_costs) > 2:
    #     gradients = np.gradient(min_costs)
    #     plt.plot(generations, gradients, 'navy', linewidth=3, marker='v', markersize=5)
    #     plt.fill_between(generations, gradients, alpha=0.3, color='navy')
    #     plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    #     plt.xlabel('代数', fontsize=12)
    #     plt.ylabel('成本梯度', fontsize=12)
    #     plt.title('成本梯度分析', fontsize=14, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #
    # 子图12: 性能总结
    plt.subplot(1, 2, 2)
    plt.axis('off')

    # 计算关键指标
    total_improvement = min_costs[0] - min_costs[-1] if len(min_costs) > 1 else 0
    improvement_pct = (total_improvement / min_costs[0] * 100) if min_costs[0] > 0 else 0
    avg_improvement_per_gen = total_improvement / len(generations) if len(generations) > 0 else 0
    best_single_improvement = max([min_costs[i-1] - min_costs[i] for i in range(1, len(min_costs))]) if len(min_costs) > 1 else 0

    summary_text = f"""性能总结:
━━━━━━━━━━━━━━━━━━━━
📊 基本指标:
  • 总代数: {len(generations)}
  • 初始成本: {min_costs[0]:.2f}
  • 最终成本: {min_costs[-1]:.2f}

🎯 改进指标:
  • 总改进量: {total_improvement:.2f}
  • 改进百分比: {improvement_pct:.2f}%
  • 平均每代改进: {avg_improvement_per_gen:.3f}
  • 最大单代改进: {best_single_improvement:.3f}

📈 收敛指标:
  • 最终5代标准差: {np.std(min_costs[-5:]):.3f}
  • 收敛状态: {'已收敛' if np.std(min_costs[-5:]) < total_improvement * 0.01 else '仍在优化'}"""

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             # fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_cost_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()