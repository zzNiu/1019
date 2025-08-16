# 平滑成本进化曲线绘制器
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
from datetime import datetime

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SmoothCostPlotter:
    """平滑成本进化曲线绘制器"""
    
    def __init__(self, logbook_data):
        """
        初始化绘制器
        
        Args:
            logbook_data: 进化历史数据，可以是DEAP logbook或字典列表
        """
        self.logbook_data = logbook_data
        self.generations, self.min_costs, self.avg_costs, self.max_costs = self._extract_data()
    
    def _extract_data(self):
        """从logbook中提取数据"""
        if not self.logbook_data:
            return [], [], [], []
        
        try:
            generations = [record['gen'] for record in self.logbook_data]
            min_costs = [record['min'] for record in self.logbook_data if np.isfinite(record['min'])]
            avg_costs = [record['avg'] for record in self.logbook_data if np.isfinite(record['avg'])]
            max_costs = [record['max'] for record in self.logbook_data if np.isfinite(record['max'])]
        except (KeyError, TypeError):
            generations = list(range(len(self.logbook_data)))
            min_costs = [record['min'] for record in self.logbook_data if np.isfinite(record['min'])]
            avg_costs = [record['avg'] for record in self.logbook_data if np.isfinite(record['avg'])]
            max_costs = [record['max'] for record in self.logbook_data if np.isfinite(record['max'])]
        
        # 确保数据长度一致
        valid_length = min(len(generations), len(min_costs), len(avg_costs), len(max_costs))
        return (generations[:valid_length], min_costs[:valid_length], 
                avg_costs[:valid_length], max_costs[:valid_length])
    
    def cubic_spline_smooth(self, x_data, y_data, num_points=None):
        """三次样条插值平滑"""
        if len(x_data) < 4:
            return x_data, y_data
        
        if num_points is None:
            num_points = len(x_data) * 5
        
        try:
            x_smooth = np.linspace(x_data[0], x_data[-1], num_points)
            cs = CubicSpline(x_data, y_data, bc_type='natural')
            y_smooth = cs(x_smooth)
            return x_smooth, y_smooth
        except Exception as e:
            print(f"⚠️ 三次样条插值失败: {e}")
            return x_data, y_data
    
    def savgol_smooth(self, y_data, window_length=None, polyorder=3):
        """Savitzky-Golay滤波平滑"""
        if len(y_data) < 5:
            return y_data
        
        if window_length is None:
            window_length = min(7, len(y_data) if len(y_data) % 2 == 1 else len(y_data) - 1)
        
        if window_length < 3:
            window_length = 3
        if window_length >= len(y_data):
            window_length = len(y_data) - 1 if len(y_data) % 2 == 0 else len(y_data) - 2
        
        try:
            return savgol_filter(y_data, window_length, polyorder)
        except Exception as e:
            print(f"⚠️ Savitzky-Golay滤波失败: {e}")
            return y_data
    
    def gaussian_smooth(self, y_data, sigma=None):
        """高斯滤波平滑"""
        if sigma is None:
            sigma = max(1, len(y_data) / 10)
        
        try:
            return gaussian_filter1d(y_data, sigma=sigma)
        except Exception as e:
            print(f"⚠️ 高斯滤波失败: {e}")
            return y_data
    
    def moving_average_smooth(self, y_data, window=None):
        """移动平均平滑"""
        if window is None:
            window = max(3, len(y_data) // 5)
        
        try:
            return pd.Series(y_data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        except Exception as e:
            print(f"⚠️ 移动平均失败: {e}")
            return y_data
    
    def plot_comparison(self, save_path=None, figsize=(16, 12)):
        """绘制多种平滑方法对比图"""
        if len(self.min_costs) < 3:
            print("⚠️ 数据点太少，无法进行有效的平滑处理")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('成本进化曲线 - 平滑方法对比分析', fontsize=16, fontweight='bold')
        
        # 1. 原始数据
        ax1 = axes[0, 0]
        ax1.plot(self.generations, self.min_costs, 'b-', linewidth=2, marker='o', 
                markersize=4, label='最佳成本', alpha=0.8)
        ax1.plot(self.generations, self.avg_costs, 'g-', linewidth=2, marker='s', 
                markersize=3, label='平均成本', alpha=0.8)
        ax1.plot(self.generations, self.max_costs, 'r-', linewidth=2, marker='^', 
                markersize=3, label='最差成本', alpha=0.8)
        ax1.set_title('原始数据', fontweight='bold')
        ax1.set_xlabel('进化代数')
        ax1.set_ylabel('成本')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 三次样条插值
        ax2 = axes[0, 1]
        x_smooth, min_smooth = self.cubic_spline_smooth(self.generations, self.min_costs)
        _, avg_smooth = self.cubic_spline_smooth(self.generations, self.avg_costs)
        _, max_smooth = self.cubic_spline_smooth(self.generations, self.max_costs)
        
        ax2.scatter(self.generations, self.min_costs, c='blue', s=20, alpha=0.5, label='原始点')
        ax2.plot(x_smooth, min_smooth, 'b-', linewidth=3, label='最佳成本(样条)', alpha=0.9)
        ax2.plot(x_smooth, avg_smooth, 'g--', linewidth=2.5, label='平均成本(样条)', alpha=0.8)
        ax2.plot(x_smooth, max_smooth, 'r:', linewidth=2, label='最差成本(样条)', alpha=0.7)
        ax2.set_title('三次样条插值', fontweight='bold')
        ax2.set_xlabel('进化代数')
        ax2.set_ylabel('成本')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Savitzky-Golay滤波
        ax3 = axes[0, 2]
        min_savgol = self.savgol_smooth(self.min_costs)
        avg_savgol = self.savgol_smooth(self.avg_costs)
        max_savgol = self.savgol_smooth(self.max_costs)
        
        ax3.plot(self.generations, self.min_costs, 'b-', linewidth=1, alpha=0.3, label='原始')
        ax3.plot(self.generations, min_savgol, 'b-', linewidth=3, label='最佳成本(S-G)', alpha=0.9)
        ax3.plot(self.generations, avg_savgol, 'g--', linewidth=2.5, label='平均成本(S-G)', alpha=0.8)
        ax3.plot(self.generations, max_savgol, 'r:', linewidth=2, label='最差成本(S-G)', alpha=0.7)
        ax3.set_title('Savitzky-Golay滤波', fontweight='bold')
        ax3.set_xlabel('进化代数')
        ax3.set_ylabel('成本')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 高斯滤波
        ax4 = axes[1, 0]
        min_gauss = self.gaussian_smooth(self.min_costs)
        avg_gauss = self.gaussian_smooth(self.avg_costs)
        max_gauss = self.gaussian_smooth(self.max_costs)
        
        ax4.plot(self.generations, self.min_costs, 'b-', linewidth=1, alpha=0.3, label='原始')
        ax4.plot(self.generations, min_gauss, 'b-', linewidth=3, label='最佳成本(高斯)', alpha=0.9)
        ax4.plot(self.generations, avg_gauss, 'g--', linewidth=2.5, label='平均成本(高斯)', alpha=0.8)
        ax4.plot(self.generations, max_gauss, 'r:', linewidth=2, label='最差成本(高斯)', alpha=0.7)
        ax4.set_title('高斯滤波', fontweight='bold')
        ax4.set_xlabel('进化代数')
        ax4.set_ylabel('成本')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 移动平均
        ax5 = axes[1, 1]
        min_ma = self.moving_average_smooth(self.min_costs)
        avg_ma = self.moving_average_smooth(self.avg_costs)
        max_ma = self.moving_average_smooth(self.max_costs)
        
        ax5.plot(self.generations, self.min_costs, 'b-', linewidth=1, alpha=0.3, label='原始')
        ax5.plot(self.generations, min_ma, 'b-', linewidth=3, label='最佳成本(移动平均)', alpha=0.9)
        ax5.plot(self.generations, avg_ma, 'g--', linewidth=2.5, label='平均成本(移动平均)', alpha=0.8)
        ax5.plot(self.generations, max_ma, 'r:', linewidth=2, label='最差成本(移动平均)', alpha=0.7)
        ax5.set_title('移动平均', fontweight='bold')
        ax5.set_xlabel('进化代数')
        ax5.set_ylabel('成本')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 综合对比（最佳成本）
        ax6 = axes[1, 2]
        ax6.plot(self.generations, self.min_costs, 'k-', linewidth=2, alpha=0.5, 
                label='原始数据', marker='o', markersize=3)
        
        # 绘制所有平滑方法的最佳成本
        x_smooth, min_spline = self.cubic_spline_smooth(self.generations, self.min_costs)
        ax6.plot(x_smooth, min_spline, 'b-', linewidth=2, label='样条插值', alpha=0.8)
        ax6.plot(self.generations, min_savgol, 'g--', linewidth=2, label='S-G滤波', alpha=0.8)
        ax6.plot(self.generations, min_gauss, 'r:', linewidth=2, label='高斯滤波', alpha=0.8)
        ax6.plot(self.generations, min_ma, 'm-.', linewidth=2, label='移动平均', alpha=0.8)
        
        ax6.set_title('最佳成本 - 所有方法对比', fontweight='bold')
        ax6.set_xlabel('进化代数')
        ax6.set_ylabel('成本')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 平滑成本进化曲线对比图已保存到: {save_path}")
        
        plt.show()
        return fig

    """绘制单一最佳平滑方法的图表"""
    def plot_best_smooth(self, method='spline', save_path=None, figsize=(12, 8)):
        """绘制单一最佳平滑方法的图表"""
        if len(self.min_costs) < 3:
            print("⚠️ 数据点太少，无法进行有效的平滑处理")
            return None
        
        plt.figure(figsize=figsize)
        
        # 根据方法选择平滑函数
        if method == 'spline':
            x_smooth, min_smooth = self.cubic_spline_smooth(self.generations, self.min_costs)
            _, avg_smooth = self.cubic_spline_smooth(self.generations, self.avg_costs)
            _, max_smooth = self.cubic_spline_smooth(self.generations, self.max_costs)
            title_suffix = '三次样条插值平滑'
        elif method == 'savgol':
            x_smooth = self.generations
            min_smooth = self.savgol_smooth(self.min_costs)
            avg_smooth = self.savgol_smooth(self.avg_costs)
            max_smooth = self.savgol_smooth(self.max_costs)
            title_suffix = 'Savitzky-Golay滤波平滑'
        elif method == 'gaussian':
            x_smooth = self.generations
            min_smooth = self.gaussian_smooth(self.min_costs)
            avg_smooth = self.gaussian_smooth(self.avg_costs)
            max_smooth = self.gaussian_smooth(self.max_costs)
            title_suffix = '高斯滤波平滑'
        elif method == 'moving_avg':
            x_smooth = self.generations
            min_smooth = self.moving_average_smooth(self.min_costs)
            avg_smooth = self.moving_average_smooth(self.avg_costs)
            max_smooth = self.moving_average_smooth(self.max_costs)
            title_suffix = '移动平均平滑'
        else:
            x_smooth = self.generations
            min_smooth = self.min_costs
            avg_smooth = self.avg_costs
            max_smooth = self.max_costs
            title_suffix = '原始数据'
        
        # 绘制原始数据点（淡色）
        plt.scatter(self.generations, self.min_costs, c='blue', s=20, alpha=0.3, label='最佳成本(原始)')
        plt.scatter(self.generations, self.avg_costs, c='green', s=15, alpha=0.3, label='平均成本(原始)')
        plt.scatter(self.generations, self.max_costs, c='red', s=15, alpha=0.3, label='最差成本(原始)')
        
        # 绘制平滑曲线
        plt.plot(x_smooth, min_smooth, 'b-', linewidth=4, label='最佳成本(平滑)', alpha=0.9)
        plt.plot(x_smooth, avg_smooth, 'g--', linewidth=3, label='平均成本(平滑)', alpha=0.8)
        plt.plot(x_smooth, max_smooth, 'r:', linewidth=2, label='最差成本(平滑)', alpha=0.7)
        
        # 添加填充区域
        if len(x_smooth) == len(min_smooth):
            plt.fill_between(x_smooth, min_smooth, avg_smooth, alpha=0.2, color='blue', label='改进空间')
        
        plt.xlabel('进化代数', fontsize=14, fontweight='bold')
        plt.ylabel('目标函数值（总成本）', fontsize=14, fontweight='bold')
        plt.title(f'遗传算法成本进化曲线 - {title_suffix}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        if len(self.min_costs) > 1:
            total_improvement = self.min_costs[0] - self.min_costs[-1]
            improvement_pct = (total_improvement / self.min_costs[0] * 100) if self.min_costs[0] > 0 else 0
            
            stats_text = f"""统计信息:
总代数: {len(self.generations)}
初始成本: {self.min_costs[0]:.2f}
最终成本: {self.min_costs[-1]:.2f}
总改进: {total_improvement:.2f} ({improvement_pct:.1f}%)"""
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 平滑成本进化曲线已保存到: {save_path}")
        
        plt.show()
        return plt.gcf()


if __name__ == "__main__":
    # 示例用法
    print("平滑成本进化曲线绘制器")
    print("请在其他脚本中导入并使用 SmoothCostPlotter 类或 create_smooth_evolution_plots 函数")
