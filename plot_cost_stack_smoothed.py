import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# 设置matplotlib中文字体，确保图表能正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# --- 平滑函数 (从 smooth_cost_plotter.py 借鉴) ---

def cubic_spline_smooth(x_data, y_data, num_points=None):
    """三次样条插值平滑"""
    if len(x_data) < 4:
        return x_data, y_data
    if num_points is None:
        num_points = len(x_data) * 2  # 减少点数以避免过于震荡
    try:
        x_smooth = np.linspace(x_data.min(), x_data.max(), num_points)
        cs = CubicSpline(x_data, y_data)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth
    except Exception:
        return x_data, y_data


def savgol_smooth(y_data, window_length=None, polyorder=3):
    """Savitzky-Golay滤波平滑"""
    if len(y_data) < 5:
        return y_data
    if window_length is None:
        window_length = min(11, len(y_data) - 1 if len(y_data) % 2 == 0 else len(y_data))
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(y_data):
        return y_data
    try:
        return savgol_filter(y_data, window_length, polyorder)
    except Exception:
        return y_data


def gaussian_smooth(y_data, sigma=None):
    """高斯滤波平滑"""
    if sigma is None:
        sigma = max(1, len(y_data) / 20)
    try:
        return gaussian_filter1d(y_data, sigma=sigma)
    except Exception:
        return y_data


def moving_average_smooth(y_data, window=None):
    """移动平均平滑"""
    if window is None:
        window = max(3, len(y_data) // 10)
    try:
        return pd.Series(y_data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(
            method='ffill').values
    except Exception:
        return y_data


# --- 绘图函数 ---

def plot_cost_stack_from_history(ax, gens, P, F, M, title="成本构成堆叠图", smoothed=False):
    """
    在指定的 Axes 上绘制成本构成堆叠图 (原始或平滑后)。
    ax: matplotlib.axes.Axes 对象
    gens, P, F, M: 绘图数据
    title: 图表标题
    smoothed: 是否为平滑曲线图，用于调整线条样式
    """
    labels = ["乘客等待成本", "货物等待成本", "MAV运输成本"]

    # 绘制堆叠面积图
    ax.stackplot(gens, P, F, M,
                 labels=labels,
                 linewidth=0.5, edgecolor="white", alpha=0.8)

    # 绘制总成本曲线
    line_style = '-' if smoothed else '--'
    ax.plot(gens, P + F + M, color='black', linewidth=1.5, linestyle=line_style, label="总成本")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("代数", fontsize=10)
    ax.set_ylabel("成本值", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 设置Y轴为科学计数法
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='both', which='major', labelsize=9)


def plot_smoothing_comparison(cost_history, save_path="cost_stack_smoothing_comparison.png"):
    """
    对比分析不同平滑方法在成本构成堆叠图上的效果。
    """
    P_orig = np.asarray(cost_history["passenger"], dtype=float)
    F_orig = np.asarray(cost_history["freight"], dtype=float)
    M_orig = np.asarray(cost_history["mav"], dtype=float)
    assert len(P_orig) == len(F_orig) == len(M_orig), "三条成本序列长度需一致"

    gens_orig = np.arange(len(P_orig))

    # 创建一个2x2的图表用于对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle("成本构成堆叠图 - 平滑方法对比分析", fontsize=16, fontweight='bold')

    # 1. 原始数据图
    plot_cost_stack_from_history(axes[0, 0], gens_orig, P_orig, F_orig, M_orig, title="原始数据")

    # 2. 三次样条插值平滑
    gens_spline, P_spline = cubic_spline_smooth(gens_orig, P_orig)
    _, F_spline = cubic_spline_smooth(gens_orig, F_orig)
    _, M_spline = cubic_spline_smooth(gens_orig, M_orig)
    plot_cost_stack_from_history(axes[0, 1], gens_spline, P_spline, F_spline, M_spline, title="三次样条插值",
                                 smoothed=True)

    # 3. Savitzky-Golay 滤波
    P_savgol = savgol_smooth(P_orig)
    F_savgol = savgol_smooth(F_orig)
    M_savgol = savgol_smooth(M_orig)
    plot_cost_stack_from_history(axes[1, 0], gens_orig, P_savgol, F_savgol, M_savgol, title="Savitzky-Golay 滤波",
                                 smoothed=True)

    # 4. 移动平均
    P_ma = moving_average_smooth(P_orig)
    F_ma = moving_average_smooth(F_orig)
    M_ma = moving_average_smooth(M_orig)
    plot_cost_stack_from_history(axes[1, 1], gens_orig, P_ma, F_ma, M_ma, title="移动平均", smoothed=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ 成本构成平滑对比图已保存: {save_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # --- 创建一些模拟的成本历史数据 ---
    np.random.seed(42)
    num_generations = 100
    gens = np.arange(num_generations)

    # 基础趋势线 (模拟成本下降)
    p_trend = 5e5 * np.exp(-gens / 40) + 2e5
    f_trend = 3e5 * np.exp(-gens / 50) + 1e5
    m_trend = 4e5 * (1 - np.exp(-gens / 30)) + 5e4

    # 添加一些噪声
    p_noise = np.random.normal(0, 3e4, num_generations)
    f_noise = np.random.normal(0, 2e4, num_generations)
    m_noise = np.random.normal(0, 2.5e4, num_generations)

    # 最终的成本历史数据
    cost_history_data = {
        "passenger": np.maximum(0, p_trend + np.cumsum(p_noise) * 0.1),
        "freight": np.maximum(0, f_trend + np.cumsum(f_noise) * 0.1),
        "mav": np.maximum(0, m_trend + np.cumsum(m_noise) * 0.1)
    }

    # --- 调用新的对比分析函数 ---
    plot_smoothing_comparison(cost_history_data)