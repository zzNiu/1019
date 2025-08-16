# 测试平滑成本进化曲线绘制器
import numpy as np
import matplotlib.pyplot as plt
from smooth_cost_plotter import SmoothCostPlotter, create_smooth_evolution_plots

def generate_test_data(num_generations=30):
    """生成测试用的进化历史数据"""
    np.random.seed(42)  # 确保结果可重现
    
    # 模拟一个典型的遗传算法收敛过程
    generations = list(range(num_generations))
    
    # 最佳成本：指数衰减 + 噪声
    base_min = 1000
    decay_rate = 0.1
    min_costs = []
    
    for gen in generations:
        # 指数衰减
        cost = base_min * np.exp(-decay_rate * gen / num_generations * 3)
        # 添加随机噪声，但确保单调递减趋势
        noise = np.random.normal(0, cost * 0.05)
        if gen > 0:
            # 确保不会比前一代更差（偶尔允许小幅回升）
            if cost + noise > min_costs[-1] and np.random.random() > 0.2:
                cost = min_costs[-1] - np.random.uniform(0, 10)
            else:
                cost += noise
        else:
            cost += noise
        min_costs.append(max(cost, base_min * 0.3))  # 设置下限
    
    # 平均成本：基于最佳成本 + 额外变化
    avg_costs = []
    for i, min_cost in enumerate(min_costs):
        avg_factor = 1.2 + 0.3 * np.exp(-i / num_generations * 2)  # 逐渐接近最佳成本
        avg_cost = min_cost * avg_factor + np.random.normal(0, min_cost * 0.03)
        avg_costs.append(avg_cost)
    
    # 最差成本：基于平均成本 + 更大变化
    max_costs = []
    for i, avg_cost in enumerate(avg_costs):
        max_factor = 1.5 + 0.5 * np.exp(-i / num_generations * 1.5)
        max_cost = avg_cost * max_factor + np.random.normal(0, avg_cost * 0.05)
        max_costs.append(max_cost)
    
    # 构造logbook格式的数据
    logbook_data = []
    for gen, min_c, avg_c, max_c in zip(generations, min_costs, avg_costs, max_costs):
        logbook_data.append({
            'gen': gen,
            'min': min_c,
            'avg': avg_c,
            'max': max_c,
            'nevals': 20  # 模拟评估次数
        })
    
    return logbook_data

def test_smooth_plotter():
    """测试平滑绘制器的各种功能"""
    print("🧪 测试平滑成本进化曲线绘制器")
    print("=" * 50)
    
    # 生成测试数据
    print("📊 生成测试数据...")
    test_data = generate_test_data(25)
    print(f"✅ 生成了 {len(test_data)} 代的进化数据")
    
    # 创建绘制器
    print("\n🎨 创建平滑绘制器...")
    plotter = SmoothCostPlotter(test_data)
    print(f"✅ 绘制器创建成功，数据点数: {len(plotter.min_costs)}")
    
    # 测试各种平滑方法
    print("\n🔧 测试各种平滑方法...")
    
    # 1. 测试三次样条插值
    print("  - 测试三次样条插值...")
    try:
        x_smooth, y_smooth = plotter.cubic_spline_smooth(plotter.generations, plotter.min_costs)
        print(f"    ✅ 样条插值成功，输出点数: {len(x_smooth)}")
    except Exception as e:
        print(f"    ❌ 样条插值失败: {e}")
    
    # 2. 测试Savitzky-Golay滤波
    print("  - 测试Savitzky-Golay滤波...")
    try:
        y_savgol = plotter.savgol_smooth(plotter.min_costs)
        print(f"    ✅ S-G滤波成功，输出点数: {len(y_savgol)}")
    except Exception as e:
        print(f"    ❌ S-G滤波失败: {e}")
    
    # 3. 测试高斯滤波
    print("  - 测试高斯滤波...")
    try:
        y_gauss = plotter.gaussian_smooth(plotter.min_costs)
        print(f"    ✅ 高斯滤波成功，输出点数: {len(y_gauss)}")
    except Exception as e:
        print(f"    ❌ 高斯滤波失败: {e}")
    
    # 4. 测试移动平均
    print("  - 测试移动平均...")
    try:
        y_ma = plotter.moving_average_smooth(plotter.min_costs)
        print(f"    ✅ 移动平均成功，输出点数: {len(y_ma)}")
    except Exception as e:
        print(f"    ❌ 移动平均失败: {e}")
    
    # 生成图表
    print("\n📈 生成图表...")
    
    # 生成对比图
    print("  - 生成方法对比图...")
    try:
        fig = plotter.plot_comparison(save_path="test_smooth_comparison.png")
        print("    ✅ 对比图生成成功")
    except Exception as e:
        print(f"    ❌ 对比图生成失败: {e}")
    
    # 生成单一方法图
    methods = ['spline', 'savgol', 'gaussian', 'moving_avg']
    for method in methods:
        print(f"  - 生成{method}方法图...")
        try:
            fig = plotter.plot_best_smooth(method=method, save_path=f"test_smooth_{method}.png")
            print(f"    ✅ {method}方法图生成成功")
        except Exception as e:
            print(f"    ❌ {method}方法图生成失败: {e}")
    
    # 测试便捷函数
    print("\n🚀 测试便捷函数...")
    try:
        output_dir = create_smooth_evolution_plots(test_data, "test_smooth_output")
        print(f"    ✅ 便捷函数成功，输出目录: {output_dir}")
    except Exception as e:
        print(f"    ❌ 便捷函数失败: {e}")
    
    print("\n🎉 测试完成！")

def compare_smoothing_effects():
    """比较不同平滑方法的效果"""
    print("\n📊 比较不同平滑方法的效果")
    print("=" * 50)
    
    # 生成带有更多噪声的测试数据
    test_data = generate_test_data(20)
    plotter = SmoothCostPlotter(test_data)
    
    # 计算平滑效果指标
    original_data = plotter.min_costs
    
    methods = {
        'Savitzky-Golay': plotter.savgol_smooth(original_data),
        'Gaussian': plotter.gaussian_smooth(original_data),
        'Moving Average': plotter.moving_average_smooth(original_data)
    }
    
    # 对于样条插值，需要特殊处理
    try:
        x_smooth, spline_data = plotter.cubic_spline_smooth(plotter.generations, original_data)
        # 插值回原始点位置
        from scipy.interpolate import interp1d
        f = interp1d(x_smooth, spline_data, kind='linear', fill_value='extrapolate')
        methods['Cubic Spline'] = f(plotter.generations)
    except:
        pass
    
    print(f"原始数据标准差: {np.std(original_data):.3f}")
    print(f"原始数据变化范围: {np.max(original_data) - np.min(original_data):.3f}")
    print()
    
    for method_name, smoothed_data in methods.items():
        if len(smoothed_data) == len(original_data):
            std_dev = np.std(smoothed_data)
            data_range = np.max(smoothed_data) - np.min(smoothed_data)
            
            # 计算平滑度（相邻点差值的标准差）
            diffs = np.diff(smoothed_data)
            smoothness = np.std(diffs)
            
            print(f"{method_name}:")
            print(f"  标准差: {std_dev:.3f}")
            print(f"  数据范围: {data_range:.3f}")
            print(f"  平滑度: {smoothness:.3f}")
            print()

if __name__ == "__main__":
    # 运行测试
    test_smooth_plotter()
    compare_smoothing_effects()
    
    print("\n📝 说明:")
    print("- 生成的图片文件保存在当前目录")
    print("- 可以查看不同平滑方法的效果对比")
    print("- 样条插值通常提供最平滑的曲线")
    print("- Savitzky-Golay滤波保持数据特征较好")
    print("- 高斯滤波和移动平均适合去除噪声")
