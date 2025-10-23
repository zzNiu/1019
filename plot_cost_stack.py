# # plot_cost_stack.py
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import numpy as np
# import pandas as pd  # <--- 新增：用于读取 Excel
# import sys  # <--- 新增：用于在出错时退出
#
# # 中文与负号
# plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
# plt.rcParams["axes.unicode_minus"] = False
#
#
# def read_costs_from_excel(file_path, sheet_name=0):
#     """
#     从 Excel 文件中读取成本数据并转换为 cost_history 字典。
#     """
#     print(f"ℹ️ 正在尝试从 '{file_path}' (工作表: {sheet_name}) 读取数据...")
#     try:
#         # 读取 Excel 文件
#         df = pd.read_excel(file_path, sheet_name=sheet_name)
#
#         # 检查必需的列是否存在
#         required_columns = [
#             "mav_transport",
#             "passenger_waiting",
#             "freight_waiting",
#             "unserved_penalty_cost"
#         ]
#
#         missing_cols = [col for col in required_columns if col not in df.columns]
#         if missing_cols:
#             print(f"❌ 错误: Excel 文件中缺少以下必需的列: {missing_cols}")
#             print("  请确保 Excel 的列标题与代码中的键名完全一致。")
#             return None
#
#         # 将 DataFrame 转换为字典，格式为: {'列名': [值1, 值2, ...]}
#         cost_history = df.to_dict('list')
#         print(f"✅ 成功读取 {len(df)} 行数据。")
#         return cost_history
#
#     except FileNotFoundError:
#         print(f"❌ 错误: 找不到文件 '{file_path}'。请检查文件名和路径是否正确。")
#         return None
#     except Exception as e:
#         print(f"❌ 读取 Excel 时发生未知错误: {e}")
#         return None
#
#
# def plot_cost_stack_from_history(cost_history, title="成本构成堆叠图", save_path=None):
#     """
#     根据每代最优个体的多项成本历史，绘制堆叠面积图。
#     cost_history: dict, 包含所有成本项的历史列表
#     """
#
#     # === 从 cost_history 中提取所有成本项 ===
#     try:
#         mav_transport = np.asarray(cost_history["mav_transport"], dtype=float)
#         passenger_waiting = np.asarray(cost_history["passenger_waiting"], dtype=float)
#         freight_waiting = np.asarray(cost_history["freight_waiting"], dtype=float)
#         unserved_penalty_cost = np.asarray(cost_history["unserved_penalty_cost"], dtype=float)
#     except KeyError as e:
#         print(f"❌ 绘图错误: 'cost_history' 字典中缺少键: {e}")
#         print("  请确保 Excel 的列标题与代码中的键名完全一致。")
#         return
#     except (TypeError, ValueError) as e:
#         print(f"❌ 绘图错误: 成本数据中可能包含非数值: {e}")
#         return
#
#     # 检查数据长度
#     all_costs = [mav_transport, passenger_waiting, freight_waiting, unserved_penalty_cost]
#     lengths = {len(arr) for arr in all_costs}
#
#     if not lengths:
#         print("❌ 错误: 所有成本序列均为空，无法绘图。")
#         return
#
#     if len(lengths) > 1:
#         print(
#             f"⚠️ 警告: 成本序列长度不一致: { {k: len(v) for k, v in cost_history.items() if k in ['mav_transport', 'passenger_waiting', 'freight_waiting', 'unserved_penalty_cost']} }")
#         # 找到最短的长度以避免绘图错误
#         min_len = min(lengths)
#         if min_len == 0:
#             print("❌ 错误: 存在空的成本序列，无法绘图。")
#             return
#
#         print(f"  将截断所有序列为最短长度: {min_len}")
#         mav_transport = mav_transport[:min_len]
#         passenger_waiting = passenger_waiting[:min_len]
#         freight_waiting = freight_waiting[:min_len]
#         unserved_penalty_cost = unserved_penalty_cost[:min_len]
#         all_costs = [mav_transport, passenger_waiting, freight_waiting, unserved_penalty_cost]
#
#     gens = np.arange(len(passenger_waiting))
#     fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
#
#     # === 定义数据堆叠和对应的图例标签 ===
#     # ！！！注意：这里的顺序必须与您希望堆叠的顺序一致
#     data_stack = [
#         passenger_waiting,  # 乘客等待（蓝色，在最下面）
#         freight_waiting,  # 货物等待（橙色）
#         mav_transport,  # MAV运输（绿色）
#         unserved_penalty_cost  # 未服务惩罚（红色，在最上面）
#     ]
#
#     labels = [
#         "乘客等待成本",
#         "货物等待成本",
#         "MAV运输成本",
#         "未服务需求惩罚",
#     ]
#
#     # 颜色（可选，但为了匹配您的原图，我指定一下）
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#
#     # === 使用新的数据和标签进行绘图 ===
#     ax.stackplot(gens, *data_stack,
#                  labels=labels,
#                  colors=colors,
#                  linewidth=0.5,  # <--- 已应用：黑色描边
#                  edgecolor="black",  # <--- 已应用：黑色描边
#                  alpha=0.9)
#
#     # 计算并绘制总成本曲线
#     total_cost = sum(data_stack)
#     ax.plot(gens, total_cost, color='black', linestyle='--', linewidth=1.5, label="总成本")
#
#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel("代数" if "代数" in title else "迭代", fontsize=12)  # 匹配您的原图
#     ax.set_ylabel("成本值", fontsize=12)
#
#     # <--- 已应用：弱化网格
#     ax.grid(True, linestyle="--", alpha=0.25)
#
#     ax.legend(loc="upper right", frameon=True)
#
#     # 设置Y轴为科学计数法
#     fmt = ScalarFormatter(useMathText=True)
#     fmt.set_powerlimits((0, 0))
#     ax.yaxis.set_major_formatter(fmt)
#
#     # 设置Y轴从0开始
#     ax.set_ylim(bottom=0)
#
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"✅ 成本构成堆叠图已保存: {save_path}")
#     plt.show()
#     plt.close(fig)
#
#
# # ===================================================================
# #                       主程序入口
# # ===================================================================
# if __name__ == "__main__":
#
#     # --- 1. 在这里配置您的文件 ---
#
#     # ⚠️ 替换为您 Excel 文件的实际名称或完整路径
#     EXCEL_FILE_PATH = "your_cost_data.xlsx"
#
#     # 如果您的数据不在第一个工作表，请指定名称或索引，例如: "Sheet1" 或 0
#     SHEET_NAME = 0
#
#     # 最终保存的图片文件名
#     OUTPUT_IMAGE_PATH = "cost_stack_plot.png"
#
#     # 图像标题 (您的原图标题是 "平均成本构成进化堆叠图")
#     PLOT_TITLE = "平均成本构成进化堆叠图"
#
#     # --- 2. 执行 ---
#
#     # 从 Excel 读取数据
#     cost_history_data = read_costs_from_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
#
#     # 如果数据读取成功，则绘图
#     if cost_history_data:
#         plot_cost_stack_from_history(
#             cost_history_data,
#             title=PLOT_TITLE,
#             save_path=OUTPUT_IMAGE_PATH
#         )
#     else:
#         print("❌ 程序终止，因为无法从 Excel 文件中读取数据。")
#         sys.exit(1)  # 退出程序


# plot_cost_stack.py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# 中文与负号（按需保留/删除）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_cost_stack_from_history(cost_history, title="成本构成堆叠图", save_path=None):
    """
    根据每代最优个体的多项成本历史，绘制堆叠面积图。
    cost_history: dict, 包含所有成本项的历史列表
    """
    # === 更新：从 cost_history 中提取所有成本项 ===

    mav_transport = np.asarray(cost_history["mav_transport"], dtype=float)
    passenger_waiting = np.asarray(cost_history["passenger_waiting"], dtype=float)
    freight_waiting = np.asarray(cost_history["freight_waiting"], dtype=float)
    unserved_penalty_cost = np.asarray(cost_history["unserved_penalty_cost"], dtype=float)
    # unserved_passenger = np.asarray(cost_history["unserved_passenger"], dtype=float)
    # unserved_freight = np.asarray(cost_history["unserved_freight"], dtype=float)

    # 确保所有成本序列的长度一致
    lengths = {len(v) for k, v in cost_history.items()}
    if len(lengths) > 1:
        print(f"❌ 警告: cost_history 中的成本序列长度不一致: { {k: len(v) for k, v in cost_history.items()} }")
        # 找到最短的长度以避免绘图错误
        min_len = min(lengths)
        passenger_waiting = passenger_waiting[:min_len]
        freight_waiting = freight_waiting[:min_len]
        mav_transport = mav_transport[:min_len]
        # unserved_passenger = unserved_passenger[:min_len]
        # unserved_freight = unserved_freight[:min_len]
        # energy = energy[:min_len]

    gens = np.arange(len(passenger_waiting))
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)  # 适当调整画布大小以容纳更多图例

    # === 更新：定义数据堆叠和对应的图例标签 ===
    data_stack = [
        mav_transport,
        passenger_waiting,
        freight_waiting,
        unserved_penalty_cost,
    ]

    labels = [
        "乘客等待成本",
        "货物等待成本",
        "MAV运输成本",
        "未服务需求惩罚",
        # "未服务货物惩罚",
        # "MAV能量成本"
    ]

    # === 更新：使用新的数据和标签进行绘图 ===
    ax.stackplot(gens, *data_stack,
                 labels=labels,
                 linewidth=0.6,
                 edgecolor="white",
                 alpha=0.9)

    # 计算并绘制总成本曲线
    total_cost = sum(data_stack)
    ax.plot(gens, total_cost, color='black', linestyle='--', linewidth=1.5, label="总成本")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("代数", fontsize=12)
    ax.set_ylabel("成本值", fontsize=12)

    # ax.grid(True, linestyle="--", alpha=0.4)
    # <--- 更改点 2: 降低 alpha 值（透明度）来弱化网格
    ax.grid(True, linestyle="--", alpha=0.25)

    ax.legend(loc="upper right", frameon=True)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))  # 强制科学计数法，如 1e6
    ax.yaxis.set_major_formatter(fmt)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ 成本构成堆叠图已保存: {save_path}")
    plt.show()
    plt.close(fig)
