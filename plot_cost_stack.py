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
    ax.grid(True, linestyle="--", alpha=0.4)
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

# # plot_cost_stack.py
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import numpy as np
#
# # 中文与负号（按需保留/删除）
# plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
# plt.rcParams["axes.unicode_minus"] = False
#
# def plot_cost_stack_from_history(cost_history, title="成本构成堆叠图", save_path=None):
#     """
#     根据每代最优个体的三项成本历史，绘制堆叠面积图。
#     cost_history: dict, 形如 {"passenger":[...], "freight":[...], "mav":[...]}
#     """
#     P = np.asarray(cost_history["passenger"], dtype=float)
#     F = np.asarray(cost_history["freight"], dtype=float)
#     M = np.asarray(cost_history["mav"], dtype=float)
#     assert len(P) == len(F) == len(M), "三条序列长度需一致"
#
#     gens = np.arange(len(P))
#     fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
#
#     ax.stackplot(gens, P, F, M,
#                  labels=["乘客等待成本", "货物等待成本", "MAV运输成本"],
#                  linewidth=0.6, edgecolor="white", alpha=0.9)
#     ax.plot(gens, P + F + M, linewidth=1.0)  # 总成本细线
#
#     ax.set_title(title, fontsize=14)
#     ax.set_xlabel("代数", fontsize=12)
#     ax.set_ylabel("成本值", fontsize=12)
#     ax.grid(True, linestyle="--", alpha=0.4)
#     ax.legend(loc="upper right", frameon=True)
#
#     fmt = ScalarFormatter(useMathText=True)
#     fmt.set_powerlimits((0, 0))          # 强制科学计数法，如 1e6
#     ax.yaxis.set_major_formatter(fmt)
#
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)
#     print(f"✅ 成本构成堆叠图已保存: {save_path}")