# plot_cost_stack.py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# 中文与负号（按需保留/删除）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

def plot_cost_stack_from_history(cost_history, title="成本构成堆叠图", save_path=None):
    """
    根据每代最优个体的三项成本历史，绘制堆叠面积图。
    cost_history: dict, 形如 {"passenger":[...], "freight":[...], "mav":[...]}
    """
    P = np.asarray(cost_history["passenger"], dtype=float)
    F = np.asarray(cost_history["freight"], dtype=float)
    M = np.asarray(cost_history["mav"], dtype=float)
    assert len(P) == len(F) == len(M), "三条序列长度需一致"

    gens = np.arange(len(P))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)

    ax.stackplot(gens, P, F, M,
                 labels=["乘客等待成本", "货物等待成本", "MAV运输成本"],
                 linewidth=0.6, edgecolor="white", alpha=0.9)
    ax.plot(gens, P + F + M, linewidth=1.0)  # 总成本细线

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("代数", fontsize=12)
    ax.set_ylabel("成本值", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))          # 强制科学计数法，如 1e6
    ax.yaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 成本构成堆叠图已保存: {save_path}")