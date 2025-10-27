#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修订版：公交调度甘特图绘制 (已模块化)
2025-09-30
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator
import numpy as np
import pandas as pd

# 设置 Matplotlib 后端为 'Agg'，避免在服务器上运行时尝试打开GUI窗口
matplotlib.use('Agg')


# ------------------------------------------------------------------
# 1. 绘图核心函数
# ------------------------------------------------------------------
def draw_station_bar_plot(data: pd.DataFrame, title: str, save_dir: str) -> str:
    """
    绘制站点-时间甘特图（含车辆轨迹）

    参数:
    data: pd.DataFrame - 包含单向 (上行或下行) 的仿真详情数据
    title: str - 图像标题 (也将用作文件名)
    save_dir: str - 图像的保存目录 (由 main.py 传入)
    """
    # ---------------- 1.1 空值保护 ----------------
    if data.empty:
        print(f"  [WARN] {title} 无数据，跳过甘特图绘制")
        return ""

    # ---------------- 1.2 输出路径 ----------------
    # (已修改) 不再创建目录，而是使用传入的 save_dir
    # (已修改) 文件名使用 title 动态生成
    out_path = os.path.join(save_dir, f"{title}.png")

    # ---------------- 1.3 站点顺序（地理顺序） ----------------
    # 上下行分别手动指定，确保与实际线路一致
    if "down" in title.lower():  # 使用 .lower() 增加鲁棒性
        station_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    else:
        station_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    stations = [s for s in station_order if s in data["站点ID"].unique()]
    if not stations:
        print(f"  [WARN] {title} 无有效站点，跳过甘特图绘制")
        return ""

    # ---------------- 1.4 画布 ----------------
    fig, ax = plt.subplots(figsize=(16, 10))
    # shown = {"Passenger": False, "Freight": False}
    shown = {"Capacity": False, "Passenger": False, "Freight": False}

    # ---------------- 1.5 动态 bar 宽度 ----------------
    time_span = (data["到达时间"].max() - data["到达时间"].min()).total_seconds()
    bar_width = np.timedelta64(max(1, int(time_span / 200)), 's')  # 至少 1 s

    # ---------------- 1.6 画站点柱状图 ----------------
    for idx, station in enumerate(stations):
        sta_data = data[data["站点ID"] == station]
        times = sta_data["到达时间"]

        # (已修改) 从 simulation_details.xlsx 中读取正确的列名
        cap_col_name = "调整后总模块数" if "调整后总模块数" in sta_data.columns else "总模块数量"
        pax_col_name = "调整后乘客模块" if "调整后乘客模块" in sta_data.columns else "乘客模块"
        frt_col_name = "调整后货物模块" if "调整后货物模块" in sta_data.columns else "货物模块"

        cap = sta_data[cap_col_name]
        passenger = sta_data[pax_col_name]
        freight = sta_data[frt_col_name]

        y_base = idx
        max_val = max(cap.max(), passenger.max(), freight.max(), 1)  # 防 0

        for t, c, p, f in zip(times, cap, passenger, freight):
            nc = c / max_val
            np_ = p / max_val
            nf = f / max_val

            label_c = "Capacity" if not shown["Capacity"] else None
            label_p = "Passenger" if not shown["Passenger"] else None
            label_f = "Freight" if not shown["Freight"] else None

            plt.bar(t - bar_width, nc, width=bar_width, bottom=y_base,
                    color='gold', edgecolor='black', linewidth=0.6, label=label_c)
            ax.bar(t, np_, width=bar_width, bottom=y_base,
                   color='steelblue', edgecolor='black', linewidth=0.6, label=label_p)
            ax.bar(t + bar_width, nf, width=bar_width, bottom=y_base,
                   color='indianred', edgecolor='black', linewidth=0.6, label=label_f)

            # 数值标签（仅当模块数>0 且字体不重叠）
            if c > 0:
                ax.text(t - bar_width, y_base + nc + 0.05, f"{c}", ha='center', va='bottom', fontsize=6)
            if p > 0:
                ax.text(t, y_base + np_ + 0.05, f"{p}", ha='center', va='bottom', fontsize=6)
            if f > 0:
                ax.text(t + bar_width, y_base + nf + 0.05, f"{f}", ha='center', va='bottom', fontsize=6)

            shown["Capacity"] = True
            shown["Passenger"] = True
            shown["Freight"] = True

    # ---------------- 1.7 画车辆轨迹 ----------------
    for _, bus in data.groupby("车辆ID"):
        bus = bus.sort_values("到达时间")
        for i in range(len(bus) - 1):
            start_time = bus.iloc[i]["到达时间"]
            end_time = bus.iloc[i + 1]["到达时间"]
            start_idx = stations.index(bus.iloc[i]["站点ID"])
            end_idx = stations.index(bus.iloc[i + 1]["站点ID"])
            ax.plot([start_time, end_time],
                    [start_idx, end_idx],
                    color='black', linewidth=1)

    # ---------------- 1.8 坐标轴与图例 ----------------
    ax.set_yticks(ticks=range(len(stations)))
    ax.set_yticklabels([f"{s}" for s in stations])
    # ax.invert_yaxis()  # ✅ 地理顺序：从上到下
    ax.set_xlabel("Time (Minute)")
    ax.set_ylabel("Station ID")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    # ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # 设置横坐标刻度：每1分钟显示1个（interval=1可调整，如5分钟显示1个则设为5）
    ax.xaxis.set_major_locator(MinuteLocator(interval=5))
    fig.autofmt_xdate()
    plt.tight_layout()

    # ---------------- 1.9 保存与关闭 ----------------
    try:
        plt.savefig(out_path, dpi=300)
        print(f"  [INFO] 甘特图已保存：{out_path}")
    except Exception as e:
        print(f"  [ERROR] 保存甘特图 {out_path} 失败: {e}")
    plt.close(fig)  # 关闭图像，释放内存
    return out_path


# ------------------------------------------------------------------
# 2. (新增) 主调用函数
# ------------------------------------------------------------------
def generate_schedule_gantt_charts(simulation_details_df: pd.DataFrame, save_dir: str):
    """
    (入口函数) 根据仿真详情 DataFrame 绘制并保存上下行甘特图。

    参数:
    simulation_details_df: pd.DataFrame - 从 simulation_results['df_enriched'] 传入的完整数据
    save_dir: str - 结果保存目录
    """

    # ---------------- 2.1 数据准备 ----------------
    df = simulation_details_df.copy()

    # 清洗：确保站点ID为整数，去掉表头/脏数据
    df["站点ID"] = pd.to_numeric(df["站点ID"], errors="coerce")
    df = df.dropna(subset=["站点ID"])
    df["站点ID"] = df["站点ID"].astype(int)

    # 时间转换
    df["到达时间"] = pd.to_datetime(df["到达时间"], unit="m")

    # 分上下行
    df_up = df[df["方向"] == "up"].copy()
    df_down = df[df["方向"] == "down"].copy()

    # ---------------- 2.2 主程序：分别绘制上下行 ----------------
    _ = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars", save_dir)
    _ = draw_station_bar_plot(df_down, "Downward_Bus_Station_Bars", save_dir)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 修订版：公交调度甘特图绘制
# 2025-09-30
# """
# import os
# import matplotlib.pyplot as plt
# from matplotlib.dates import DateFormatter
# import numpy as np
# import pandas as pd
#
# # ------------------------------------------------------------------
# # 1. 读数据
# # ------------------------------------------------------------------
# file_path = "best_solution_ω_0.15_φ_0.85_20251023_222353/simulation_details.xlsx"
# df = pd.read_excel(file_path, header=0)
# # df = pd.read_excel(file_path, header=0, skiprows=[1])  # 跳过第二行垃圾表头
#
# # 清洗：确保站点ID为整数，去掉表头/脏数据
# df["站点ID"] = pd.to_numeric(df["站点ID"], errors="coerce")
# df = df.dropna(subset=["站点ID"])
# df["站点ID"] = df["站点ID"].astype(int)
#
# # 时间转换
# df["到达时间"] = pd.to_datetime(df["到达时间"], unit="s")
#
# # 分上下行
# df_up = df[df["方向"] == "up"].copy()
# df_down = df[df["方向"] == "down"].copy()
#
# # ------------------------------------------------------------------
# # 2. 绘图函数
# # ------------------------------------------------------------------
# def draw_station_bar_plot(data: pd.DataFrame, title: str) -> str:
#     """绘制站点-时间甘特图（含车辆轨迹）"""
#     # ---------------- 2.1 空值保护 ----------------
#     if data.empty:
#         print(f"[WARN] {title} 无数据，跳过绘图")
#         return ""
#
#     # ---------------- 2.2 输出目录 ----------------
#     out_dir = "results_20250602_101935"
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{title}.png")
#
#     # ---------------- 2.3 站点顺序（地理顺序） ----------------
#     # 上下行分别手动指定，确保与实际线路一致
#     if "down" in title:
#         station_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#     else:
#         station_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#     stations = [s for s in station_order if s in data["站点ID"].unique()]
#     if not stations:
#         print(f"[WARN] {title} 无有效站点，跳过绘图")
#         return ""
#
#     # ---------------- 2.4 画布 ----------------
#     plt.figure(figsize=(16, 10))
#     shown = {"Passenger": False, "Freight": False}
#     # shown = {"Capacity": False, "Passenger": False, "Freight": False}
#
#     # ---------------- 2.5 动态 bar 宽度 ----------------
#     time_span = (data["到达时间"].max() - data["到达时间"].min()).total_seconds()
#     bar_width = np.timedelta64(max(1, int(time_span / 200)), 's')  # 至少 1 s
#
#     # ---------------- 2.6 画站点柱状图 ----------------
#     for idx, station in enumerate(stations):
#         sta_data = data[data["站点ID"] == station]
#         times = sta_data["到达时间"]
#         cap = sta_data["调整后总模块数"]
#         passenger = sta_data["调整后乘客模块"]
#         freight = sta_data["调整后货物模块"]
#
#         y_base = idx
#         max_val = max(cap.max(), passenger.max(), freight.max(), 1)  # 防 0
#
#         for t, c, p, f in zip(times, cap, passenger, freight):
#             nc = c / max_val
#             np_ = p / max_val
#             nf = f / max_val
#
#             # label_c = "Capacity" if not shown["Capacity"] else None
#             label_p = "Passenger" if not shown["Passenger"] else None
#             label_f = "Freight" if not shown["Freight"] else None
#
#             # plt.bar(t - bar_width, nc, width=bar_width, bottom=y_base,
#             #         color='gold', edgecolor='black', linewidth=0.6, label=label_c)
#             plt.bar(t, np_, width=bar_width, bottom=y_base,
#                     color='steelblue', edgecolor='black', linewidth=0.6, label=label_p)
#             plt.bar(t + bar_width, nf, width=bar_width, bottom=y_base,
#                     color='indianred', edgecolor='black', linewidth=0.6, label=label_f)
#
#             # 数值标签（仅当模块数>0 且字体不重叠）
#             if c > 0:
#                 plt.text(t - bar_width, y_base + nc + 0.05, f"{c}", ha='center', va='bottom', fontsize=6)
#             if p > 0:
#                 plt.text(t, y_base + np_ + 0.05, f"{p}", ha='center', va='bottom', fontsize=6)
#             if f > 0:
#                 plt.text(t + bar_width, y_base + nf + 0.05, f"{f}", ha='center', va='bottom', fontsize=6)
#
#             # shown["Capacity"] = True
#             shown["Passenger"] = True
#             shown["Freight"] = True
#
#     # ---------------- 2.7 画车辆轨迹 ----------------
#     for _, bus in data.groupby("车辆ID"):
#         bus = bus.sort_values("到达时间")
#         for i in range(len(bus) - 1):
#             start_time = bus.iloc[i]["到达时间"]
#             end_time = bus.iloc[i + 1]["到达时间"]
#             start_idx = stations.index(bus.iloc[i]["站点ID"])
#             end_idx = stations.index(bus.iloc[i + 1]["站点ID"])
#             plt.plot([start_time, end_time],
#                      [start_idx, end_idx],
#                      color='black', linewidth=1)
#
#     # ---------------- 2.8 坐标轴与图例 ----------------
#     plt.yticks(ticks=range(len(stations)), labels=[f"{s}" for s in stations])
#     # plt.gca().invert_yaxis()  # ✅ 地理顺序：从上到下
#     plt.xlabel("Time")
#     plt.ylabel("Station ID")
#     plt.title(title)
#     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#     plt.legend(loc='upper right')
#     plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
#     plt.gcf().autofmt_xdate()
#     plt.tight_layout()
#
#     # ---------------- 2.9 保存与关闭 ----------------
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"[INFO] 已保存：{out_path}")
#     return out_path
#
#
# # ------------------------------------------------------------------
# # 3. 主程序：分别绘制上下行
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     _ = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
#     _ = draw_station_bar_plot(df_down, "Downward_Bus_Station_Bars")
#
#
