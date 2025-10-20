#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修订版：公交调度甘特图绘制
2025-09-30
"""
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. 读数据
# ------------------------------------------------------------------
file_path = "best_solution_20250928_150836/simulation_details.xlsx"
df = pd.read_excel(file_path, header=0)
# df = pd.read_excel(file_path, header=0, skiprows=[1])  # 跳过第二行垃圾表头

# 清洗：确保站点ID为整数，去掉表头/脏数据
df["站点ID"] = pd.to_numeric(df["站点ID"], errors="coerce")
df = df.dropna(subset=["站点ID"])
df["站点ID"] = df["站点ID"].astype(int)

# 时间转换
df["到达时间"] = pd.to_datetime(df["到达时间"], unit="s")

# 分上下行
df_up = df[df["方向"] == "up"].copy()
df_down = df[df["方向"] == "down"].copy()

# ------------------------------------------------------------------
# 2. 绘图函数
# ------------------------------------------------------------------
def draw_station_bar_plot(data: pd.DataFrame, title: str) -> str:
    """绘制站点-时间甘特图（含车辆轨迹）"""
    # ---------------- 2.1 空值保护 ----------------
    if data.empty:
        print(f"[WARN] {title} 无数据，跳过绘图")
        return ""

    # ---------------- 2.2 输出目录 ----------------
    out_dir = "results_20250602_101935"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{title}.png")

    # ---------------- 2.3 站点顺序（地理顺序） ----------------
    # 上下行分别手动指定，确保与实际线路一致
    if "down" in title:
        station_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    else:
        station_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    stations = [s for s in station_order if s in data["站点ID"].unique()]
    if not stations:
        print(f"[WARN] {title} 无有效站点，跳过绘图")
        return ""

    # ---------------- 2.4 画布 ----------------
    plt.figure(figsize=(16, 10))
    shown = {"Passenger": False, "Freight": False}
    # shown = {"Capacity": False, "Passenger": False, "Freight": False}

    # ---------------- 2.5 动态 bar 宽度 ----------------
    time_span = (data["到达时间"].max() - data["到达时间"].min()).total_seconds()
    bar_width = np.timedelta64(max(1, int(time_span / 200)), 's')  # 至少 1 s

    # ---------------- 2.6 画站点柱状图 ----------------
    for idx, station in enumerate(stations):
        sta_data = data[data["站点ID"] == station]
        times = sta_data["到达时间"]
        cap = sta_data["调整后总模块数"]
        passenger = sta_data["调整后乘客模块"]
        freight = sta_data["调整后货物模块"]

        y_base = idx
        max_val = max(cap.max(), passenger.max(), freight.max(), 1)  # 防 0

        for t, c, p, f in zip(times, cap, passenger, freight):
            nc = c / max_val
            np_ = p / max_val
            nf = f / max_val

            # label_c = "Capacity" if not shown["Capacity"] else None
            label_p = "Passenger" if not shown["Passenger"] else None
            label_f = "Freight" if not shown["Freight"] else None

            # plt.bar(t - bar_width, nc, width=bar_width, bottom=y_base,
            #         color='gold', edgecolor='black', linewidth=0.6, label=label_c)
            plt.bar(t, np_, width=bar_width, bottom=y_base,
                    color='steelblue', edgecolor='black', linewidth=0.6, label=label_p)
            plt.bar(t + bar_width, nf, width=bar_width, bottom=y_base,
                    color='indianred', edgecolor='black', linewidth=0.6, label=label_f)

            # 数值标签（仅当模块数>0 且字体不重叠）
            if c > 0:
                plt.text(t - bar_width, y_base + nc + 0.05, f"{c}", ha='center', va='bottom', fontsize=6)
            if p > 0:
                plt.text(t, y_base + np_ + 0.05, f"{p}", ha='center', va='bottom', fontsize=6)
            if f > 0:
                plt.text(t + bar_width, y_base + nf + 0.05, f"{f}", ha='center', va='bottom', fontsize=6)

            # shown["Capacity"] = True
            shown["Passenger"] = True
            shown["Freight"] = True

    # ---------------- 2.7 画车辆轨迹 ----------------
    for _, bus in data.groupby("车辆ID"):
        bus = bus.sort_values("到达时间")
        for i in range(len(bus) - 1):
            start_time = bus.iloc[i]["到达时间"]
            end_time = bus.iloc[i + 1]["到达时间"]
            start_idx = stations.index(bus.iloc[i]["站点ID"])
            end_idx = stations.index(bus.iloc[i + 1]["站点ID"])
            plt.plot([start_time, end_time],
                     [start_idx, end_idx],
                     color='black', linewidth=1)

    # ---------------- 2.8 坐标轴与图例 ----------------
    plt.yticks(ticks=range(len(stations)), labels=[f"{s}" for s in stations])
    # plt.gca().invert_yaxis()  # ✅ 地理顺序：从上到下
    plt.xlabel("Time")
    plt.ylabel("Station ID")
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # ---------------- 2.9 保存与关闭 ----------------
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] 已保存：{out_path}")
    return out_path


# ------------------------------------------------------------------
# 3. 主程序：分别绘制上下行
# ------------------------------------------------------------------
if __name__ == "__main__":
    _ = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
    _ = draw_station_bar_plot(df_down, "Downward_Bus_Station_Bars")

# import matplotlib.pyplot as plt
# from matplotlib.dates import DateFormatter
# import numpy as np
# import pandas as pd
#
# file_path = "best_solution_20250928_150836/simulation_details.xlsx"
# # file_path = "results_20250610_091940/最优个体调度表.xlsx"
# df = pd.read_excel(file_path, header=0)  # header=0 表示第一行是标题行
#
# # 时间转换
# df["到达时间"] = pd.to_datetime(df["到达时间"], unit="s")
#
# # 分上下行
# df_up = df[df["方向"] == "上行"]
# df_down = df[df["方向"] == "下行"]
#
# # 绘图函数
# def draw_station_bar_plot(data, title):
#     plt.figure(figsize=(16, 10))
#     stations = sorted(data["站点ID"].unique())
#     shown = {"Capacity": False, "Passenger": False, "Freight": False}
#
#     for idx, station in enumerate(stations):
#         station_data = data[data["站点ID"] == station]
#         times = station_data["到达时间"]
#         cap = station_data["总模块数量"]
#         passenger = station_data["乘客模块"]
#         freight = station_data["货物模块"]
#
#         y_base = idx  # 让站点从下到上排列
#         bar_height = 0.25
#
#         bar_width = np.timedelta64(500, 'ms')
#
#         max_value = max(cap.max(), passenger.max(), freight.max())
#         for t, c, p, f in zip(times, cap, passenger, freight):
#             nc = c / max_value
#             np_ = p / max_value
#             nf = f / max_value
#
#             label_c = "Capacity" if not shown["Capacity"] else None
#             label_p = "Passenger" if not shown["Passenger"] else None
#             label_f = "Freight" if not shown["Freight"] else None
#
#             plt.bar(t - bar_width, nc, width=bar_width, bottom=y_base, color='gold', edgecolor='black', linewidth=0.6, label=label_c)
#             plt.bar(t, np_, width=bar_width, bottom=y_base, color='steelblue', edgecolor='black', linewidth=0.6, label=label_p)
#             plt.bar(t + bar_width, nf, width=bar_width, bottom=y_base, color='indianred', edgecolor='black', linewidth=0.6, label=label_f)
#
#             plt.text(t - bar_width, y_base + nc + 0.05, f"{c}", ha='center', va='bottom', fontsize=6)
#             plt.text(t, y_base + np_ + 0.05, f"{p}", ha='center', va='bottom', fontsize=6)
#             plt.text(t + bar_width, y_base + nf + 0.05, f"{f}", ha='center', va='bottom', fontsize=6)
#
#             shown["Capacity"] = True
#             shown["Passenger"] = True
#             shown["Freight"] = True
#
#     # 绘制车辆运行轨迹线段
#     for _, bus_data in data.groupby("车辆ID"):
#         bus_data = bus_data.sort_values("到达时间")
#         for i in range(len(bus_data) - 1):
#             start_time = bus_data.iloc[i]["到达时间"]
#             end_time = bus_data.iloc[i + 1]["到达时间"]
#             # start_station = len(stations) - stations.index(bus_data.iloc[i]["站点ID"])
#             # end_station = len(stations) - stations.index(bus_data.iloc[i + 1]["站点ID"])
#             start_station = stations.index(bus_data.iloc[i]["站点ID"])
#             end_station = stations.index(bus_data.iloc[i + 1]["站点ID"])
#
#             if pd.notna(start_station) and pd.notna(end_station):
#                 plt.plot([start_time, end_time], [start_station, end_station], 'black', linewidth=1)
#
#     # 图表设置
#     # plt.yticks([len(stations)-i for i in range(len(stations))], [f'{s}' for s in stations])
#     plt.yticks(ticks=range(len(stations)), labels=[f"{s}" for s in stations])
#     plt.xlabel("Time")
#     plt.ylabel("Station ID")
#     plt.title(title)
#     plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#     plt.legend(loc='upper right')
#     plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
#     plt.gcf().autofmt_xdate()
#     plt.tight_layout()
#
#     out_path = f"results_20250602_101935/{title}.png"
#     plt.savefig(out_path)
#
#     # 显示图像
#     plt.show()
#
#     plt.close()
#
#     return out_path
#
# # 绘制上下行图
# up_path = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
# down_path = draw_station_bar_plot(df_down, "Downward_Bus_Station_Bars")
# up_path, down_path
#
# # # 执行图绘制
# # plot_path_with_trajectory = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
# # plot_path_with_trajectory
