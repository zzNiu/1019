import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd

file_path = "results_20250610_091940/最优个体调度表.xlsx"
df = pd.read_excel(file_path, header=0)  # header=0 表示第一行是标题行

# 时间转换
df["到达时间"] = pd.to_datetime(df["到达时间"], unit="s")

# 分上下行
df_up = df[df["方向"] == "上行"]
df_down = df[df["方向"] == "下行"]

# 绘图函数
def draw_station_bar_plot(data, title):
    plt.figure(figsize=(16, 10))
    stations = sorted(data["站点ID"].unique())
    shown = {"Capacity": False, "Passenger": False, "Freight": False}

    for idx, station in enumerate(stations):
        station_data = data[data["站点ID"] == station]
        times = station_data["到达时间"]
        cap = station_data["总模块数量"]
        passenger = station_data["乘客模块"]
        freight = station_data["货物模块"]

        y_base = idx  # 让站点从下到上排列
        bar_height = 0.25

        bar_width = np.timedelta64(500, 'ms')

        max_value = max(cap.max(), passenger.max(), freight.max())
        for t, c, p, f in zip(times, cap, passenger, freight):
            nc = c / max_value
            np_ = p / max_value
            nf = f / max_value

            label_c = "Capacity" if not shown["Capacity"] else None
            label_p = "Passenger" if not shown["Passenger"] else None
            label_f = "Freight" if not shown["Freight"] else None

            plt.bar(t - bar_width, nc, width=bar_width, bottom=y_base, color='gold', edgecolor='black', linewidth=0.6, label=label_c)
            plt.bar(t, np_, width=bar_width, bottom=y_base, color='steelblue', edgecolor='black', linewidth=0.6, label=label_p)
            plt.bar(t + bar_width, nf, width=bar_width, bottom=y_base, color='indianred', edgecolor='black', linewidth=0.6, label=label_f)

            plt.text(t - bar_width, y_base + nc + 0.05, f"{c}", ha='center', va='bottom', fontsize=6)
            plt.text(t, y_base + np_ + 0.05, f"{p}", ha='center', va='bottom', fontsize=6)
            plt.text(t + bar_width, y_base + nf + 0.05, f"{f}", ha='center', va='bottom', fontsize=6)

            shown["Capacity"] = True
            shown["Passenger"] = True
            shown["Freight"] = True

    # 绘制车辆运行轨迹线段
    for _, bus_data in data.groupby("车辆ID"):
        bus_data = bus_data.sort_values("到达时间")
        for i in range(len(bus_data) - 1):
            start_time = bus_data.iloc[i]["到达时间"]
            end_time = bus_data.iloc[i + 1]["到达时间"]
            # start_station = len(stations) - stations.index(bus_data.iloc[i]["站点ID"])
            # end_station = len(stations) - stations.index(bus_data.iloc[i + 1]["站点ID"])
            start_station = stations.index(bus_data.iloc[i]["站点ID"])
            end_station = stations.index(bus_data.iloc[i + 1]["站点ID"])

            if pd.notna(start_station) and pd.notna(end_station):
                plt.plot([start_time, end_time], [start_station, end_station], 'black', linewidth=1)

    # 图表设置
    # plt.yticks([len(stations)-i for i in range(len(stations))], [f'{s}' for s in stations])
    plt.yticks(ticks=range(len(stations)), labels=[f"{s}" for s in stations])
    plt.xlabel("Time")
    plt.ylabel("Station ID")
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    out_path = f"results_20250602_101935/{title}.png"
    plt.savefig(out_path)

    # 显示图像
    plt.show()

    plt.close()

    return out_path

# 绘制上下行图
up_path = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
down_path = draw_station_bar_plot(df_down, "Downward_Bus_Station_Bars")
up_path, down_path

# # 执行图绘制
# plot_path_with_trajectory = draw_station_bar_plot(df_up, "Upward_Bus_Station_Bars")
# plot_path_with_trajectory
