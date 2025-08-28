import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ==============================================================================
# --- 用户配置区 ---
# ==============================================================================

# 1. 指定仿真程序输出的Excel文件路径
EXCEL_FILE_PATH = 'best_solution_20250827_234942/simulation_details.xlsx'  # <--- 确保这个文件名与您保存的文件名一致

# 2. 列名映射
COLUMN_MAPPING = {
    'train_id': '车辆ID',
    'station': '站点ID',
    'direction': '方向',
    'arrival_time': '到达时间',
    'num_p_modules': '调整后乘客模块',
    'num_f_modules': '调整后货物模块',
    'on_board_p': '上车后在车乘客数量',
    'on_board_f': '上车后在车货物数量',
}

# 3. 绘图参数配置
STOP_DURATION_MINUTES = 1
MAX_MODULES_FOR_SCALING = 10
MAX_PASSENGERS_FOR_SCALING = 100


# ==============================================================================
# --- 核心代码区 ---
# ==============================================================================

def load_and_process_simulation_data(file_path, col_map):
    """
    从Excel加载完整数据并进行预处理。
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请先运行仿真程序生成该文件。")
        return None

    inverted_map = {v: k for k, v in col_map.items() if v in df.columns}
    df.rename(columns=inverted_map, inplace=True)

    if 'direction' not in df.columns:
        print("错误：数据中缺少 'direction' 列，无法按方向拆分绘图。")
        return None

    start_time = datetime.strptime("00:00:00", "%H:%M:%S")
    df['arrival_time_dt'] = df['arrival_time'].apply(lambda x: start_time + timedelta(minutes=x))
    df['departure_time_dt'] = df['arrival_time_dt'] + timedelta(minutes=STOP_DURATION_MINUTES)

    return df


def transform_df_to_schedule(df):
    """
    将DataFrame转换为绘图函数需要的嵌套字典结构。
    """
    stations = sorted(df['station'].unique().tolist())
    schedule_data = []
    for train_name, train_group in df.groupby('train_id'):
        train_info = {'train_id': train_name}
        stops = train_group.sort_values(by='arrival_time').to_dict('records')
        train_info['stops'] = stops
        schedule_data.append(train_info)
    return schedule_data, stations


def plot_gantt_chart_on_ax(ax, schedule_data, stations, title):
    """
    在一个指定的matplotlib Axes对象(ax)上绘制甘特图。
    """
    if not schedule_data or not stations:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # --- 优化后的条形图参数 ---
    bar_height_scale_factor = 1.5  # 调整回合理的高度缩放
    capacity_bar_width = timedelta(seconds=100)  # 容量条（模块）更宽
    load_bar_width = timedelta(seconds=80)  # 负载条（乘客/货物）更窄
    group_offset = timedelta(seconds=30)  # 乘客组和货物组之间的偏移

    for train in schedule_data:
        stops = train['stops']
        for i in range(len(stops)):
            current_stop = stops[i]
            arrival_time = current_stop['arrival_time_dt']
            station_y_pos = stations.index(current_stop['station'])

            if i > 0:
                prev_stop = stops[i - 1]
                prev_departure_time = prev_stop['departure_time_dt']
                prev_station_y_pos = stations.index(prev_stop['station'])
                ax.plot([prev_departure_time, arrival_time],
                        [prev_station_y_pos, station_y_pos],
                        color='dimgray', linewidth=1.5, solid_capstyle='round', zorder=1)

            # --- 优化后的条形图绘制逻辑 ---
            # 乘客组（左侧）
            passenger_group_x = arrival_time - group_offset
            # 1a. 乘客模块数 (半透明的宽底座)
            h_p_mod = (current_stop.get('num_p_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
            ax.bar(passenger_group_x, h_p_mod, width=capacity_bar_width, bottom=station_y_pos,
                   color='lightblue', align='center', alpha=0.6, label="Passenger Modules")
            # 1b. 在车乘客数 (不透明的窄条，覆盖在底座上)
            h_p_board = (current_stop.get('on_board_p', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
            ax.bar(passenger_group_x, h_p_board, width=load_bar_width, bottom=station_y_pos,
                   color='blue', align='center', label="On-board Passengers", zorder=2)

            # 货物组（右侧）
            freight_group_x = arrival_time + group_offset
            # 2a. 货物模块数 (半透明的宽底座)
            h_f_mod = (current_stop.get('num_f_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
            ax.bar(freight_group_x, h_f_mod, width=capacity_bar_width, bottom=station_y_pos,
                   color='lightgreen', align='center', alpha=0.6, label="Freight Modules")
            # 2b. 在车货物数 (不透明的窄条，覆盖在底座上)
            h_f_board = (current_stop.get('on_board_f', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
            ax.bar(freight_group_x, h_f_board, width=load_bar_width, bottom=station_y_pos,
                   color='darkgreen', align='center', label="On-board Freight", zorder=2)

    # --- 图表美化 ---
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations, fontsize=12)
    ax.set_ylim(-0.5, len(stations) - 0.5)
    ax.set_xlabel('Time', fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)
    ax.set_title(title, fontsize=16, pad=20)


# --- 主程序入口 ---
if __name__ == '__main__':
    full_df = load_and_process_simulation_data(EXCEL_FILE_PATH, COLUMN_MAPPING)

    if full_df is not None:
        # 创建一个图窗和两个水平排列、共享Y轴的子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
        fig.suptitle('Vehicle Schedule Analysis', fontsize=20, y=0.98)  # 添加总标题

        # --- 处理和绘制 'up' 方向的图 ---
        df_up = full_df[full_df['direction'] == 'up'].copy()
        if not df_up.empty:
            schedule_data_up, stations_up = transform_df_to_schedule(df_up)
            plot_gantt_chart_on_ax(ax1, schedule_data_up, stations_up, 'Direction: Up')
        else:
            ax1.text(0.5, 0.5, 'No data for "Up" direction', ha='center', va='center')
        ax1.set_ylabel('Station ID', fontsize=14)  # 为左侧图表添加Y轴标签

        # --- 处理和绘制 'down' 方向的图 ---
        df_down = full_df[full_df['direction'] == 'down'].copy()
        if not df_down.empty:
            schedule_data_down, stations_down = transform_df_to_schedule(df_down)
            plot_gantt_chart_on_ax(ax2, schedule_data_down, stations_down, 'Direction: Down')
        else:
            ax2.text(0.5, 0.5, 'No data for "Down" direction', ha='center', va='center')
        ax2.set_ylabel('Station ID', fontsize=14)  # 为右侧图表添加Y轴标签

        # --- 创建统一的图例 ---
        handles, labels = ax1.get_legend_handles_labels()
        # 去除重复的图例项
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center',
                   bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局为总标题和图例留出空间

        output_filename = 'gantt_chart_optimized.png'
        plt.savefig(output_filename)
        print(f"\n已将优化后的图表保存到 {output_filename}")

        plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from datetime import datetime, timedelta
#
# # ==============================================================================
# # --- 用户配置区 ---
# # ==============================================================================
#
# # 1. 指定仿真程序输出的Excel文件路径
# EXCEL_FILE_PATH = 'best_solution_20250827_225403/simulation_details.xlsx'  # <--- 确保这个文件名与您保存的文件名一致
#
# # 2. 列名映射
# COLUMN_MAPPING = {
#     # 标准名          # 您Excel中的实际列名
#     'train_id': '车辆ID',
#     'station': '站点ID',
#     'direction': '方向',
#     'arrival_time': '到达时间',
#     'num_p_modules': '调整后乘客模块',
#     'num_f_modules': '调整后货物模块',
#     'on_board_p': '上车后在车乘客数量',
#     'on_board_f': '上车后在车货物数量',
# }
#
# # 3. 绘图参数配置
# STOP_DURATION_MINUTES = 1
# MAX_MODULES_FOR_SCALING = 10
# MAX_PASSENGERS_FOR_SCALING = 100
#
# # ==============================================================================
# # --- 核心代码区 ---
# # ==============================================================================
#
# def load_and_process_simulation_data(file_path, col_map):
#     """
#     从Excel加载完整数据并进行预处理。
#     """
#     try:
#         df = pd.read_excel(file_path)
#     except FileNotFoundError:
#         print(f"错误：文件 '{file_path}' 未找到。请先运行仿真程序生成该文件。")
#         return None
#
#     inverted_map = {v: k for k, v in col_map.items() if v in df.columns}
#     df.rename(columns=inverted_map, inplace=True)
#
#     if 'direction' not in df.columns:
#         print("错误：数据中缺少 'direction' 列，无法按方向拆分绘图。")
#         return None
#
#     start_time = datetime.strptime("00:00:00", "%H:%M:%S")
#     df['arrival_time_dt'] = df['arrival_time'].apply(lambda x: start_time + timedelta(minutes=x))
#     df['departure_time_dt'] = df['arrival_time_dt'] + timedelta(minutes=STOP_DURATION_MINUTES)
#
#     return df
#
#
# def transform_df_to_schedule(df):
#     """
#     将DataFrame转换为绘图函数需要的嵌套字典结构。
#     """
#     stations = sorted(df['station'].unique().tolist())
#     # stations.reverse()
#     schedule_data = []
#     for train_name, train_group in df.groupby('train_id'):
#         train_info = {'train_id': train_name}
#         stops = train_group.sort_values(by='arrival_time').to_dict('records')
#         train_info['stops'] = stops
#         schedule_data.append(train_info)
#     return schedule_data, stations
#
#
# def plot_gantt_chart_on_ax(ax, schedule_data, stations, title):
#     """
#     在一个指定的matplotlib Axes对象(ax)上绘制甘特图。
#     """
#     if not schedule_data or not stations:
#         print(f"数据为空，无法为 '{title}' 绘制图表。")
#         ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
#         ax.set_title(title)
#         return
#
#     legend_added = False
#
#     for train in schedule_data:
#         stops = train['stops']
#         for i in range(len(stops)):
#             current_stop = stops[i]
#             arrival_time = current_stop['arrival_time_dt']
#
#             try:
#                 station_y_pos = stations.index(current_stop['station'])
#             except (ValueError, KeyError):
#                 continue
#
#             if i > 0:
#                 prev_stop = stops[i - 1]
#                 prev_departure_time = prev_stop['departure_time_dt']
#                 prev_station_y_pos = stations.index(prev_stop['station'])
#                 ax.plot([prev_departure_time, arrival_time],
#                         [prev_station_y_pos, station_y_pos],
#                         color='dimgray', linewidth=1.5, solid_capstyle='round')
#
#             # --- 1. 定义通用参数 ---
#
#             # 整体高度缩放因子：0.8表示条形图最高占Y轴一个刻度间隔的80%
#             bar_height_scale_factor = 5
#
#             # 单个条形的宽度：设置为15秒的时间增量
#             bar_width = timedelta(seconds=100)
#
#             # 条形图之间的水平偏移量：用于将4个条形图并排摆放，避免重叠
#             offset = timedelta(seconds=100)
#
#             labels = {"p_mod": "Passenger Modules", "p_board": "On-board Passengers",
#                       "f_mod": "Freight Modules", "f_board": "On-board Freight"}
#             if legend_added:
#                 labels = {k: "" for k in labels}
#
#             h_p_mod = (current_stop.get('num_p_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
#             ax.bar(arrival_time - 1.5 * offset, h_p_mod, width=bar_width, bottom=station_y_pos,
#                    color='lightblue', align='center', label=labels['p_mod'])
#
#             h_p_board = (current_stop.get('on_board_p', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
#             ax.bar(arrival_time + 0.5 * offset, h_p_board, width=bar_width, bottom=station_y_pos,
#                    color='blue', align='center', label=labels['p_board'])
#
#             h_f_mod = (current_stop.get('num_f_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
#             ax.bar(arrival_time - 0.5 * offset, h_f_mod, width=bar_width, bottom=station_y_pos,
#                    color='lightgreen', align='center', label=labels['f_mod'])
#
#             h_f_board = (current_stop.get('on_board_f', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
#             ax.bar(arrival_time + 1.5 * offset, h_f_board, width=bar_width, bottom=station_y_pos,
#                    color='green', align='center', label=labels['f_board'])
#
#             if not legend_added: legend_added = True
#
#     # --- 图表美化 ---
#     ax.set_yticks(range(len(stations)))
#     ax.set_yticklabels(stations, fontsize=12)
#     ax.set_ylabel('Station ID', fontsize=14)
#     ax.set_ylim(-0.5, len(stations) - 0.5)
#
#     ax.set_xlabel('Time', fontsize=14)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
#     # 旋转X轴标签
#     for label in ax.get_xticklabels():
#         label.set_rotation(45)
#         label.set_ha('right')
#
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=12)
#     ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)
#     ax.set_title(title, fontsize=16, pad=50)
#
#
# # --- 主程序入口 ---
# if __name__ == '__main__':
#     full_df = load_and_process_simulation_data(EXCEL_FILE_PATH, COLUMN_MAPPING)
#
#     if full_df is not None:
#         # 创建一个图窗和两个垂直排列的子图
#         # figsize的高度需要设置得更高，以容纳两个图
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 18))
#
#         # --- 处理和绘制 'up' 方向的图 ---
#         df_up = full_df[full_df['direction'] == 'up'].copy()
#         if not df_up.empty:
#             print("正在为 'up' 方向的数据生成图表...")
#             schedule_data_up, stations_up = transform_df_to_schedule(df_up)
#             plot_gantt_chart_on_ax(
#                 ax1,
#                 schedule_data_up,
#                 stations_up,
#                 title='Vehicle Schedule (Direction: Up)'
#             )
#         else:
#             print("未找到 'up' 方向的数据。")
#             ax1.text(0.5, 0.5, 'No data for "Up" direction', ha='center', va='center')
#             ax1.set_title('Vehicle Schedule (Direction: Up)')
#
#         # --- 处理和绘制 'down' 方向的图 ---
#         df_down = full_df[full_df['direction'] == 'down'].copy()
#         if not df_down.empty:
#             print("\n正在为 'down' 方向的数据生成图表...")
#             schedule_data_down, stations_down = transform_df_to_schedule(df_down)
#             plot_gantt_chart_on_ax(
#                 ax2,
#                 schedule_data_down,
#                 stations_down,
#                 title='Vehicle Schedule (Direction: Down)'
#             )
#         else:
#             print("未找到 'down' 方向的数据。")
#             ax2.text(0.5, 0.5, 'No data for "Down" direction', ha='center', va='center')
#             ax2.set_title('Vehicle Schedule (Direction: Down)')
#
#         # 调整整体布局以防重叠
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#         # 保存整个图窗到一个文件
#         output_filename = 'gantt_chart_combined.png'
#         plt.savefig(output_filename)
#         print(f"\n已将合并后的图表保存到 {output_filename}")
#
#         # 显示图表
#         plt.show()
#
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates
# # from datetime import datetime, timedelta
# #
# # # ==============================================================================
# # # --- 用户配置区 ---
# # # ==============================================================================
# #
# # # 1. 指定仿真程序输出的Excel文件路径
# # EXCEL_FILE_PATH = 'best_solution_20250827_183648/simulation_details.xlsx'  # <--- 确保这个文件名与您保存的文件名一致
# #
# # # 2. 列名映射 (已根据您的要求配置好)
# # COLUMN_MAPPING = {
# #     'train_id': '车辆ID',
# #     'direction': '方向',
# #     'station': '站点ID',
# #     'arrival_time': '到达时间',
# #     'num_p_modules': '调整后乘客模块',
# #     'num_f_modules': '调整后货物模块',
# #     'on_board_p': '在车乘客数量',
# #     'on_board_f': '在车货物数量',
# # }
# #
# # # 3. 绘图参数配置
# # # 假设的站点停留时间（分钟），用于计算画图所需的“出发时间”
# # STOP_DURATION_MINUTES = 1
# #
# # # Y轴条形图高度的缩放标准
# # # 用于模块数量的缩放，应略大于数据中的最大模块数
# # MAX_MODULES_FOR_SCALING = 10
# # # 用于乘客/货物数量的缩放，应略大于数据中的最大乘客/货物数量
# # MAX_PASSENGERS_FOR_SCALING = 100
# #
# #
# # # ==============================================================================
# # # --- 核心代码区 ---
# # # ==============================================================================
# #
# # def load_and_process_simulation_data(file_path, col_map):
# #     """
# #     从仿真输出的Excel文件中加载数据，并将其转换为绘图所需的数据结构。
# #     """
# #     try:
# #         df = pd.read_excel(file_path)
# #     except FileNotFoundError:
# #         print(f"错误：文件 '{file_path}' 未找到。请先运行仿真程序生成该文件。")
# #         return None, None
# #
# #     # 重命名列以方便处理
# #     inverted_map = {v: k for k, v in col_map.items() if v in df.columns}
# #     df.rename(columns=inverted_map, inplace=True)
# #
# #     # --- 数据转换 ---
# #     # 时间列单位是分钟，需要转换成 time 对象
# #     # 假设起始时间为 00:00
# #     start_time = datetime.strptime("00:00:00", "%H:%M:%S")
# #     df['arrival_time_dt'] = df['arrival_time'].apply(lambda x: start_time + timedelta(minutes=x))
# #
# #     # 计算画图用的出发时间
# #     df['departure_time_dt'] = df['arrival_time_dt'] + timedelta(minutes=STOP_DURATION_MINUTES)
# #
# #     # 获取所有车站的唯一列表并排序
# #     stations = sorted(df['station'].unique().tolist())
# #     stations.reverse()
# #
# #     # 将DataFrame转换为绘图函数需要的嵌套字典结构
# #     schedule_data = []
# #     for train_name, train_group in df.groupby('train_id'):
# #         train_info = {'train_id': train_name}
# #         # 按到达时间排序
# #         stops = train_group.sort_values(by='arrival_time').to_dict('records')
# #         train_info['stops'] = stops
# #         schedule_data.append(train_info)
# #
# #     return schedule_data, stations
# #
# #
# # def plot_gantt_chart_from_simulation(schedule_data, stations):
# #     """
# #     根据处理好的仿真数据绘制甘特图。
# #     """
# #     if not schedule_data or not stations:
# #         print("数据为空，无法绘制图表。")
# #         return
# #
# #     fig, ax = plt.subplots(figsize=(20, 10))
# #     legend_added = False
# #
# #     for train in schedule_data:
# #         stops = train['stops']
# #         for i in range(len(stops)):
# #             current_stop = stops[i]
# #
# #             arrival_time = current_stop['arrival_time_dt']
# #
# #             try:
# #                 station_y_pos = stations.index(current_stop['station'])
# #             except (ValueError, KeyError):
# #                 continue
# #
# #                 # 绘制列车运行线
# #             if i > 0:
# #                 prev_stop = stops[i - 1]
# #                 prev_departure_time = prev_stop['departure_time_dt']
# #                 prev_station_y_pos = stations.index(prev_stop['station'])
# #                 ax.plot([prev_departure_time, arrival_time],
# #                         [prev_station_y_pos, station_y_pos],
# #                         color='dimgray', linewidth=1.5, solid_capstyle='round')
# #
# #             # --- 绘制条形图 ---
# #             bar_height_scale_factor = 0.8
# #             bar_width_seconds = 15 * 60 / 100  # 将宽度设置为时间的某个比例，例如15秒
# #             bar_width_days = bar_width_seconds / (24 * 3600)  # Matplotlib内部使用天为单位处理时间
# #
# #             # 为了并排显示，给x坐标一个微小的偏移
# #             offset = timedelta(minutes=0.3)
# #
# #             # 定义标签，确保只在图例中显示一次
# #             labels = {
# #                 'p_mod': "Passenger Modules",
# #                 'f_mod': "Freight Modules",
# #                 'p_board': "On-board Passengers",
# #                 'f_board': "On-board Freight"
# #             }
# #             if legend_added:
# #                 labels = {k: "" for k in labels}
# #
# #             # 1. 乘客模块
# #             h_p_mod = (current_stop.get('num_p_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
# #             ax.bar(arrival_time - 1.5 * offset, h_p_mod, width=bar_width_days, bottom=station_y_pos,
# #                    color='lightblue', align='center', label=labels['p_mod'])
# #
# #             # 2. 货物模块
# #             h_f_mod = (current_stop.get('num_f_modules', 0) / MAX_MODULES_FOR_SCALING) * bar_height_scale_factor
# #             ax.bar(arrival_time - 0.5 * offset, h_f_mod, width=bar_width_days, bottom=station_y_pos,
# #                    color='lightgreen', align='center', label=labels['f_mod'])
# #
# #             # 3. 在车乘客
# #             h_p_board = (current_stop.get('on_board_p', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
# #             ax.bar(arrival_time + 0.5 * offset, h_p_board, width=bar_width_days, bottom=station_y_pos,
# #                    color='blue', align='center', label=labels['p_board'])
# #
# #             # 4. 在车货物
# #             h_f_board = (current_stop.get('on_board_f', 0) / MAX_PASSENGERS_FOR_SCALING) * bar_height_scale_factor
# #             ax.bar(arrival_time + 1.5 * offset, h_f_board, width=bar_width_days, bottom=station_y_pos,
# #                    color='green', align='center', label=labels['f_board'])
# #
# #             if not legend_added: legend_added = True
# #
# #     # --- 图表美化 ---
# #     ax.set_yticks(range(len(stations)))
# #     ax.set_yticklabels(stations, fontsize=12)
# #     ax.set_ylabel('Station ID', fontsize=14)
# #     ax.set_ylim(-0.5, len(stations) - 0.5)
# #
# #     ax.set_xlabel('Time', fontsize=14)
# #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# #     ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
# #     plt.xticks(rotation=45, ha="right")
# #
# #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=12)
# #     ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)
# #     ax.set_title('Vehicle Schedule Gantt Chart (from Simulation)', fontsize=16, pad=50)
# #
# #     plt.tight_layout(rect=[0, 0, 1, 0.93])
# #     plt.savefig("simulation_gantt_chart.png")
# #     plt.show()
# #
# #
# # def transform_df_to_schedule(df):
# #     """
# #     将DataFrame转换为绘图函数需要的嵌套字典结构。
# #     """
# #     # 获取所有车站的唯一列表并排序
# #     stations = sorted(df['station'].unique().tolist())
# #     stations.reverse()
# #
# #     schedule_data = []
# #     for train_name, train_group in df.groupby('train_id'):
# #         train_info = {'train_id': train_name}
# #         stops = train_group.sort_values(by='arrival_time').to_dict('records')
# #         train_info['stops'] = stops
# #         schedule_data.append(train_info)
# #
# #     return schedule_data, stations
# #
# # # # --- 主程序入口 ---
# # # if __name__ == '__main__':
# # #     # 1. 加载和处理仿真数据
# # #     schedule_data, stations_list = load_and_process_simulation_data(EXCEL_FILE_PATH, COLUMN_MAPPING)
# # #
# # #     # 2. 如果数据成功加载，则绘制图表
# # #     if schedule_data:
# # #         plot_gantt_chart_from_simulation(schedule_data, stations_list)
# # # --- 主程序入口 ---
# # if __name__ == '__main__':
# #     # 1. 加载并预处理整个数据文件
# #     full_df = load_and_process_simulation_data(EXCEL_FILE_PATH, COLUMN_MAPPING)
# #
# #     if full_df is not None:
# #         # 2. 按'direction'列的值进行筛选
# #         df_up = full_df[full_df['direction'] == 'up'].copy()
# #         df_down = full_df[full_df['direction'] == 'down'].copy()
# #
# #         # 3. 为 'up' 方向的数据绘图
# #         if not df_up.empty:
# #             print("正在为 'up' 方向的数据生成图表...")
# #             schedule_data_up, stations_up = transform_df_to_schedule(df_up)
# #             plot_gantt_chart_from_simulation(
# #                 schedule_data_up,
# #                 stations_up,
# #                 title='Vehicle Schedule (Direction: Up)',
# #                 output_filename='gantt_chart_up.png'
# #             )
# #         else:
# #             print("未找到 'up' 方向的数据。")
# #
# #         # 4. 为 'down' 方向的数据绘图
# #         if not df_down.empty:
# #             print("\n正在为 'down' 方向的数据生成图表...")
# #             schedule_data_down, stations_down = transform_df_to_schedule(df_down)
# #             plot_gantt_chart_from_simulation(
# #                 schedule_data_down,
# #                 stations_down,
# #                 title='Vehicle Schedule (Direction: Down)',
# #                 output_filename='gantt_chart_down.png'
# #             )
# #         else:
# #             print("未找到 'down' 方向的数据。")
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import matplotlib.dates as mdates
# # # from datetime import datetime
# # #
# # # # ==============================================================================
# # # # --- 用户配置区 ---
# # # # 您只需要修改这部分内容，以匹配您的Excel文件
# # # # ==============================================================================
# # #
# # # # 1. 指定您的Excel文件路径
# # # EXCEL_FILE_PATH = 'your_schedule_file.xlsx'  # <--- 修改这里: 您的文件名
# # #
# # # # 2. 列名映射：将程序需要的标准列名映射到您Excel文件中的实际列名
# # # #    请将右侧的 '...' 替换为您表格中对应的列标题
# # # COLUMN_MAPPING = {
# # #     # 标准名          # 您Excel中的实际列名
# # #     'train_id': '车辆ID',  # <--- 修改这里: 例如 '车次' 或 '车辆ID'
# # #     'station': '站点ID',  # <--- 修改这里: 例如 '车站'
# # #     'arrival_time': '到达时间',  # <--- 修改这里: 例如 '抵达时刻'
# # #     # 'departure_time': '出发时间',  # <--- 修改这里: 例如 '发车时刻'
# # #     'num_p_modules': '调整后乘客模块',  # <--- 修改这里: 对应“Capacity for passengers”
# # #     'num_f_modules': '调整后货物模块',  # <--- 修改这里: 对应“Capacity for passengers”
# # #     'on_board_p': '上车后在车乘客数量',  # <--- 修改这里: 对应“Amount of passengers on the train”
# # #     'on_board_f': '上车后在车货物数量',  # <--- 修改这里: 对应“Amount of passengers on the train”
# # #     # 'arriving': '到达乘客及货物数量',  # <--- 修改这里: 对应“Amount of arrival passengers”，如果您的文件中没有此列，可以忽略或设为None
# # # }
# # #
# # # # 3. 绘图参数配置
# # # # 用于归一化的最大值，决定了条形图的高度。应略大于您数据中的最大容量/乘客数。
# # # MAX_VALUE_FOR_SCALING = 1200  # <--- 修改这里: 根据您的数据调整
# # #
# # # # Y轴（车站）的显示顺序。'auto' 会自动从数据中获取顺序。
# # # # 如果需要手动指定顺序，请使用列表，例如: ['站点A', '站点B', '站点C']
# # # STATION_ORDER = 'auto'
# # #
# # #
# # # # ==============================================================================
# # # # --- 核心代码区 ---
# # # # 通常您不需要修改以下代码
# # # # ==============================================================================
# # #
# # # def load_and_process_data(file_path, col_map):
# # #     """
# # #     从Excel文件中加载数据，并将其转换为绘图所需的数据结构。
# # #     """
# # #     try:
# # #         df = pd.read_excel(file_path)
# # #     except FileNotFoundError:
# # #         print(f"错误：文件 '{file_path}' 未找到。请检查文件名和路径是否正确。")
# # #         return None, None
# # #
# # #     # 为了方便处理，将用户提供的列名重命名为标准列名
# # #     # 检查必要的列是否存在
# # #     for key, val in col_map.items():
# # #         if val not in df.columns and val is not None:
# # #             print(f"错误：在Excel文件中找不到列 '{val}'。请检查COLUMN_MAPPING配置。")
# # #             return None, None
# # #
# # #     # 执行重命名
# # #     inverted_map = {v: k for k, v in col_map.items() if v is not None}
# # #     df.rename(columns=inverted_map, inplace=True)
# # #
# # #     # --- 数据转换 ---
# # #     # 将时间列转换为datetime对象，如果已经是datetime格式则忽略错误
# # #     for col in ['arrival_time', 'departure_time']:
# # #         # 先判断是否是时间类型，如果不是则进行转换
# # #         if df[col].dtype != 'datetime64[ns]':
# # #             df[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.time
# # #
# # #     # 获取所有车站的唯一列表
# # #     if STATION_ORDER == 'auto':
# # #         # 自动按出现的顺序（或排序）获取车站
# # #         stations = df['station'].unique().tolist()
# # #     else:
# # #         stations = STATION_ORDER
# # #     stations.reverse()  # 反转以匹配示例图（起点在图的底部）
# # #
# # #     # 将DataFrame转换为绘图函数需要的嵌套字典结构
# # #     schedule_data = []
# # #     for train_name, train_group in df.groupby('train_id'):
# # #         train_info = {'train_id': train_name}
# # #         stops = train_group.sort_values(by='arrival_time').to_dict('records')
# # #         train_info['stops'] = stops
# # #         schedule_data.append(train_info)
# # #
# # #     return schedule_data, stations
# # #
# # #
# # # def plot_gantt_chart(schedule_data, stations):
# # #     """
# # #     根据处理好的数据绘制甘特图。
# # #     """
# # #     if not schedule_data or not stations:
# # #         print("数据为空，无法绘制图表。")
# # #         return
# # #
# # #     fig, ax = plt.subplots(figsize=(18, 10))
# # #     legend_added = False
# # #
# # #     # 辅助函数：将time对象转换为datetime对象用于绘图
# # #     def to_datetime(t):
# # #         return datetime.combine(datetime.today(), t)
# # #
# # #     for train in schedule_data:
# # #         stops = train['stops']
# # #         for i in range(len(stops)):
# # #             current_stop = stops[i]
# # #
# # #             arrival_time_dt = to_datetime(current_stop['arrival_time'])
# # #
# # #             try:
# # #                 station_y_pos = stations.index(current_stop['station'])
# # #             except ValueError:
# # #                 continue  # 如果站点不在列表中，则跳过
# # #
# # #             # 绘制列车运行线
# # #             if i > 0:
# # #                 prev_stop = stops[i - 1]
# # #                 prev_departure_time_dt = to_datetime(prev_stop['departure_time'])
# # #                 prev_station_y_pos = stations.index(prev_stop['station'])
# # #                 ax.plot([prev_departure_time_dt, arrival_time_dt],
# # #                         [prev_station_y_pos, station_y_pos],
# # #                         color='dimgray', linewidth=1.5, solid_capstyle='round')
# # #
# # #             # 绘制条形图
# # #             bar_height_scale_factor = 0.8
# # #             bar_width = timedelta(seconds=20)
# # #             offset = timedelta(seconds=30)
# # #
# # #             # 标签只添加一次
# # #             label_cap = "Capacity" if not legend_added else ""
# # #             label_arr = "Arriving" if not legend_added else ""
# # #             label_onb = "On-board" if not legend_added else ""
# # #
# # #             # 绘制容量条
# # #             if 'capacity' in current_stop and pd.notna(current_stop['capacity']):
# # #                 h = (current_stop['capacity'] / MAX_VALUE_FOR_SCALING) * bar_height_scale_factor
# # #                 ax.bar(arrival_time_dt, h, width=bar_width, bottom=station_y_pos,
# # #                        color='orange', alpha=0.9, align='center', label=label_cap)
# # #
# # #             # 绘制到达乘客条
# # #             if 'arriving' in current_stop and pd.notna(current_stop['arriving']):
# # #                 h = (current_stop['arriving'] / MAX_VALUE_FOR_SCALING) * bar_height_scale_factor
# # #                 ax.bar(arrival_time_dt + offset, h, width=bar_width, bottom=station_y_pos,
# # #                        color='steelblue', alpha=0.9, align='center', label=label_arr)
# # #
# # #             # 绘制在车乘客条
# # #             if 'on_board' in current_stop and pd.notna(current_stop['on_board']):
# # #                 h = (current_stop['on_board'] / MAX_VALUE_FOR_SCALING) * bar_height_scale_factor
# # #                 ax.bar(arrival_time_dt + 2 * offset, h, width=bar_width, bottom=station_y_pos,
# # #                        color='firebrick', alpha=0.9, align='center', label=label_onb)
# # #
# # #             if not legend_added: legend_added = True
# # #
# # #     # --- 图表美化 ---
# # #     ax.set_yticks(range(len(stations)))
# # #     ax.set_yticklabels(stations, fontsize=12)
# # #     ax.set_ylabel('Station', fontsize=14)
# # #     ax.set_ylim(-0.5, len(stations) - 0.5)
# # #
# # #     ax.set_xlabel('Time', fontsize=14)
# # #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# # #     ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
# # #     plt.xticks(rotation=45, ha="right")
# # #
# # #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=12)
# # #     ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)
# # #     ax.set_title('Train Schedule Gantt Chart', fontsize=16, pad=40)
# # #
# # #     plt.tight_layout(rect=[0, 0, 1, 0.95])
# # #     plt.show()
# # #
# # #
# # # # --- 主程序入口 ---
# # # if __name__ == '__main__':
# # #     # 1. 加载和处理数据
# # #     schedule_data, stations_list = load_and_process_data(EXCEL_FILE_PATH, COLUMN_MAPPING)
# # #
# # #     # 2. 如果数据成功加载，则绘制图表
# # #     if schedule_data:
# # #         plot_gantt_chart(schedule_data, stations_list)