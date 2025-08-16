import pandas as pd


def reconstruct_schedule_dataframe(individual, parameters, direction):
    vehicle_module_adjustments = individual[direction]["module_adjustments"]
    vehicle_dispatch = individual[direction]["vehicle_dispatch"]
    initial_allocation = individual[direction]["initial_allocation"]

    df_schedule = []

    # 初始化站点库存
    station_module_stock = {sid: {"modules": 0} for sid in range(parameters["up_station_count"] * 2)}

    time_per_station = parameters.get("t_s_s1", 1)  # 站点间行驶时间（配置）

    for vid in vehicle_module_adjustments:
        vid_int = int(vid)
        dispatch_time = vehicle_dispatch[vid_int]["arrival_time"]

        current_p_modules = initial_allocation[vid]["passenger_modules"]
        current_f_modules = initial_allocation[vid]["freight_modules"]

        # 计算每个站点的到达时间
        arrival_times = {}
        arrival_time = dispatch_time
        arrival_times[0] = arrival_time
        for station_id in sorted(vehicle_module_adjustments[vid].keys()):
            if station_id == 0:
                continue  # 第一站的到达时间就是发车时间
            arrival_time += time_per_station
            arrival_times[station_id] = arrival_time

        # 保证站点按顺序处理
        for station_id in sorted(vehicle_module_adjustments[vid].keys()):

            station_module_stock_before = station_module_stock[station_id]["modules"]

            adjustment = vehicle_module_adjustments[vid][station_id]
            delta_p = adjustment["delta_p"]
            delta_f = adjustment["delta_f"]

            current_p_modules += delta_p
            current_f_modules += delta_f

            station_module_stock[station_id]["modules"] -= (delta_p + delta_f)
            station_module_stock_after = station_module_stock[station_id]["modules"]

            df_schedule.append({
                "车辆ID": vid_int,
                "站点ID": station_id,
                "到达时间": arrival_times.get(station_id, dispatch_time),
                "乘客模块": current_p_modules,
                "货物模块": current_f_modules,
                "总模块数量": current_p_modules + current_f_modules,
                "乘客增量": delta_p,
                "货物增量": delta_f,
                "站点进站前库存": station_module_stock_before,
                "站点离站后库存": station_module_stock_after
            })

    return pd.DataFrame(df_schedule)