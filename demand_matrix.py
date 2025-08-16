# 初始化需求矩阵
def initialize_demand_matrices(num_vehicles, station_range, num_timestamps, passenger_demand, freight_demand, MAX_HEADWAY, up_station_count, margin, parameters):
    MAX_TIME = num_vehicles * MAX_HEADWAY + up_station_count * parameters["t_s_s1"] + margin
    num_timestamps = max(num_timestamps, MAX_TIME)

    a_matrix_p = {s: {s_p: {t: 0 for t in range(num_timestamps)} for s_p in station_range if s_p > s} for s in station_range}
    a_matrix_f = {s: {s_p: {t: 0 for t in range(num_timestamps)} for s_p in station_range if s_p > s} for s in station_range}

    for passenger in passenger_demand:
        a_matrix_p[passenger["origin"]][passenger["destination"]][passenger["arrival_time"]] += passenger["num_passengers"]

    for freight in freight_demand:
        a_matrix_f[freight["origin"]][freight["destination"]][freight["arrival_time"]] += freight["volume"]

    return a_matrix_p, a_matrix_f