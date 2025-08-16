import random

def generate_dual_direction_individual_combined(
    num_vehicles,
    max_modules,
):
    def generate_one_direction(direction):

        vid_offset = 0 if direction == "up" else 100
        vehicle_dispatch = {}
        current_time = 0

        # 生成车头时距
        for vid in range(num_vehicles):
            global_vid = vid + vid_offset
            headway = random.randint(3, 20)
            vehicle_dispatch[global_vid] = {"headway": headway, "arrival_time": current_time}
            current_time += headway

        # 生成发车模块构成
        vehicle_initial_allocation = {}
        for vid in range(num_vehicles):
            global_vid = vid + vid_offset
            total = random.randint(1, max_modules)
            p = random.randint(0, total)
            f = total - p
            vehicle_initial_allocation[global_vid] = {"passenger_modules": p, "freight_modules": f}

        return {
            "vehicle_dispatch": vehicle_dispatch,  # 发车时间
            "initial_allocation": vehicle_initial_allocation,  # 初始模块配置
        }

    individual_up = generate_one_direction("up")
    individual_down = generate_one_direction("down")

    individual = {}
    individual["up"] = individual_up
    individual["down"] = individual_down
    # individual["initial_module_pool"] = {
    #     "up_start": random.randint(30, 50),
    #     "down_start": random.randint(20, 40)
    # }

    return individual

# {'up': {'vehicle_dispatch':
#             {0: {'headway': 16, 'arrival_time': 0},
#              1: {'headway': 18, 'arrival_time': 16},
#              2: {'headway': 7, 'arrival_time': 34}},
#         'initial_allocation': {
#             0: {'passenger_modules': 2, 'freight_modules': 2},
#             1: {'passenger_modules': 1, 'freight_modules': 0},
#             2: {'passenger_modules': 3, 'freight_modules': 1}}},
# 'down': {'vehicle_dispatch':
#             {100: {'headway': 3, 'arrival_time': 0},
#              101: {'headway': 12, 'arrival_time': 3},
#              102: {'headway': 17, 'arrival_time': 15}},
#         'initial_allocation': {
#             100: {'passenger_modules': 0, 'freight_modules': 1},
#             101: {'passenger_modules': 1, 'freight_modules': 0},
#             102: {'passenger_modules': 0, 'freight_modules': 1}}}}