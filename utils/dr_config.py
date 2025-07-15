import random
import csv
import os

# 创建一个随机扰动参数函数，接收外部传入的 domain_randomization_ranges
def randomize_environment_params(dynamic_ranges=None, round_id=None, save_path='data/dr_log.csv'):
    # 是否处于冷启动阶段（轮数前100轮）
    is_cold_start = (round_id is not None and round_id <= 300)

    # 冷启动阶段固定可选方向角度
    discrete_angles = [0.0, 45.0, 90.0, 135.0]

    # 如果未传入范围，则使用默认静态范围
    default_ranges = {
        'wind_speed_range': [2.8, 3.4],
        'wind_direction_range': [0.0, 360.0],
        'current_speed_range': [1.1, 1.4],
        'current_direction_range': [0.0, 360.0],
        'wave_height_range': [3.2, 3.8],
        'wave_direction_range': [0.0, 360.0],
    }

    ranges = dynamic_ranges if dynamic_ranges else default_ranges

    # 使用冷启动逻辑时，设置为指定范围和离散角度
    if is_cold_start:
        dr_params = {
            'wind_speed': round(random.uniform(2.6, 3.6), 2),
            'wind_direction': random.choice(discrete_angles),
            'current_speed': round(random.uniform(1.0, 1.5), 2),
            'current_direction': random.choice(discrete_angles),
            'wave_height': round(random.uniform(3.0, 4.0), 2),
            'wave_direction': random.choice(discrete_angles),
        }
    else:
        # 正常逻辑（使用传入或默认范围内的连续值）
        dr_params = {
            'wind_speed': round(random.uniform(*ranges['wind_speed_range']), 2),
            'wind_direction': round(random.uniform(*ranges['wind_direction_range']), 2),
            'current_speed': round(random.uniform(*ranges['current_speed_range']), 2),
            'current_direction': round(random.uniform(*ranges['current_direction_range']), 2),
            'wave_height': round(random.uniform(*ranges['wave_height_range']), 2),
            'wave_direction': round(random.uniform(*ranges['wave_direction_range']), 2),
        }

    # 保存当前轮扰动参数到CSV
    if round_id is not None:
        # 确保data目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_exists = os.path.isfile(save_path)
        with open(save_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['round'] + list(dr_params.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({'round': round_id, **dr_params})

    return dr_params
