import numpy as np
import json
import matplotlib.pyplot as plt
import torch

def evaluate_trajectory(env, model_action, max_steps=1000):
    state = env.reset()
    trajectory = []
    headings = []
    rudder_angles = []
    cross_track_errors = []
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        state_tensor = torch.FloatTensor(state).reshape(1, 3)
        action, _ = model_action(state_tensor)
        next_state, _, done, _ = env.step([action.item()])

        full_state = env.get_full_state()
        heading_error_deg = np.degrees(full_state[0])  # ✅ 使用 heading_error 而非 psi
        cross_track_error = full_state[1]
        rudder_angle = full_state[2]
        x, y = full_state[3], full_state[4]

        trajectory.append((x, y))
        headings.append(np.abs(heading_error_deg))  # ✅ 绝对值，避免负误差互相抵消
        rudder_angles.append(rudder_angle)
        cross_track_errors.append(cross_track_error)

        state = next_state
        step_count += 1

    trajectory = np.array(trajectory)
    headings = np.array(headings)
    rudder_angles = np.array(rudder_angles)
    cross_track_errors = np.array(cross_track_errors)

    metrics = {
        "Mean_Cross_Track_Error": float(np.mean(np.abs(cross_track_errors))),
        "Max_Cross_Track_Error": float(np.max(np.abs(cross_track_errors))),
        "Mean_Heading_Error": float(np.mean(headings)),  # ✅ 已经是绝对值
        "Mean_Absolute_Rudder_Change": float(np.mean(np.abs(np.diff(rudder_angles))))
    }

    save_metrics(metrics)  # 保存指标
    return metrics, trajectory

def save_metrics(metrics, filename="data/eval_metrics.json"):
    # 确保data目录存在
    import os
    os.makedirs("data", exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_trajectory(trajectory, waypoints):
    trajectory = np.array(trajectory)
    waypoints = np.array(waypoints)

    plt.figure(figsize=(10, 8))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'k--', label='Target Path')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', label='Agent Path')
    plt.title('Trajectory Evaluation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectory_evaluation.png")
    plt.close()
