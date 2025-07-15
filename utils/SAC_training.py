# SAC_training.py
import random, numpy as np, torch, json, os
from matplotlib import pyplot as plt
from simulator.simulator_training import USVStateR,generate_path
from simulator.sim2 import marinerwind
from utils.evaluate import evaluate_trajectory

plt.switch_backend('agg')
angle_deg, length, interval = 30, 1000, 1
waypoints = generate_path(angle_deg, length, interval)

# 全局模型和参数
model_action = model_value1 = model_value2 = model_value_next1 = model_value_next2 = alpha = None
datas = []

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class ModelAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.ReLU())
        self.fc_mu = torch.nn.Linear(128, 1)
        self.fc_std = torch.nn.Sequential(torch.nn.Linear(128, 1), torch.nn.Softplus())
        self.apply(init_weights)

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state).clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample)
        entropy = log_prob - (1 - action ** 2 + 1e-7).log()
        return action * 35, -entropy

class ModelValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(4, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1))
        self.apply(init_weights)

    def forward(self, state, action):
        state = torch.cat([state, action], dim=1)
        return self.sequential(state)

def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    with torch.no_grad():
        action, _ = model_action(state)
    return action.item()

def get_sample():
    if len(datas) < 64:
        return None, None, None, None, None
    samples = random.sample(datas, 64)
    return (
        torch.FloatTensor([i[0] for i in samples]).reshape(-1, 3),
        torch.FloatTensor([i[1] for i in samples]).reshape(-1, 1),
        torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1),
        torch.FloatTensor([i[3] for i in samples]).reshape(-1, 3),
        torch.LongTensor([i[4] for i in samples]).reshape(-1, 1),
    )

def get_target(next_state, reward, over):
    action, entropy = model_action(next_state)
    target1 = model_value_next1(next_state, action)
    target2 = model_value_next2(next_state, action)
    target = torch.min(target1, target2) + alpha.exp() * entropy
    return reward + 0.99 * (1 - over) * target

def get_loss_action(state):
    action, entropy = model_action(state)
    value = torch.min(model_value1(state, action), model_value2(state, action))
    return (-alpha.exp() * entropy - value).mean(), entropy

def soft_update(model, model_next):
    for param, param_next in zip(model.parameters(), model_next.parameters()):
        param_next.data.copy_(param_next.data * 0.995 + param.data * 0.005)

def update_data(env, episode_id=None):
    try:
        state, over, total_reward, step_count = env.reset(), False, 0, 0
        while not over and step_count < 2000:
            action = get_action(state)
            next_state, reward, over, _ = env.step([action])

        # === 打印 x, y, distance_to_target, current_index ===
            x, y = env.x[3], env.x[4]
            target_x, target_y = env.waypoints[-1][0], env.waypoints[-1][1]
            distance_to_target = np.linalg.norm([x - target_x, y - target_y])
            current_index = env.current_index
            print(f"x={x:.2f}, y={y:.2f}, distance_to_target={distance_to_target:.2f}, current_index={current_index}")

            # 检查数值合法性
            if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e10):
                print("⚠️ 环境状态异常，跳过当前 episode")
                return None

            datas.append((state, action, reward, next_state, over))
            state = next_state
            total_reward += reward
            step_count += 1

        # 控制样本池大小
        while len(datas) > 100000:
            datas.pop(0)

        if episode_id is not None:
            print(f"✅ Episode {episode_id} 用时步数：{step_count}, 总奖励：{total_reward:.2f}")
            print(f"📦 当前 replay buffer 大小：{len(datas)}")

        return total_reward
    except Exception as e:
        print(f"❌ 仿真错误: {e}")
        return None


def init_training():
    global model_action, model_value1, model_value2, model_value_next1, model_value_next2, alpha, datas
    model_action = ModelAction()
    model_value1 = ModelValue()
    model_value2 = ModelValue()
    model_value_next1 = ModelValue()
    model_value_next2 = ModelValue()
    model_value_next1.load_state_dict(model_value1.state_dict())
    model_value_next2.load_state_dict(model_value2.state_dict())
    alpha = torch.tensor(np.log(0.01), requires_grad=True)
    datas = []

    global optimizer_action, optimizer_value1, optimizer_value2, optimizer_alpha, loss_fn
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=1e-4)
    optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=1e-3)
    optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=1e-3)
    optimizer_alpha = torch.optim.Adam([alpha], lr=1e-5)
    loss_fn = torch.nn.MSELoss()

def train_one_episode(dr_params, episode_id):
    print(f"\n====== Episode {episode_id} 开始，扰动参数：{dr_params} ======")
    env = USVStateR(
        waypoints=waypoints,
        current_index=1,
        x=np.array([0.0, 0.0, 0.0, 300.0, 0.0, np.radians(20), 0.0], dtype=np.float32),
        ui=0.0,
        model_func=marinerwind,
        wind_speed=dr_params.get("wind_speed", 3.0),
        wind_direction=dr_params.get("wind_direction", 90.0),
        current_speed=dr_params.get("current_speed", 1.2),
        current_direction=dr_params.get("current_direction", 45.0),
        wave_height=dr_params.get("wave_height", 3.5),
        beta=np.radians(dr_params.get("wave_direction", 90.0)),
    )

    total_reward = update_data(env, episode_id=episode_id)
    if total_reward is None:
        print("⚠️ 当前仿真失败，跳过训练")
        return 0, {"status": "failed"}
    
    # === 新增：loss统计 ===
    value1_losses = []
    policy_losses = []

    for _ in range(2000):
        sample = get_sample()
        if sample[0] is None: continue
        state, action, reward, next_state, over = sample
        reward = (reward + 8) / 8
        target = get_target(next_state, reward, over).detach()
        value1, value2 = model_value1(state, action), model_value2(state, action)
        optimizer_value1.zero_grad()
        loss1 = loss_fn(value1, target)
        loss1.backward()
        optimizer_value1.step()

        optimizer_value2.zero_grad()
        loss2 = loss_fn(value2, target)
        loss2.backward()
        optimizer_value2.step()

        loss_action, entropy = get_loss_action(state)
        optimizer_action.zero_grad()
        loss_action.backward()
        optimizer_action.step()

        loss_alpha = (entropy + 1).detach() * alpha.exp()
        optimizer_alpha.zero_grad()
        loss_alpha.mean().backward()
        optimizer_alpha.step()

        soft_update(model_value1, model_value_next1)
        soft_update(model_value2, model_value_next2)

        # === 新增：记录loss ===
        value1_losses.append(loss1.item())
        policy_losses.append(loss_action.item())

    # === 新增：计算均值 ===
    value_loss_avg = float(np.mean(value1_losses)) if value1_losses else 0
    policy_loss_avg = float(np.mean(policy_losses)) if policy_losses else 0

    metrics, _ = evaluate_trajectory(env, model_action)
    metrics["value_loss"] = value_loss_avg
    metrics["policy_loss"] = policy_loss_avg
    with open("eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return total_reward, metrics

def train(dr_sequence):
    init_training()
    for episode_id, dr_param in enumerate(dr_sequence, start=1):
        train_one_episode(dr_param, episode_id)

if __name__ == "__main__":
    dr_list_json = os.environ.get("DR_LIST_JSON")
    dr_dict_json = os.environ.get("DR_DICT_JSON")
    if dr_list_json:
        try:
            dr_sequence = json.loads(dr_list_json)
            train(dr_sequence)
        except Exception as e:
            print(f"❌ 无法解析 DR_LIST_JSON: {e}")
    elif dr_dict_json:
        try:
            dr_param = json.loads(dr_dict_json)
            init_training()
            train_one_episode(dr_param, episode_id=999)
        except Exception as e:
            print(f"❌ 无法解析 DR_DICT_JSON: {e}")
    else:
        print("❌ 未提供扰动参数")