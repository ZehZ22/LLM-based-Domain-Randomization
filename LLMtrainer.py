import os
import subprocess
import json
import time
import csv
import torch
from openai import OpenAI
from utils.dr_config import randomize_environment_params

# 初始化 LLM 客户端（DeepSeek）
client = OpenAI(
    api_key="你的API",
    base_url="https://api.deepseek.com/v1"
)

TOTAL_EPISODES = 400
COLD_START_EPISODES = 300

INITIAL_PROMPT = """
你是一个强化学习专家，任务是调节风、流、浪扰动的强度，以提升USV路径跟踪策略的泛化能力。
请你输出一个包含扰动参数范围的 Python 字典，用于USV路径跟踪任务的域随机化训练。
每个值应是一个长度为2的列表，例如：
{
    "wind_speed_range": [0.0, 10.0],
    "wind_direction_range": [0, 360],
    "current_speed_range": [0.0, 5.0],
    "current_direction_range": [0, 360],
    "wave_height_range": [0.0, 5.0],
    "wave_direction_range": [0, 360]
}
"""


def ask_llm_for_dr_config(eval_result, episode_id):
    prompt = f"""
你是一个海事智能体训练专家，当前需要根据第 {episode_id} 轮训练的评估结果，调节下一轮的风、流、浪扰动范围：

评估结果如下：
{eval_result}

请你输出符合以下格式的 JSON 字典（不要加注释、文字说明或代码块）：
{{
  "wind_speed_range": [min, max],
  "wind_direction_range": [min, max],
  "current_speed_range": [min, max],
  "current_direction_range": [min, max],
  "wave_height_range": [min, max],
  "wave_direction_range": [min, max]
}}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个强化学习专家"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def extract_dict(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return eval(content)
        except Exception:
            return None


def save_dr_csv(dr_dict, episode_id):
    keys = list(dr_dict.keys())
    # 确保data目录存在
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists("data/domain_ranges.csv")
    with open("data/domain_ranges.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["episode"] + keys)
        if not file_exists:
            writer.writeheader()
        row = {"episode": episode_id}
        row.update(dr_dict)
        writer.writerow(row)


def load_eval_metrics():
    try:
        with open("data/eval_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def main():
    import utils.SAC_training  # 动态引入，避免提前初始化模型
    utils.SAC_training.init_training()

    reward_history = []
    value_loss_avg_history = []
    policy_loss_avg_history = []
    import utils.SAC_training  # 动态引入，避免提前初始化模型
    SAC_training = utils.SAC_training
    for episode in range(1, TOTAL_EPISODES + 1):
        print(f"\n==== Episode {episode} ====")

        # 获取扰动参数
        if episode <= COLD_START_EPISODES:
            dr_param = randomize_environment_params(round_id=episode)
        else:
            print(f"请求 LLM 生成第 {episode} 轮的扰动范围...")
            eval_metrics = load_eval_metrics()
            llm_output = ask_llm_for_dr_config(json.dumps(eval_metrics, indent=2, ensure_ascii=False), episode)
            dynamic_range = extract_dict(llm_output)
            if dynamic_range is None:
                print("⚠️ 无法解析LLM返回内容，跳过本轮")
                continue
            dr_param = randomize_environment_params(dynamic_ranges=dynamic_range, round_id=episode)

        # 保存当前扰动参数
        save_dr_csv(dr_param, episode)

        # 单轮训练
        reward, metrics = SAC_training.train_one_episode(dr_param, episode)
        reward_history.append(reward)
        value_loss_avg_history.append(metrics.get("value_loss", 0))
        policy_loss_avg_history.append(metrics.get("policy_loss", 0))

        # 保存 summary
        summary = {
            "episode": episode,
            "dr_param": dr_param,
            "eval_metrics": metrics,
            "total_reward": reward,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        # 确保data目录存在
        os.makedirs("data", exist_ok=True)
        with open(f"data/round_summary_{episode}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        time.sleep(1)

    # === 模型保存与绘图 ===
    import matplotlib.pyplot as plt
    os.makedirs("plots", exist_ok=True)
    x_epochs = list(range(1, len(reward_history) + 1))

    plt.figure()
    plt.plot(x_epochs, reward_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/total_reward.png")
    plt.close()

    plt.figure()
    plt.plot(x_epochs, value_loss_avg_history, label='Avg Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Value Loss per Episode')
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/value_loss.png")
    plt.close()

    plt.figure()
    plt.plot(x_epochs, policy_loss_avg_history, label='Avg Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy Loss per Episode')
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/policy_loss.png")
    plt.close()

    # 确保data目录存在
    os.makedirs("data", exist_ok=True)
    with open("data/reward_history.json", "w", encoding="utf-8") as f:
        json.dump(reward_history, f, indent=2)
    with open("data/loss_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "value_loss": value_loss_avg_history,
            "policy_loss": policy_loss_avg_history
        }, f, indent=2)

    torch.save(SAC_training.model_action.state_dict(), 'policys/model_action.pth')
    torch.save(SAC_training.model_value1.state_dict(), 'policys/model_value1.pth')
    torch.save(SAC_training.model_value2.state_dict(), 'policys/model_value2.pth')


if __name__ == "__main__":
    main()
