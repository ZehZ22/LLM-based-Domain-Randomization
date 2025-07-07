# LLM-based-Domain-Randomization
通过大语言模型（基于deepseek）自动反馈域随机化参数分布，实现不依赖人工调参的领域随机化，从而缩小reality gap。
LLMtrainer：将训练分为两个阶段，第一阶段为冷启动模式使用预定的域随机化分布，可设置训练轮数确保模型收敛；第二阶段由LLM给出域随机分布，每轮进行更新，直到训练结束
