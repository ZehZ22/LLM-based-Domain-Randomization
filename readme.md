# USV路径跟踪强化学习项目

## 项目结构

```
代码/
├── disturbances/                        # 环境干扰函数模块
│   ├── current.py            
│   ├── isherwood72.py             
│   ├── wave.py                  
│   └── wind.py           
│      
├── data/                       # 数据
│   ├── domain_ranges.csv                
│   ├── dr_log.csv  
│   ├── loss_history.json
│   ├── reward_history.json                
│   └── eval_metrics.json                
├── dynamic_models/                #船舶动力学模型        
│   ├── mariner_wind.py            
│   ├── mariner2.py              
├── plots/                         #可视化图表     
│   ├── total_reward.png 
│   ├── policy_loss.png                      
│   └── value_loss.png 
├── policys/                             #RL生成的策略模型    
│   ├── model_action.pth             
│   ├── model_value1.pth                
│   └── model_value2.pth
├── simulator/                           #仿真器    
│   ├── sim2.py 
│   ├── simulator_training.py                    
│   └── simulator_pseudo.py
├── utils/                               
│   ├── comparsion of human_design and LLM.py
│   ├── dr_config.py      
│   ├── evaluate.py               
│   └── SAC_training.py 
├── LLMtrainer.py                     # 主入口
├── env_params.py   
├── ship_params.py   
└── readme.md
```
