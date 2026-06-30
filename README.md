# RL Dino Mini Game

这是一个从强化学习角度设计的简化版小恐龙小游戏。

项目重点不是先做复杂画面，而是先把游戏抽象成强化学习常见结构：

- `reset()`：重置一局游戏
- `step(action)`：执行一个动作，推进一帧
- `observation`：返回当前状态
- `reward`：返回奖励
- `done`：表示游戏是否结束

## 文件说明

- `dino_env.py`：核心游戏环境，不依赖 PyGame，后续训练 AI 主要用它
- `game.py`：PyGame 可视化入口，可手动玩，也可看规则 AI 或随机 AI 玩
- `q_learning.py`：最小 Q-learning 训练脚本，会生成 `q_table.json`
- `requirements.txt`：依赖列表

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 运行

手动玩：

```bash
python3 game.py
```

空格或方向上键跳跃，`R` 重新开始。

规则 AI 自动玩：

```bash
python3 game.py --agent rule
```

随机 AI 自动玩：

```bash
python3 game.py --agent random
```

训练 Q-learning AI：

```bash
python3 q_learning.py --episodes 3000
```

训练后用 Q-learning AI 自动玩：

```bash
python3 game.py --agent q
```

## 强化学习接口

动作空间：

- `0`：不动
- `1`：跳跃

状态 observation：

```text
[
  distance_to_obstacle,
  obstacle_width,
  obstacle_speed,
  dino_y,
  dino_velocity_y,
  on_ground
]
```

奖励设计：

- 每活一帧：`+0.1`
- 成功越过障碍：`+5.0`
- 撞到障碍：`-10.0`

之后可以在 `dino_env.py` 的基础上接 Q-learning、DQN 或 Stable-Baselines3。

## Q-learning 版本说明

`q_learning.py` 做了三件事：

- 把连续状态离散成有限数量的桶
- 用 Q 表记录每个状态下两个动作的价值
- 使用 epsilon-greedy 让 AI 在“探索”和“利用已有经验”之间切换

训练轮数越多，`q_table.json` 里的状态经验越多。这个版本适合理解强化学习基本流程，后面再升级成 DQN 会更自然。
