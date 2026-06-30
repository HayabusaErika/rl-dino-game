from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dino_env import DinoEnv


ACTION_COUNT = 2
DEFAULT_Q_TABLE = Path("q_table.json")


def discretize(observation: list[float]) -> tuple[int, int, int, int, int, int]:
    distance, obstacle_width, obstacle_speed, dino_y, dino_velocity_y, on_ground = observation #拆变量
    return (
         min(15, int(distance * 32)),        # 距离 → 16 个档
        min(4, int(obstacle_width * 5)),     # 宽度 → 5 个档
        min(7, int(obstacle_speed * 8)),     # 速度 → 8 个档
        min(8, int(dino_y * 9)),             # 恐龙高度 → 9 个档
        max(-5, min(5, int(dino_velocity_y * 6))),  # 恐龙速度 → 11 个档
        int(on_ground),                      # 是否在地面 → 0 或 1
    )


def state_key(state: tuple[int, int, int, int, int, int]) -> str: #制造元组（不是邦多利！）
    return ",".join(str(value) for value in state) #数值之间需要用逗号隔开



def get_q_values(q_table: dict[str, list[float]], state: tuple[int, int, int, int, int, int]) -> list[float]: #查q表
    return q_table.setdefault(state_key(state), [0.0] * ACTION_COUNT) #查表


def choose_action( 
    q_table: dict[str, list[float]],
    state: tuple[int, int, int, int, int, int],
    epsilon: float,
) -> int:
    if random.random() < epsilon: #epsilon由我设定
        return random.randrange(ACTION_COUNT)

    q_values = get_q_values(q_table, state)
    return max(range(ACTION_COUNT), key=lambda action: q_values[action])


def train(
    episodes: int, #训练局数
    output: Path, #结果文件，也就是持久化之后的字典
    learning_rate: float = 0.1, #一次学习的步长，越大越快，但可能不稳定
    discount: float = 0.95, #折扣因子，越大越看重未来奖励，越小越看重当前奖励
    start_epsilon: float = 1.0, #初始探索率，越大越激进，越小越保守
    min_epsilon: float = 0.03, #最小探索率，决定程序右倾到什么程度
) -> dict[str, list[float]]: #初始化字典与pb
    q_table: dict[str, list[float]] = {}
    best_score = 0

    for episode in range(1, episodes + 1): #注意episode不是epsilon，episode是训练局数，epsilon是探索率
        env = DinoEnv(seed=episode) #每局游戏都不一样！
        observation = env.reset() #observation是一个列表，里面有6个元素，分别是距离、障碍物宽度、障碍物速度、恐龙高度、恐龙速度、是否在地面，而env.reset()会返回一个observation，意味着game start！
        state = discretize(observation) #discretize函数是把observation乘以桶数取整
        epsilon = max(min_epsilon, start_epsilon * (1 - episode / episodes))

        while not env.done and env.steps < 6000:
            action = choose_action(q_table, state, epsilon)
            next_observation, reward, done, info = env.step(action)
            next_state = discretize(next_observation)

            q_values = get_q_values(q_table, state)
            next_q_values = get_q_values(q_table, next_state)
            target = reward + discount * max(next_q_values) * (0 if done else 1)
            q_values[action] += learning_rate * (target - q_values[action])
            state = next_state

        best_score = max(best_score, int(info["score"]))
        if episode == 1 or episode % max(1, episodes // 10) == 0:
            print(
                f"episode={episode:5d} score={info['score']:3d} "
                f"best={best_score:3d} epsilon={epsilon:.3f} states={len(q_table)}"
            )

    save_q_table(q_table, output)
    print(f"saved {len(q_table)} states to {output}")
    return q_table


def save_q_table(q_table: dict[str, list[float]], path: Path) -> None:
    path.write_text(json.dumps(q_table, indent=2), encoding="utf-8")


def load_q_table(path: str | Path = DEFAULT_Q_TABLE) -> dict[str, list[float]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def q_agent_action(env: DinoEnv, q_table: dict[str, list[float]]) -> int:
    state = discretize(env.get_observation())
    q_values = get_q_values(q_table, state)
    return max(range(ACTION_COUNT), key=lambda action: q_values[action])


def evaluate(q_table: dict[str, list[float]], episodes: int) -> None:
    scores = []
    for episode in range(1, episodes + 1):
        env = DinoEnv(seed=10_000 + episode)
        env.reset()
        while not env.done and env.steps < 6000:
            env.step(q_agent_action(env, q_table))
        scores.append(env.score)
    average = sum(scores) / len(scores)
    print(f"evaluation episodes={episodes} average_score={average:.2f} best_score={max(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple Q-learning agent for the Dino environment.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q_TABLE)
    parser.add_argument("--eval", type=int, default=20)
    args = parser.parse_args()

    q_table = train(args.episodes, args.output)
    evaluate(q_table, args.eval)


if __name__ == "__main__":
    main()
