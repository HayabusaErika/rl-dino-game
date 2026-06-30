from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from dino_env import DinoEnv


ACTION_COUNT = 2
DEFAULT_Q_TABLE = Path("q_table.json")

DIST_BUCKETS = 16
WIDTH_BUCKETS = 5
SPEED_BUCKETS = 8
DY_BUCKETS = 9
VEL_BUCKETS = 11
ONGROUND_BUCKETS = 2

M_WIDTH = SPEED_BUCKETS * DY_BUCKETS * VEL_BUCKETS * ONGROUND_BUCKETS
M_SPEED = DY_BUCKETS * VEL_BUCKETS * ONGROUND_BUCKETS
M_DY = VEL_BUCKETS * ONGROUND_BUCKETS
M_VEL = ONGROUND_BUCKETS
MAX_STATES = DIST_BUCKETS * WIDTH_BUCKETS * SPEED_BUCKETS * DY_BUCKETS * VEL_BUCKETS * ONGROUND_BUCKETS


def discretize(observation: list[float]) -> int:
    distance, obstacle_width, obstacle_speed, dino_y, dino_velocity_y, on_ground = observation
    d = min(DIST_BUCKETS - 1, int(distance * DIST_BUCKETS))
    w = min(WIDTH_BUCKETS - 1, int(obstacle_width * WIDTH_BUCKETS))
    s = min(SPEED_BUCKETS - 1, int(obstacle_speed * SPEED_BUCKETS))
    y = min(DY_BUCKETS - 1, int(dino_y * DY_BUCKETS))
    v = max(-5, min(5, int(dino_velocity_y * 6))) + 5
    g = int(on_ground)
    return d * M_WIDTH + w * M_SPEED + s * M_SPEED // SPEED_BUCKETS + y * M_DY // DY_BUCKETS + v * M_VEL // VEL_BUCKETS + g


def state_key(key: int) -> str:
    return str(key)


def train_fast(
    episodes: int,
    output: Path,
    learning_rate: float = 0.1,
    discount: float = 0.95,
    start_epsilon: float = 1.0,
    min_epsilon: float = 0.03,
) -> dict[str, list[float]]:
    q_table = np.zeros((MAX_STATES, ACTION_COUNT), dtype=np.float64)

    best_score = 0
    print_interval = max(1, episodes // 10)

    for episode in range(1, episodes + 1):
        epsilon = max(min_epsilon, start_epsilon * (1 - episode / episodes))

        env = DinoEnv(seed=episode)
        observation = env.reset()
        state = discretize(observation)

        while not env.done and env.steps < 6000:
            if random.random() < epsilon:
                action = random.randrange(ACTION_COUNT)
            else:
                action = 0 if q_table[state, 0] >= q_table[state, 1] else 1

            next_observation, reward, done, info = env.step(action)
            next_state = discretize(next_observation)

            target = reward + discount * max(q_table[next_state, 0], q_table[next_state, 1]) * (0 if done else 1)
            q_table[state, action] += learning_rate * (target - q_table[state, action])

            state = next_state

        score = int(info["score"])
        best_score = max(best_score, score)
        if episode == 1 or episode % print_interval == 0:
            print(f"episode={episode:6d} score={score:3d} best={best_score:3d} epsilon={epsilon:.3f}")

    print("converting to q_table...")
    q_dict: dict[str, list[float]] = {}
    for i in range(MAX_STATES):
        if q_table[i, 0] != 0.0 or q_table[i, 1] != 0.0:
            q_dict[str(i)] = [float(q_table[i, 0]), float(q_table[i, 1])]

    save_q_table(q_dict, output)
    print(f"saved {len(q_dict)} states to {output}")
    return q_dict


def save_q_table(q_table: dict[str, list[float]], path: Path) -> None:
    path.write_text(json.dumps(q_table, indent=2), encoding="utf-8")


def load_q_table(path: str | Path = DEFAULT_Q_TABLE) -> dict[str, list[float]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(q_dict: dict[str, list[float]], episodes: int) -> None:
    scores = []
    for episode in range(1, episodes + 1):
        env = DinoEnv(seed=10_000 + episode)
        env.reset()
        while not env.done and env.steps < 6000:
            state = discretize(env.get_observation())
            key = str(state)
            if key in q_dict:
                q0, q1 = q_dict[key]
            else:
                q0, q1 = 0.0, 0.0
            action = 0 if q0 >= q1 else 1
            env.step(action)
        scores.append(env.score)
    average = sum(scores) / len(scores)
    print(f"evaluation episodes={episodes} average_score={average:.2f} best_score={max(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Q-learning with numpy for M1 Pro.")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q_TABLE)
    parser.add_argument("--eval", type=int, default=30)
    args = parser.parse_args()

    q_dict = train_fast(args.episodes, args.output)
    evaluate(q_dict, args.eval)


if __name__ == "__main__":
    main()
