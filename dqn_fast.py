from __future__ import annotations

import argparse
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dino_env import DinoEnv


DEFAULT_MODEL = Path("dqn_model.pth")


class QNetwork(nn.Module):
    def __init__(self, state_size: int = 6, action_size: int = 2, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def push_batch(self, experiences: list[tuple]):
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def collect_episode(seed: int, epsilon: float) -> tuple[list[tuple], int]:
    env = DinoEnv(seed=seed)
    observation = env.reset()
    state = list(observation)
    experiences = []

    while not env.done and env.steps < 6000:
        action = random.randrange(2) if random.random() < epsilon else -1
        next_observation, reward, done, info = env.step(action if action >= 0 else 0)
        next_state = list(next_observation)
        experiences.append((state, action if action >= 0 else 0, reward, next_state, float(done)))
        state = next_state

    return experiences, int(info["score"])


def collect_batch_parallel(seeds: list[int], epsilon: float, workers: int = 8) -> tuple[list[tuple], list[int]]:
    all_experiences = []
    scores = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect_episode, seed, epsilon) for seed in seeds]
        for future in futures:
            exps, score = future.result()
            all_experiences.extend(exps)
            scores.append(score)

    return all_experiences, scores


class DQNAgent:
    def __init__(
        self,
        state_size: int = 6,
        action_size: int = 2,
        hidden_size: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 128,
        target_update: int = 100,
        buffer_size: int = 100000,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, updates: int = 1) -> float:
        total_loss = 0.0
        count = 0

        for _ in range(updates):
            if len(self.replay_buffer) < self.batch_size:
                break

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1
            self.steps += 1

            if self.steps % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return total_loss / max(count, 1)

    def save(self, path: Path) -> None:
        torch.save({
            "q_network": self.q_network.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["q_network"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


def train(
    episodes: int,
    output: Path,
    workers: int = 8,
    collect_size: int = 16,
    updates_per_batch: int = 32,
    hidden_size: int = 128,
    lr: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    batch_size: int = 128,
    target_update: int = 100,
    buffer_size: int = 100000,
) -> DQNAgent:
    agent = DQNAgent(
        state_size=6, action_size=2, hidden_size=hidden_size,
        lr=lr, gamma=gamma, epsilon_start=epsilon_start,
        epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
        batch_size=batch_size, target_update=target_update,
        buffer_size=buffer_size,
    )

    best_score = 0
    all_scores = []
    completed = 0
    start_time = time.time()

    print(f"Device: {agent.device}")
    print(f"Workers: {workers}, Collect batch: {collect_size}, Updates per batch: {updates_per_batch}")
    print(f"Starting training...")
    print()

    while completed < episodes:
        batch_count = min(collect_size, episodes - completed)
        seeds = list(range(completed, completed + batch_count))

        exps, scores = collect_batch_parallel(seeds, agent.epsilon, workers)
        agent.replay_buffer.push_batch(exps)
        avg_loss = agent.update(updates_per_batch)

        all_scores.extend(scores)
        best_score = max(best_score, max(scores))
        completed += batch_count

        elapsed = time.time() - start_time
        eps_per_sec = completed / elapsed

        recent = all_scores[-100:] if len(all_scores) >= 100 else all_scores
        avg = sum(recent) / len(recent)
        print(
            f"ep={completed:6d} best={best_score:3d} avg={avg:.1f} "
            f"eps={agent.epsilon:.3f} loss={avg_loss:.4f} "
            f"speed={eps_per_sec:.1f} ep/s"
        )

        if completed % 1000 == 0:
            agent.save(output)

    agent.save(output)
    elapsed = time.time() - start_time
    print()
    print(f"Done: {episodes} episodes in {elapsed:.1f}s ({episodes/elapsed:.1f} ep/s)")
    print(f"Model saved to {output}")
    return agent


def evaluate(agent: DQNAgent, episodes: int) -> None:
    agent.epsilon = 0.0
    scores = []
    print(f"Running evaluation ({episodes} episodes)...")
    for episode in range(1, episodes + 1):
        env = DinoEnv(seed=10_000 + episode)
        observation = env.reset()
        state = observation
        while not env.done and env.steps < 6000:
            action = agent.select_action(state)
            state, _, _, _ = env.step(action)
        scores.append(env.score)
        if episode % 10 == 0:
            print(f"  eval {episode}/{episodes}")
    average = sum(scores) / len(scores)
    print(f"evaluation episodes={episodes} average_score={average:.2f} best_score={max(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast DQN: GPU + multithreaded env collection.")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--eval", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8, help="thread workers for env collection")
    parser.add_argument("--collect-size", type=int, default=16, help="episodes per collection batch")
    parser.add_argument("--updates-per-batch", type=int, default=32, help="GPU updates per batch")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=100000)
    args = parser.parse_args()

    agent = train(
        args.episodes, args.output, args.workers,
        args.collect_size, args.updates_per_batch,
        hidden_size=args.hidden, lr=args.lr, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay, batch_size=args.batch_size,
        target_update=args.target_update, buffer_size=args.buffer_size,
    )
    evaluate(agent, args.eval)


if __name__ == "__main__":
    main()
