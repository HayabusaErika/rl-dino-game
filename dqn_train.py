from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dino_env import DinoEnv


DEFAULT_MODEL = Path("dqn_model.pth")
DEFAULT_REPLAY = Path("replay_buffer.json")


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
    def __init__(self, capacity: int = 50000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

    def save(self, path: Path) -> None:
        data = list(self.buffer)
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.buffer = deque(data, maxlen=self.buffer.maxlen)


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
        batch_size: int = 64,
        target_update: int = 100,
        buffer_size: int = 50000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

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

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).max(dim=1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path: Path) -> None:
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


def train(
    episodes: int,
    output: Path,
    buffer_output: Path,
    hidden_size: int = 128,
    lr: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    batch_size: int = 64,
    target_update: int = 100,
    buffer_size: int = 50000,
    save_every: int = 10000,
) -> DQNAgent:
    agent = DQNAgent(
        state_size=6,
        action_size=2,
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update=target_update,
        buffer_size=buffer_size,
    )

    best_score = 0
    print_interval = max(1, episodes // 10)
    scores = []

    for episode in range(1, episodes + 1):
        env = DinoEnv(seed=episode)
        observation = env.reset()
        state = observation
        episode_reward = 0
        episode_loss = []

        while not env.done and env.steps < 6000:
            action = agent.select_action(state)
            next_observation, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_observation, float(done))

            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)

            state = next_observation
            episode_reward += reward

        score = int(info["score"])
        scores.append(score)
        best_score = max(best_score, score)

        if episode == 1 or episode % print_interval == 0:
            avg_score = sum(scores[-print_interval:]) / len(scores[-print_interval:])
            avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
            print(
                f"episode={episode:6d} score={score:3d} best={best_score:3d} "
                f"avg={avg_score:.1f} eps={agent.epsilon:.3f} loss={avg_loss:.4f} "
                f"buffer={len(agent.replay_buffer)}"
            )

        if episode % save_every == 0:
            agent.save(output)
            agent.replay_buffer.save(buffer_output)

    agent.save(output)
    agent.replay_buffer.save(buffer_output)
    print(f"model saved to {output}")
    print(f"buffer saved to {buffer_output}")
    return agent


def evaluate(agent: DQNAgent, episodes: int) -> None:
    agent.epsilon = 0.0
    scores = []
    for episode in range(1, episodes + 1):
        env = DinoEnv(seed=10_000 + episode)
        observation = env.reset()
        state = observation
        while not env.done and env.steps < 6000:
            action = agent.select_action(state)
            state, _, _, _ = env.step(action)
        scores.append(env.score)
    average = sum(scores) / len(scores)
    print(f"evaluation episodes={episodes} average_score={average:.2f} best_score={max(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN training for Dino environment.")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--buffer", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--eval", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=10000)
    args = parser.parse_args()

    agent = train(
        args.episodes, args.output, args.buffer,
        hidden_size=args.hidden, lr=args.lr, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay, batch_size=args.batch_size,
        target_update=args.target_update, buffer_size=args.buffer_size,
        save_every=args.save_every,
    )
    evaluate(agent, args.eval)


if __name__ == "__main__":
    main()
