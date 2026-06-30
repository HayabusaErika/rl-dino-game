from __future__ import annotations

from dataclasses import dataclass
import random


WIDTH = 800
HEIGHT = 300
GROUND_Y = 245
GRAVITY = 1.1
JUMP_VELOCITY = -17.0
MAX_OBSTACLE_SPEED = 14.0


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def intersects(self, other: "Rect") -> bool:
        return (
            self.x < other.right
            and self.right > other.x
            and self.y < other.bottom
            and self.bottom > other.y
        )


@dataclass
class Dino:
    x: float = 80
    y: float = GROUND_Y - 44
    width: float = 34
    height: float = 44
    velocity_y: float = 0.0

    @property
    def on_ground(self) -> bool:
        return self.y >= GROUND_Y - self.height


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width


class DinoEnv:
    """Small Dino-like game environment designed around RL concepts.

    Actions:
        0 = do nothing
        1 = jump

    Observation:
        [distance_to_obstacle, obstacle_width, obstacle_speed, dino_y, dino_velocity_y, on_ground]
    """

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.dino = Dino()
        self.obstacle = Obstacle(WIDTH + 120, GROUND_Y - 35, 24, 35)
        self.speed = 7.0
        self.score = 0
        self.steps = 0
        self.done = False

    def reset(self) -> list[float]:
        self.dino = Dino()
        self.speed = 7.0
        self.score = 0
        self.steps = 0
        self.done = False
        self._spawn_obstacle(WIDTH + 160)
        return self.get_observation()

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, int | float]]:
        if self.done:
            return self.get_observation(), 0.0, True, self.info

        if action == 1 and self.dino.on_ground:
            self.dino.velocity_y = JUMP_VELOCITY

        self.dino.velocity_y += GRAVITY
        self.dino.y += self.dino.velocity_y

        ground_y = GROUND_Y - self.dino.height
        if self.dino.y > ground_y:
            self.dino.y = ground_y
            self.dino.velocity_y = 0.0

        self.obstacle.x -= self.speed
        if self.obstacle.right < 0:
            self.score += 1
            self._spawn_obstacle(WIDTH + self.random.randint(80, 260))
            self.speed = min(MAX_OBSTACLE_SPEED, self.speed + 0.25)

        self.steps += 1
        reward = 0.1
        if self._collided():
            self.done = True
            reward = -10.0
        elif self.obstacle.x + self.obstacle.width < self.dino.x and self.obstacle.x + self.obstacle.width + self.speed >= self.dino.x:
            reward = 5.0

        return self.get_observation(), reward, self.done, self.info

    def get_observation(self) -> list[float]:
        distance = max(0.0, self.obstacle.x - self.dino.x)
        return [
            distance / WIDTH,
            self.obstacle.width / 60.0,
            self.speed / MAX_OBSTACLE_SPEED,
            self.dino.y / HEIGHT,
            self.dino.velocity_y / 20.0,
            1.0 if self.dino.on_ground else 0.0,
        ]

    @property
    def info(self) -> dict[str, int | float]:
        return {"score": self.score, "steps": self.steps, "speed": self.speed}

    def dino_rect(self) -> Rect:
        return Rect(self.dino.x, self.dino.y, self.dino.width, self.dino.height)

    def obstacle_rect(self) -> Rect:
        return Rect(self.obstacle.x, self.obstacle.y, self.obstacle.width, self.obstacle.height)

    def _collided(self) -> bool:
        return self.dino_rect().intersects(self.obstacle_rect())

    def _spawn_obstacle(self, x: float) -> None:
        width = self.random.choice([22, 28, 36])
        height = self.random.choice([32, 40, 48])
        self.obstacle = Obstacle(x, GROUND_Y - height, width, height)
