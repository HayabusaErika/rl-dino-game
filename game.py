import argparse
import random

import pygame

from dino_env import GROUND_Y, HEIGHT, WIDTH, DinoEnv
from q_learning import DEFAULT_Q_TABLE, load_q_table, q_agent_action


WHITE = (248, 248, 248)
BLACK = (28, 28, 28)
GREEN = (38, 130, 84)
BLUE = (58, 104, 180)
GRAY = (120, 120, 120)


def draw(screen: pygame.Surface, env: DinoEnv, font: pygame.font.Font) -> None:
    screen.fill(WHITE)
    pygame.draw.line(screen, GRAY, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

    dino = env.dino_rect()
    obstacle = env.obstacle_rect()
    pygame.draw.rect(screen, BLUE, pygame.Rect(dino.x, dino.y, dino.width, dino.height))
    pygame.draw.rect(screen, GREEN, pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height))

    score_text = font.render(f"Score: {env.score}  Speed: {env.speed:.1f}", True, BLACK)
    screen.blit(score_text, (18, 16))

    if env.done:
        text = font.render("Game Over - Press R to restart", True, BLACK)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 80))


def rule_agent(env: DinoEnv) -> int:
    distance = env.obstacle.x - env.dino.x
    danger_zone = 85 + env.speed * 7
    if env.dino.on_ground and 0 < distance < danger_zone:
        return 1
    return 0


def random_agent() -> int:
    return 1 if random.random() < 0.04 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Dino game designed for RL experiments.")
    parser.add_argument("--agent", choices=["human", "rule", "random", "q"], default="human")
    parser.add_argument("--q-table", default=str(DEFAULT_Q_TABLE))
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Dino Mini Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    env = DinoEnv()
    env.reset()
    q_table = load_q_table(args.q_table) if args.agent == "q" else None

    running = True
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1
                elif event.key == pygame.K_r:
                    env.reset()

        keys = pygame.key.get_pressed()
        if args.agent == "human" and keys[pygame.K_SPACE]:
            action = 1
        elif args.agent == "rule":
            action = rule_agent(env)
        elif args.agent == "random":
            action = random_agent()
        elif args.agent == "q" and q_table is not None:
            action = q_agent_action(env, q_table)

        if not env.done:
            env.step(action)

        draw(screen, env, font)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
