import pygame
import numpy as np
import gym
from gym import spaces

class CatchGameEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(CatchGameEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(0, grid_size - 1, shape=(2, 2), dtype=np.int32)

        self.catcher_pos = np.array([0, 0])
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])

        pygame.init()
        self.screen_width = 400
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Catch Game")

    def reset(self):
        self.catcher_pos = np.array([0, 0])
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.catcher_pos, self.runner_pos])

    def step(self, action):
        if action == 0:
            self.catcher_pos[0] = max(0, self.catcher_pos[0] - 1)
        elif action == 1:
            self.catcher_pos[0] = min(self.grid_size - 1, self.catcher_pos[0] + 1)
        elif action == 2:
            self.catcher_pos[1] = max(0, self.catcher_pos[1] - 1)
        elif action == 3:
            self.catcher_pos[1] = min(self.grid_size - 1, self.catcher_pos[1] + 1)

        done = np.array_equal(self.catcher_pos, self.runner_pos)
        reward = 1 if done else -0.1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        cell_size = self.screen_width // self.grid_size
        self.screen.fill((255, 255, 255))

        pygame.draw.rect(self.screen, (255, 0, 0), (self.runner_pos[1] * cell_size, self.runner_pos[0] * cell_size, cell_size, cell_size))
        pygame.draw.rect(self.screen, (0, 0, 255), (self.catcher_pos[1] * cell_size, self.catcher_pos[0] * cell_size, cell_size, cell_size))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()