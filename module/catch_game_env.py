# catch_game_env.py

import pygame
import numpy as np
import gym
from gym import spaces

class CatchGameEnv(gym.Env):
    def __init__(self, grid_size=10, max_steps=100):
        super(CatchGameEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # 上下左右四个动作
        self.observation_space = spaces.Box(0, grid_size - 1, shape=(2,), dtype=np.int32)

        # 抓捕者和逃跑者的初始位置
        self.catcher_pos = np.array([0, 0])
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])

        # 初始化Pygame窗口
        pygame.init()
        self.screen_width = 400
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Catch Game")

        # 设置字体
        self.font = pygame.font.Font(None, 36)
        self.rounds = 0
        self.steps = 0
        self.catcher_wins = 0
        self.runner_wins = 0

    def reset(self):
        # 重置抓捕者和逃跑者的位置
        self.catcher_pos = np.array([0, 0])
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.rounds += 1  # 每次重置增加回合数
        self.steps = 0  # 重置步数
        return self._get_obs()

    def _get_obs(self):
        # 返回当前状态观察
        obs = np.zeros((self.grid_size, self.grid_size, 2))      # 创建一个全零的二维数组作为状态观察
        obs[self.catcher_pos[0], self.catcher_pos[1], 0] = 1     # 在抓捕者位置上标记为1
        obs[self.runner_pos[0], self.runner_pos[1], 1] = 1       # 在逃跑者位置上标记为1
        return obs.reshape((self.grid_size, self.grid_size, 2))  # 返回状态观察，并将形状调整为 (grid_size, grid_size, 2)

    def step(self, action):
        # 根据动作更新抓捕者的位置
        if action == 0:
            self.catcher_pos[0] = max(0, self.catcher_pos[0] - 1)
        elif action == 1:
            self.catcher_pos[0] = min(self.grid_size - 1, self.catcher_pos[0] + 1)
        elif action == 2:
            self.catcher_pos[1] = max(0, self.catcher_pos[1] - 1)
        elif action == 3:
            self.catcher_pos[1] = min(self.grid_size - 1, self.catcher_pos[1] + 1)

        # 随机更新逃跑者的位置
        # self.runner_pos += np.random.randint(-1, 2, size=2)

        # 边界检查
        self.runner_pos = np.clip(self.runner_pos, 0, self.grid_size - 1)

        # 检查是否抓到逃跑者
        done = np.array_equal(self.catcher_pos, self.runner_pos) or self.steps >= self.max_steps
        reward = 1 if done else -0.1

        if done:
            if np.array_equal(self.catcher_pos, self.runner_pos):
                self.catcher_wins += 1
            else:
                self.runner_wins += 1

        # 增加步数
        self.steps += 1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        # 绘制游戏界面
        self.screen.fill((255, 255, 255))

        cell_size = self.screen_width // self.grid_size
        # 画网格
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (200, 200, 200), (i * cell_size, 0), (i * cell_size, self.screen_height))
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * cell_size), (self.screen_width, i * cell_size))

        # 画抓捕者和逃跑者
        catcher_color = (255, 0, 0)
        runner_color = (0, 0, 255)
        pygame.draw.rect(self.screen, catcher_color, (self.catcher_pos[1] * cell_size, self.catcher_pos[0] * cell_size, cell_size, cell_size))
        pygame.draw.rect(self.screen, runner_color, (self.runner_pos[1] * cell_size, self.runner_pos[0] * cell_size, cell_size, cell_size))

        # 绘制回合数、胜负数和步数
        text_rounds = self.font.render(f"Round: {self.rounds}", True, (0, 0, 0))
        text_catcher_wins = self.font.render(f"Catcher Wins: {self.catcher_wins}", True, (0, 0, 0))
        text_runner_wins = self.font.render(f"Runner Wins: {self.runner_wins}", True, (0, 0, 0))
        text_steps = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))

        self.screen.blit(text_rounds, (10, 10))
        self.screen.blit(text_catcher_wins, (10, 50))
        self.screen.blit(text_runner_wins, (10, 90))
        self.screen.blit(text_steps, (10, 130))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
