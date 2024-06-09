# main.py

"""
tensorflow当前版本无法调用gpu，暂时命令行先执行
export PATH=/home/stevenk/.pyenv/versions/3.9.16/lib/python3.9/site-packages/nvidia/cudnn/lib:$PATH
"""

import os
import time
import tensorflow as tf
from module.catch_game_env import CatchGameEnv
from module.DQN_agent import DQNAgent

def main():
    env = CatchGameEnv(grid_size=5, max_steps=100)
    # 创建抓捕者和逃跑者的智能体
    catcher_agent = DQNAgent(env)
    runner_agent = DQNAgent(env)

    # 交替训练抓捕者和逃跑者
    for episode in range(100):
        # 抓捕者训练
        catcher_agent.train(1)
        # 逃跑者训练
        runner_agent.train(1)

    state = env.reset()

    running = True
    while running:
        # 抓捕者选择动作并执行
        catcher_action = catcher_agent.choose_action(state)
        next_state, _, done, _ = env.step(catcher_action)
        state = next_state
        env.render()

        if done:
            state = env.reset()

        # 逃跑者选择
        runner_action = runner_agent.choose_action(state)
        next_state, _, done, _ = env.step(runner_action)
        state = next_state
        env.render()

        if done:
            state = env.reset()
        time.sleep(0.3)

if __name__ == "__main__":
    main()