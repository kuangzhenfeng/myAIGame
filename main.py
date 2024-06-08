import time
from module.catch_game_env import CatchGameEnv
from module.q_learning_agent import QLearningAgent

def main():
    env = CatchGameEnv()
    agent = QLearningAgent(env)
    agent.train(1000)

    state = env.reset()

    running = True
    while running:
        action = agent.choose_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state
        env.render()

        if done:
            state = env.reset()
        time.sleep(0.3)

if __name__ == "__main__":
    main()