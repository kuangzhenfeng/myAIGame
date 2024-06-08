import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.action_space.n))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state = tuple(state.flatten())
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        state = tuple(state.flatten())
        next_state = tuple(next_state.flatten())
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.epsilon *= self.epsilon_decay
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Epsilon {self.epsilon:.4f}")
