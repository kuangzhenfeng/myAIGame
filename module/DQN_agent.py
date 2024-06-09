# DQN_agent.py

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

        # 创建深度神经网络模型
        self.model = self._build_model()

    def _build_model(self):
        # 构建一个卷积神经网络
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(self.env.grid_size, self.env.grid_size, 2)))
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def update_model(self, state, action, reward, next_state):
        target = self.model.predict(np.expand_dims(state, axis=0))
        target[0][action] = reward + self.gamma * np.max(self.model.predict(np.expand_dims(next_state, axis=0))[0])
        self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_model(state, action, reward, next_state)
                state = next_state
                # 训练可视化
                self.env.render()
                # time.sleep(0.3)
            self.epsilon *= self.epsilon_decay
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Epsilon {self.epsilon:.4f}")