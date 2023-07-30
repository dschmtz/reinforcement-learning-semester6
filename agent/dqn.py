import random
import numpy as np
import tensorflow as tf
from collections import deque


class DQN:
    """
    Deep Q-Network (DQN) class for reinforcement learning.
    Reference: https://arxiv.org/pdf/1312.5602.pdf

    Attributes:
        action_space (int): Number of possible actions the agent can take.
        learning_rate (float): Learning rate for the neural network optimizer.
        window_size (int): Number of days in the price window for input data.
        gamma (float): Discount factor for future rewards in Q-learning.
        epsilon (float): Current exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon is decayed over time.
        epsilon_min (float): Minimum exploration rate for epsilon-greedy policy.
        batch_size (int): Size of mini-batches used in training.
        memory (deque): Experience replay memory, stores tuples of (state, action, reward, next_state, done).
        model (tf.keras.Model): Deep neural network model for Q-value approximation.

    Methods:
        __init__: Initialize the DQN agent with specified parameters and build the neural network model.
        build_model: Construct the neural network model using TensorFlow/Keras.
        remember: Store an experience tuple in the experience replay memory.
        act: Choose an action for the given state using epsilon-greedy policy.
        replay: Train the neural network using experience replay and Q-learning.
        save_model: Placeholder method for saving the trained model (not implemented).
    """

    def __init__(self, action_space, learning_rate=0.001, 
                 window_size=90, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, batch_size=64, memory_size=1000):
        self.action_space = action_space
        self.window_size = window_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()

    def build_model(self):
        # One input will be the balance and the current amount of stocks
        input_portfolio = tf.keras.Input(shape=(2), name="input_portfolio")
        # The other will be the prices of the last window_size days
        input_window = tf.keras.Input(shape=(self.window_size), name="input_window")

        window = tf.keras.layers.Dense(64, activation="relu")(input_window)
        window = tf.keras.layers.Dense(32, activation="relu")(window)
        window = tf.keras.layers.Dense(4, activation="relu")(window)

        combined = tf.keras.layers.Concatenate()([window, input_portfolio])
        combined = tf.keras.layers.Dense(128, activation="relu")(combined)
        combined = tf.keras.layers.Dense(64, activation="relu")(combined)

        output = tf.keras.layers.Dense(self.action_space, activation="softmax", name="actions")(combined)

        model = tf.keras.Model(inputs=[input_portfolio, input_window], outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: Chosing random actions
            return np.random.choice(2, 3)
        else:
            # Exploitation: Use modell predictions
            general, prices = state
            general = np.expand_dims(general, axis=0)
            prices = np.expand_dims(prices, axis=0)

            actions = self.model.predict([general, prices], verbose=0) 
            return actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Random sample of experiences from memory
        mini_batch = random.sample(self.memory, self.batch_size)

        for state, actions, reward, next_state, done in mini_batch:
            q_target = reward
            
            if not done:
                general, prices = next_state
                general = np.expand_dims(general, axis=0)
                prices = np.expand_dims(prices, axis=0)

                next_actions = self.model.predict([general, prices], verbose=0)
                q_target += self.gamma * np.amax(next_actions[0])

            general, prices = state
            general = np.expand_dims(general, axis=0)
            prices = np.expand_dims(prices, axis=0)

            next_actions = self.model.predict([general, prices], verbose=0)
            next_actions[0][np.argmax(actions)] = q_target
            
            result = self.model.train_on_batch([general, prices], next_actions)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return result
    
    def save_model(self):
        pass
    