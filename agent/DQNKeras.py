# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from deepEnvLive import CarrierEnvLive


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='DQN')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = CarrierEnvLive()
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.state.shape[0]
        print(self.state_size)
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

       
        for i in range(self.batch_size):
            # print(f"target[i] shape: {target[i].shape}")
            # print(f"action[i]: {action[i]}")
            # print(f"reward[i]: {reward[i]}")
            # print(f"target_next[i]: {target_next[i]}")
            # print(f"max_future_q_value: {np.amax(target_next[i])}")
            # print(f"reward[i] + gamma * max_future_q_value: {reward[i] + self.gamma * np.amax(target_next[i])}")
            # correction on the Q value for the action used
            if np.all(done[i]):
                if np.all(done[i]):
                    if isinstance(reward[i], np.ndarray):
                        # Handle multidimensional rewards
                        for dim in range(len(reward[i])):
                            target[i][action[i]] = reward[i][dim]
                    else:
                    # Handle scalar rewards
                        target[i][action[i]] = reward[i]
                else:
                # Update Q-values for multidimensional or scalar rewards
                    if isinstance(reward[i], np.ndarray):
                        # Handle multidimensional rewards
                        for dim in range(len(reward[i])):
                            target[i][action[i]] = reward[i][dim] + self.gamma * np.amax(target_next[i])
                    else:
                        # Handle scalar rewards
                        target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not np.all(done):
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not np.all(done) or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if np.all(done):                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    # if i == 500:
                self.replay()
        if e % 100 == 0:
            print(f"Saving trained model after {e} episodes")
            self.save(f"cartpole-dqn-{e}.h5")
    
        # Save the model at the end of training
        print("Saving final trained model as cartpole-dqn.h5")
        self.save("carrier.h5")
        # return

    def test(self):
        self.load("carrier.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not np.all(done):
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if np.all(done):
                    print("episode: {}/{}, action: {}".format(e, self.EPISODES, action))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    # agent.run()
    agent.test()