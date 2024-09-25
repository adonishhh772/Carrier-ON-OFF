import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import csv
import threading
from deepEnvLive import CarrierEnvLive


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs=X_input, outputs=X, name='DQN')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = CarrierEnvLive()
        self.state_size = self.env.state.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 10000
        self.memory = deque(maxlen=2000)
        self.energy_efficiency_list = []
        self.reward_list = []
        self.penalty_list = []
        self.actions_frequency = []
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        self.local_output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

        # CSV file setup
        self.csv_file = os.path.join(self.local_output_file,"training_log.csv")
        self.model_file = os.path.join(self.local_output_file,"carrier.h5")
        self.ee_plot = os.path.join(self.local_output_file,"energy_efficiency_plot.png")
        self.reward_plot = os.path.join(self.local_output_file,"reward_plot.png")
        self.penalty_plot = os.path.join(self.local_output_file,"penalty_plot.png")
        self.action_plot = os.path.join(self.local_output_file,"actions_frequency_plot.png")

        self.create_csv_log()

    def create_csv_log(self):
        """Initialize the CSV log with headers."""
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Step", "State", "Action", "Reward", "Penalty", "Energy Efficiency"])

    def log_to_csv(self, episode, step, state, action, reward, penalty, energy_efficiency):
        """Log the episode details to CSV."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, step, state.tolist(), action, reward, penalty, energy_efficiency])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.env.num_sbs * 3 - 1)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if np.all(done[i]):
                if isinstance(reward[i], np.ndarray):
                    for dim in range(len(reward[i])):
                        target[i][action[i]] = reward[i][dim]
                else:
                    target[i][action[i]] = reward[i]
            else:
                if isinstance(reward[i], np.ndarray):
                    for dim in range(len(reward[i])):
                        target[i][action[i]] = reward[i][dim] + self.gamma * np.amax(target_next[i])
                else:
                    target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for e in range(self.EPISODES):
            total_reward = 0
            total_penalty = 0
            episode_actions = []  # Track actions for each episode
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_energy_efficiency = 0
            i = 0
            while not np.all(done):
                action = self.act(state)
                episode_actions.append(action)  # Store each action in the episode
                next_state, reward, done, info = self.env.step(action, e)
                penalty = info['total_penalty']
                energy_efficiency = info['energy_efficiency']
                total_energy_efficiency += energy_efficiency
                next_state = np.reshape(next_state, [1, self.state_size])

                total_reward += reward
                total_penalty += penalty
                self.remember(state, action, reward, next_state, done)
                
                # Log to CSV
                self.log_to_csv(e, i, state, action, reward, penalty, energy_efficiency)

                state = next_state
                i += 1

                if np.all(done):
                    print(f"episode: {e}/{self.EPISODES}, score: {i}, e: {self.epsilon:.2}")

                self.replay()

            # Store metrics
            avg_energy_efficiency = total_energy_efficiency / (i+1) if i > 0 else 0
            self.energy_efficiency_list.append(avg_energy_efficiency)
            self.reward_list.append(total_reward)
            self.penalty_list.append(total_penalty)
            
            # Count action frequencies for the episode
            action_freq = np.bincount(episode_actions, minlength=self.action_size)
            self.actions_frequency.append(action_freq)

            # Save data and update plots after each episode
            if (e + 1) % 2500 == 0:
                start_episode = e - 2499
                end_episode = e + 1
                self.save_plots(start_episode, end_episode)
            # self.save_plots()

        self.save(self.model_file)

    # def save_plots(self):
        #     """Update and save plots after each episode."""

        #     # Energy Efficiency Plot
        #     plt.figure()
        #     plt.plot(self.energy_efficiency_list, label="Energy Efficiency", color='green')
        #     plt.xlabel("Episode")
        #     plt.ylabel("Energy Efficiency")
        #     plt.grid(True)
        #     plt.savefig(self.ee_plot)
        #     plt.close()

        #     # Reward Plot
        #     plt.figure()
        #     plt.plot(self.reward_list, label="Total Reward", color='blue')
        #     plt.xlabel("Episode")
        #     plt.ylabel("Total Reward")
        #     plt.grid(True)
        #     plt.savefig(self.reward_plot)
        #     plt.close()

        #     # Penalty Plot
        #     plt.figure()
        #     plt.plot(self.penalty_list, label="Total Penalty", color='red')
        #     plt.xlabel("Episode")
        #     plt.ylabel("Total Penalty")
        #     plt.grid(True)
        #     plt.savefig(self.penalty_plot)
        #     plt.close()

        #     # Action Frequency Plot
        #     plt.figure()
        #     action_freq_sum = np.sum(self.actions_frequency, axis=0)  # Sum action frequencies over episodes
        #     plt.bar(range(self.action_size), action_freq_sum, color='purple')
        #     plt.xlabel("Action")
        #     plt.ylabel("Frequency")
        #     plt.title("Action Frequency Across Episodes")
        #     plt.grid(True)
        #     plt.savefig(self.action_plot)
        #     plt.close()

    def save_plots(self, start_episode, end_episode):
        """Update and save plots for only 2500 episodes at a time."""
        
        # Get the data slice for the 2500 episodes
        energy_efficiency_slice = self.energy_efficiency_list[start_episode:end_episode]
        reward_slice = self.reward_list[start_episode:end_episode]
        penalty_slice = self.penalty_list[start_episode:end_episode]
        actions_frequency_slice = np.sum(self.actions_frequency[start_episode:end_episode], axis=0)  # Sum action frequencies over these episodes

        # Energy Efficiency Plot
        plt.figure()
        plt.plot(energy_efficiency_slice, label="Energy Efficiency", color='green')
        plt.xlabel("Episode")
        plt.ylabel("Energy Efficiency")
        plt.title(f"Energy Efficiency (Episodes {start_episode} to {end_episode})")
        plt.grid(True)
        plt.savefig(os.path.join(self.local_output_file, f"energy_efficiency_{start_episode}_to_{end_episode}.png"))
        plt.close()

        # Reward Plot
        plt.figure()
        plt.plot(reward_slice, label="Total Reward", color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Total Reward (Episodes {start_episode} to {end_episode})")
        plt.grid(True)
        plt.savefig(os.path.join(self.local_output_file, f"reward_{start_episode}_to_{end_episode}.png"))
        plt.close()

        # Penalty Plot
        plt.figure()
        plt.plot(penalty_slice, label="Total Penalty", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Total Penalty")
        plt.title(f"Total Penalty (Episodes {start_episode} to {end_episode})")
        plt.grid(True)
        plt.savefig(os.path.join(self.local_output_file, f"penalty_{start_episode}_to_{end_episode}.png"))
        plt.close()

        # Action Frequency Plot
        plt.figure()
        plt.bar(range(self.action_size), actions_frequency_slice, color='purple')
        plt.xlabel("Action")
        plt.ylabel("Frequency")
        plt.title(f"Action Frequency (Episodes {start_episode} to {end_episode})")
        plt.grid(True)
        plt.savefig(os.path.join(self.local_output_file, f"actions_frequency_{start_episode}_to_{end_episode}.png"))
        plt.close()

    def train(self, start_episodes, num_episodes, ue_count):
        """
        Train the agent for a given number of episodes and a given UE count.
        :param start_episodes: Starting episode number (for logging purposes).
        :param num_episodes: Number of episodes to train in this phase.
        :param ue_count: The number of UEs to set in the environment for this training phase.
        """
        # Update UE count in the environment
        self.env.num_ue = ue_count
        # self.env.reset()

        for e in range(start_episodes, start_episodes + num_episodes):
            total_reward = 0
            total_penalty = 0
            episode_actions = [] 
            total_energy_efficiency = 0
            i = 0
            # Reset environment at the start of each episode
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_reward = 0

            while not done:
                # Choose action
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action, e)
                next_state = np.reshape(next_state, [1, self.state_size])

                penalty = info['total_penalty']
                energy_efficiency = info['energy_efficiency']
                total_energy_efficiency += energy_efficiency
                total_reward += reward
                total_penalty += penalty

                # Store the experience in memory
                self.remember(state, action, reward, next_state, done)
                self.log_to_csv(e, i, state, action, reward, penalty, energy_efficiency)
                state = next_state

                i += 1

                if done:
                    print(f"Episode: {e}/{start_episodes + num_episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.2}")
                    break

                # Perform experience replay
                self.replay()

            avg_energy_efficiency = total_energy_efficiency / (i+1) if i > 0 else 0
            self.energy_efficiency_list.append(avg_energy_efficiency)
            self.reward_list.append(total_reward)
            self.penalty_list.append(total_penalty)

             # Count action frequencies for the episode
            action_freq = np.bincount(episode_actions, minlength=self.action_size)
            self.actions_frequency.append(action_freq)

            # Save data and update plots after each episode
            if (e + 1) % 2500 == 0:
                start_episode = e - 2499
                end_episode = e + 1
                self.save_plots(start_episode, end_episode)

            # # Save model periodically
            # if e % 1000 == 0:
            #     self.save(self.model_file)
            #     print(f"Model saved after episode {e}")

    def run_training_with_progressive_ues(self, ue_counts, episodes_per_ue=10000):
        """
        Train the model progressively with increasing UEs.
        :param ue_counts: List of UE counts (e.g., [5, 10, 15, 20, 25, 30, 35, 40])
        :param episodes_per_ue: Number of episodes to train per UE configuration.
        """
        for i, ue_count in enumerate(ue_counts):
            print(f"Starting training with {ue_count} UEs.")

            # Load model if not starting from scratch
            if i > 0:
                self.load(self.model_file)

            # Train for the specified number of episodes for the current UE count
            self.train(i * episodes_per_ue, episodes_per_ue, ue_count)

            # Save the model after training with the current UE count
            self.save(self.model_file)
            print(f"Model saved after training with {ue_count} UEs.")


if __name__ == "__main__":
    agent = DQNAgent()
    # agent = DQNAgent()

    # List of UE counts to train progressively
    ue_counts = [5, 10, 15, 20, 25, 30, 35, 40]
    episodes_per_ue = 10

    # Run progressive training with increasing UEs
    # agent.run_training_with_progressive_ues(ue_counts, episodes_per_ue=10000)

    # # Create and start training thread
    agent_thread = threading.Thread(target=agent.run_training_with_progressive_ues,args=(ue_counts, episodes_per_ue))
    agent_thread.start()

    # # Wait for the training thread to complete
    agent_thread.join()
