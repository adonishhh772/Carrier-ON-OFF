import torch
import numpy as np
import pandas as pd
import random
from collections import deque
import seaborn as sns
import copy
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from numpy import savetxt
import csv
from deepEnvLive import CarrierEnvLive
import threading

"""
Implementation of DQN for gym environments with discrete action space.
source: 
"""

def test_model(model, env,state_flattened_size):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
    done = False
    rewards = []
    states_actions = []
    while not np.all(done):
        # DQN
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        states_actions.append((state.cpu().numpy(), action))  # Capture state and action
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        rewards.append(reward)
        print("Request:", len(rewards), "Action:", action, "Reward:", reward)
    print("Reward sum:", sum(rewards))
    return rewards, states_actions

def save_plot_and_csv_train(total_rewards):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_train.png')
    # save to csv file
    savetxt('dqn_train.csv', total_rewards, delimiter=',')

def save_plot_and_csv_test(total_rewards):
    num_bs = len(total_rewards[0])
    # Create a dictionary to store rewards for each BS
    rewards_dict = {f'BS{i+1} Reward': [entry[i] for entry in total_rewards] for i in range(num_bs)}
    # Add episode numbers to the dictionary
    rewards_dict['Episode'] = range(1, len(total_rewards) + 1)
    # Create a DataFrame for plotting
    df = pd.DataFrame(rewards_dict)

    # Convert 'inf' values to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Plot the values
    plt.figure(figsize=(12, 6))
    # for i in range(num_bs):
    sns.lineplot(data=df, x='Episode', y=f'BS1 Reward', marker='o', label=f'BS1 Reward')

    plt.title('Rewards for All Base Stations Over Training Episodes')
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.ylim(0, 1)  # Adjust the y-axis range if needed
    plt.savefig('bs_rewards_line_plot.png')
    # save to csv file
    savetxt('dqn_test.csv', total_rewards, delimiter=',')

def overall_reward_plot(data):
    # # Create a dictionary to store rewards for each BS
    overall_dict = {'Overall Reward': [sum(entry) for entry in data]}
    # Add overall reward to the dictionary
    # Add episode numbers to the dictionary
    overall_dict['Episode'] = range(1, len(data) + 1)
    # Create a DataFrame for plotting
    df_overall = pd.DataFrame(overall_dict)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_overall, x='Episode', y='Overall Reward', marker='o', label='Overall Reward', linewidth=2.5)
    plt.title('Overall Reward Over Training Episodes')
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig('bs_rewards_overall_line_plot.png')

def dqn_agent(gamma = 0.9, epsilon = 0.5, learning_rate = 1e-3,state_flattened_size = 845, epochs = 5000,mem_size = 50000,
    batch_size = 128,sync_freq = 16,l1 = 845, l2 = 1500, l3 = 700,l4 = 200, l5 = 5, env=""):
    """
    :param gamma: reward discount factor
    :param epsilon: probability to take a random action during training
    :param learning_rate: learning rate for the Q-Network
    :param batch_size: see above
    :param env_name: name of the gym environment
    :return: 
    """
    env= CarrierEnvLive()
    state_flattened_size = env.state.shape[0]
    l5 = env.action_space.n
    l1 = env.state.shape[0]
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, l5)
    )

    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    total_reward_list = []

    n_action = env.action_space.n
    
    replay = deque(maxlen=mem_size)
    
    states_actions = []
    
    for i in range(epochs):
        print("Starting training, epoch:", i)
        cnt = 0
        total_reward = 0
        _state = env.reset()
        state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        done = False
        # env.reset()
        print("TRAIN AGENT")
        while not np.all(done):
            print("Step:", cnt + 1)
            cnt += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0, n_action -1)
            else:
                action_ = np.argmax(qval_)

            states_actions.append((state1.cpu().numpy(), action_))
            state, reward, done, _ = env.step(action_)
            state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, state_flattened_size)

            exp = (state1, action_, reward, state2, done)

            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)

                # Ensure action_batch is correctly shaped
                action_batch = action_batch.long().view(-1, 1)

                X_list = []
                Y_list = []

                # Process each dimension in reward_batch and done_batch
                for dim in range(reward_batch.shape[1]):
                    reward_step = reward_batch[:, dim]
                    done_step = done_batch[:, dim]

                    # Compute Y for each element in the batch
                    max_q_value_next = torch.max(Q2, dim=1)[0]
                    Y = reward_step + gamma * ((1 - done_step) * max_q_value_next)

                    # Gather Q-values for the taken actions
                    X = Q1.gather(dim=1, index=action_batch).squeeze()

                    # print(f"X shape: {X.shape}")
                    # print(X)
                    
                     # Store X and Y values
                    X_list.append(X.unsqueeze(dim=1))  # Add an extra dimension for stacking
                    Y_list.append(Y.unsqueeze(dim=1))  # Add an extra dimension for stacking

                # Stack X and Y values from all dimensions to create the correct shapes
                X_all = torch.cat(X_list, dim=1)
                Y_all = torch.cat(Y_list, dim=1)

                # Compute loss and backpropagate
                loss = loss_fn(X_all, Y_all)
                print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                # Synchronize model parameters with model2 at specified frequency
                if cnt % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
            
            print("Reward:",reward)
            total_reward += reward

        total_reward_list.append(total_reward)
        save_plot_and_csv_train(total_reward_list)

        print("Episode reward:", total_reward)

        if epsilon > 0.01:
            epsilon -= (1 / epochs)

        #GUARDAR total_reward_list do TRAIN
    torch.save(model.state_dict(), 'dqn.pt')

def create_action_line_plot(action_history):
    action_list = []
    for i in range(len(action_history)):
        action_list.append([i,action_history[i][0][1]])
    # # Flatten the action history and create a DataFrame
    # action_data = [(episode, step, action) for episode, actions in enumerate(action_history) for step, action in enumerate(actions)]
    df = pd.DataFrame(action_list, columns=['Episode', 'Action'])
    
    # # Create a line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Episode', y='Action', marker='o')
    plt.title('Actions Taken Over Training Episodes')
    plt.xlabel('Training Episode')
    plt.ylabel('Action')
    plt.savefig('action_line_plot.png')
    # plt.show()

def create_truth_table(data,rewards):
    num_bs =len(rewards[0])
    flattened_data = [(arr[0][:num_bs].tolist(), action) for arr, action in data]

    # Convert the flattened data to a DataFrame
    df = pd.DataFrame(flattened_data, columns=['State', 'Action'])
    state_columns = [f'User_BS_{i+1}' for i in range(num_bs)]
    df[state_columns] = pd.DataFrame(df['State'].tolist(), index=df.index)
    df.drop(columns=['State'], inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv('truth_table.csv', index=False)

    # Example 1: Bar Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Action')
    plt.title('Frequency of Each Action')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.savefig('action_frequency_bar_plot.png')

    # Example 2: Heatmap
    # Creating a pivot table for heatmap
    heatmap_data = df.pivot_table(index=state_columns[0], columns='Action', aggfunc='size', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f'Heatmap of Users per {state_columns[0]} and Actions')
    plt.xlabel('Action')
    plt.ylabel(state_columns[0])
    plt.savefig('users_bs1_action_heatmap.png')

    # Example 3: Pair Plot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, hue='Action', vars=state_columns)
    plt.savefig('state_action_pair_plot.png')

    # Example 4: Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = df[state_columns + ['Action']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap of Users per BS and Action')
    plt.savefig('state_action_correlation_heatmap.png')

    # Example 5: Box Plot
    plt.figure(figsize=(12, 8))
    df_melted = df.melt(id_vars=['Action'], value_vars=state_columns, var_name='State', value_name='Value')
    sns.boxplot(x='State', y='Value', hue='Action', data=df_melted)
    plt.title('Box Plot of Users per BS by Action')
    plt.savefig('state_action_box_plot.png')

def create_each_power_sinr_datarate_plot(data):   
    # Initialize lists
    num_episodes = len(data)

    # Initialize lists
    episode_numbers = []
    num_ue_list = []
    power_list = []
    sinr_dict = {}
    data_rate_dict = {}

    # Populate the lists and dictionaries with data
    for episode_idx, episode in enumerate(data):
        episode_numbers.append(episode_idx + 1)
        
        num_ue = int(episode[0][0][0][0] + episode[0][0][0][1])
        num_ue_list.append(num_ue)
        power = episode[0][0][0][-1]  # Last element for power
        power_list.append(power)
        
        for i in range(num_ue):
            sinr_key = f'UE{i+1} SINR'
            data_rate_key = f'UE{i+1} Data Rate'
            
            if sinr_key not in sinr_dict:
                sinr_dict[sinr_key] = []
            if data_rate_key not in data_rate_dict:
                data_rate_dict[data_rate_key] = []
            
            sinr_dict[sinr_key].append(episode[0][0][0][i + 2])
            data_rate_dict[data_rate_key].append(episode[0][0][0][i + 2 + num_ue])

    # Create DataFrames for plotting
    power_df = pd.DataFrame({'Episode': episode_numbers, 'Power': power_list})
    sinr_df = pd.DataFrame(sinr_dict)
    sinr_df['Episode'] = episode_numbers
    data_rate_df = pd.DataFrame(data_rate_dict)
    data_rate_df['Episode'] = episode_numbers

    # Calculate the number of subplots needed
    num_sinr_plots = len(sinr_dict)
    num_data_rate_plots = len(data_rate_dict)
    total_plots = 1 + num_sinr_plots + num_data_rate_plots

    # Create a figure with subplots for power, SINR, and data rate
    rows = (total_plots + 2) // 3  # Number of rows needed
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot power
    axes[0].set_xlabel('Training Episode')
    axes[0].set_ylabel('Power')
    axes[0].set_ylim(0, max(power_list) * 1.1)
    axes[0].plot(power_df['Episode'], power_df['Power'], color='tab:blue', marker='o', label='Power')
    axes[0].legend(loc='upper left')
    axes[0].set_title('Power Over Training Episodes')

    # Plot SINR for each UE
    for idx, col in enumerate(sinr_df.columns):
        if col != 'Episode':
            sinr_ax = axes[idx + 1]
            sinr_ax.set_xlabel('Training Episode')
            sinr_ax.set_ylabel('SINR')
            sinr_ax.set_ylim(0, sinr_df[col].max() * 1.1)
            sinr_ax.plot(sinr_df['Episode'], sinr_df[col], marker='o', label=col)
            sinr_ax.legend(loc='upper left')
            sinr_ax.set_title(f'SINR Over Training Episodes for {col}')

    # Plot data rate for each UE
    for idx, col in enumerate(data_rate_df.columns):
        if col != 'Episode':
            data_rate_ax = axes[num_sinr_plots + idx + 1]
            data_rate_ax.set_xlabel('Training Episode')
            data_rate_ax.set_ylabel('Data Rate')
            data_rate_ax.set_ylim(0, data_rate_df[col].max() * 1.1)
            data_rate_ax.plot(data_rate_df['Episode'], data_rate_df[col], marker='o', label=col)
            data_rate_ax.legend(loc='upper left')
            data_rate_ax.set_title(f'Data Rate Over Training Episodes for {col}')

    # Hide any unused subplots
    for i in range(num_sinr_plots + num_data_rate_plots + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('subplots_power_sinr_datarate.png')

def create_combined_sinr_data_power(data):
    num_episodes = len(data)

    # Initialize lists
    episode_numbers = []
    num_ue_list = []
    power_list = []
    sinr_dict = {}
    data_rate_dict = {}

    # Populate the lists and dictionaries with data
    for episode_idx, episode in enumerate(data):
        episode_numbers.append(episode_idx + 1)
        
        num_ue = int(episode[0][0][0][0] + episode[0][0][0][1])
        num_ue_list.append(num_ue)
        power = episode[0][0][0][-1]  # Last element for power
        power_list.append(power)
        
        for i in range(num_ue):
            sinr_key = f'UE{i+1} SINR'
            data_rate_key = f'UE{i+1} Data Rate'
            
            if sinr_key not in sinr_dict:
                sinr_dict[sinr_key] = []
            if data_rate_key not in data_rate_dict:
                data_rate_dict[data_rate_key] = []
            
            sinr_dict[sinr_key].append(episode[0][0][0][i + 2])
            data_rate_dict[data_rate_key].append(episode[0][0][0][i + 2 + num_ue])

    # Create DataFrames for plotting
    power_df = pd.DataFrame({'Episode': episode_numbers, 'Power': power_list})
    sinr_df = pd.DataFrame(sinr_dict)
    sinr_df['Episode'] = episode_numbers
    data_rate_df = pd.DataFrame(data_rate_dict)
    data_rate_df['Episode'] = episode_numbers

    # Create a figure with subplots for power, combined SINR, and combined data rate
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Plot power
    axes[0].set_xlabel('Training Episode')
    axes[0].set_ylabel('Power')
    axes[0].set_ylim(0, max(power_list) * 1.1)
    axes[0].plot(power_df['Episode'], power_df['Power'], color='tab:blue', marker='o', label='Power')
    axes[0].legend(loc='upper left')
    axes[0].set_title('Power Over Training Episodes')

    # Plot combined SINR for all UEs
    for col in sinr_df.columns:
        if col != 'Episode':
            axes[1].plot(sinr_df['Episode'], sinr_df[col], marker='o', label=col)
    axes[1].set_xlabel('Training Episode')
    axes[1].set_ylabel('SINR')
    axes[1].set_ylim(0, max(sinr_df.iloc[:, :-1].max()) * 1.1)  # Adjust y-axis to start from 0
    axes[1].legend(loc='upper left')
    axes[1].set_title('SINR Over Training Episodes for All UEs')

    # Plot combined data rate for all UEs
    for col in data_rate_df.columns:
        if col != 'Episode':
            axes[2].plot(data_rate_df['Episode'], data_rate_df[col], marker='o', label=col)
    axes[2].set_xlabel('Training Episode')
    axes[2].set_ylabel('Data Rate')
    axes[2].set_ylim(0, max(data_rate_df.iloc[:, :-1].max()) * 1.1)  # Adjust y-axis to start from 0
    axes[2].legend(loc='upper left')
    axes[2].set_title('Data Rate Over Training Episodes for All UEs')

    plt.tight_layout()
    plt.savefig('combined_subplots_power_sinr_datarate.png')

def test_dqn_agent():
    env = CarrierEnvLive()
    state_flattened_size = env.state.shape[0]
    l5 = env.action_space.n
    l1 = env.state.shape[0]
    l2 = 1500
    l3 = 700
    l4 = 200

    model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4),
            torch.nn.ReLU(),
            torch.nn.Linear(l4, l5)
        )
    print("TESTING AGENT")
    model_test=model
       
    model_test.load_state_dict(torch.load("dqn.pt"))
    total_reward_list = []
    total_test_states_action_list = []
    for i in range(0,100):
        test_rewards, test_states_actions=test_model(model_test,env,state_flattened_size)
        total_reward_list.append(test_rewards)
        total_test_states_action_list.append(test_states_actions)
   
    flattened_data = [list(arr[0]) for arr in total_reward_list]
    save_plot_and_csv_test(flattened_data)
    overall_reward_plot(flattened_data)
    # print(test_states_actions)
    # exit()

    create_action_line_plot(total_test_states_action_list)

    create_each_power_sinr_datarate_plot(total_test_states_action_list)

    create_combined_sinr_data_power(total_test_states_action_list)
    
   
    # flattened_data.insert(0, list(reward_sum))  # Insert the reward sum at the beginning if needed
   
   
    truth_table_data = [list(arr[0]) for arr in total_test_states_action_list]
    create_truth_table(truth_table_data,flattened_data)


# Function for training in each thread
def train_thread(model, model2, env, replay, batch_size, sync_freq, state_flattened_size, epochs, thread_id, gamma, epsilon, total_reward_list, lock, thread_log):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_action = env.action_space.n
    
    
    for i in range(epochs):
        print(f"Thread {thread_id} - Epoch: {i}")
        # print(epochs)
        
        cnt = 0
        total_reward = 0
        _state = env.reset()
        state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        done = False
        
        while not np.all(done):
            cnt += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, n_action)
            else:
                action_ = np.argmax(qval_)
                
            state, reward, done, _ = env.step(action_)
            state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, state_flattened_size)
            
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)
                    
                action_batch = action_batch.long().view(-1, 1)
                
                X_list = []
                Y_list = []

                # Process each dimension in reward_batch and done_batch
                for dim in range(reward_batch.shape[1]):
                    reward_step = reward_batch[:, dim]
                    done_step = done_batch[:, dim]

                    # Compute Y for each element in the batch
                    max_q_value_next = torch.max(Q2, dim=1)[0]
                    Y = reward_step + gamma * ((1 - done_step) * max_q_value_next)

                    # Gather Q-values for the taken actions
                    X = Q1.gather(dim=1, index=action_batch).squeeze()

                    # print(f"X shape: {X.shape}")
                    # print(X)
                    
                     # Store X and Y values
                    X_list.append(X.unsqueeze(dim=1))  # Add an extra dimension for stacking
                    Y_list.append(Y.unsqueeze(dim=1))  # Add an extra dimension for stacking

                # Stack X and Y values from all dimensions to create the correct shapes
                X_all = torch.cat(X_list, dim=1)
                Y_all = torch.cat(Y_list, dim=1)

                # Compute loss and backpropagate
                loss = loss_fn(X_all, Y_all)
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if cnt % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                    
            total_reward += reward
            
            # Log step details (Thread ID, Epoch, Step, Reward)
            thread_log.append([thread_id, i, cnt, reward])
        
        # Update the shared rewards list safely
        with lock:
            total_reward_list.append(total_reward)
        print(f"Thread {thread_id} - Total reward: {total_reward}")

def dqn_agent_multithreaded(num_threads=1, gamma=0.9, epsilon=0.5, state_flattened_size=845, total_epochs=5000, mem_size=50000,
                            batch_size=256, sync_freq=16):
    # Initialize the environment and models
    env = CarrierEnvLive()
    state_flattened_size = env.state.shape[0]
    action_size = env.action_space.n
    
    model = torch.nn.Sequential(
        torch.nn.Linear(state_flattened_size, 1500),
        torch.nn.ReLU(),
        torch.nn.Linear(1500, 700),
        torch.nn.ReLU(),
        torch.nn.Linear(700, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, action_size)
    )
    
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    
    replay = deque(maxlen=mem_size)
    total_reward_list = []
    lock = threading.Lock()
    
    # Create a list to store logs for each thread
    thread_logs = [[] for _ in range(num_threads)]
    
    # Calculate epochs per thread
    epochs_per_thread = total_epochs // num_threads
    
    
    # Create and start threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(target=train_thread, args=(model, model2, env, replay, batch_size, sync_freq, state_flattened_size,
                                                             epochs_per_thread, thread_id, gamma, epsilon, total_reward_list, lock, thread_logs[thread_id]))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    print("Training completed.")
    
    # Save the model
    torch.save(model.state_dict(), 'dqnLive.pt')
    
    # Plot and save rewards
    plt.plot(total_reward_list)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Epochs')
    plt.savefig('training_rewards.png')
    
    savetxt('training_rewards.csv', total_reward_list, delimiter=',')
    
    # Save thread logs to a CSV file
    with open('dqn_multithreaded_logs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Thread ID', 'Epoch', 'Step', 'Reward'])  # Write header
        for log in thread_logs:
            writer.writerows(log)  # Write each thread's log

if __name__ == "__main__":
    dqn_agent_multithreaded()

