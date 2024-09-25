from gym import Env
from gym.spaces import Discrete, Box
import random
import math
import numpy as np
from RANParser import RANParser

class CarrierEnvLive(Env):
    def __init__(self):
        self.parser = RANParser()
        self.config = self.parser.parse_args()
        #look into the distribution of random arrays
        self.num_sbs = self.config.num_sbs
        self.total_ue = self.config.total_user
        self.num_ue = self.config.num_usr
        self.max_ue_per_sbs = self.config.max_num_usr
        self.min_ue_per_sbs = self.config.min_num_usr
        self.min_distance = self.config.min_distance
        self.max_distance = self.config.max_distance
        self.min_avg_datarate = self.config.avg_datarate_min
        self.max_avg_datarate = self.config.avg_datarate_max
        self.min_avg_power = self.config.avg_pwr_min
        self.max_avg_power = self.config.avg_pwr_max
        self.increasing = True
        self._max_episode_steps = 500
        # Fixed BS locations
        self.sbs_state = np.ones(self.num_sbs)  # All SBSs are initially on
        # Track number of users per SBS
        # Fixed BS locations
        self.bs_locations = self.generate_bs_locations()
    
        # Randomly generate user locations within the area
        self.user_locations = self.generate_user_locations()
       
        # Calculate distances and associate users with the nearest BS
        self.distances, self.user_associations = self.calculate_distances_and_associations()
        # Set up action and observation space
        self.action_space = Discrete(self.num_sbs * 3)  # Each action represents turning off one SBS
        self.state = self.get_observation_state()

         # Penalty and reward escalation counters
        self.penalty_counter = np.zeros(self.num_sbs)  # Track how often the agent makes a mistake for each SBS
        self.reward_counter = np.zeros(self.num_sbs)   # Track how often the agent makes the right decision for each SBS

    def reset_penalty_and_reward_counters(self):
        """Resets the penalty and reward counters at the start of each episode."""
        self.penalty_counter = np.zeros(self.num_sbs)
        self.reward_counter = np.zeros(self.num_sbs)
        # print(self.state)


    def step(self, action, current_episode):
        # Turn off the selected SBS
        # self.turn_off_sbs(action)
        sbs_index = action // 3  # Get SBS index
        sbs_action = action % 3  # Get action type (0 = off, 1 = on, 2 = do nothing)

        if sbs_action == 0 and self.sbs_state[sbs_index] == 1:
            # Turn off the SBS
            self.turn_off_sbs(sbs_index)
        elif sbs_action == 1 and self.sbs_state[sbs_index] == 0:
            # Turn on the SBS
            self.turn_on_sbs(sbs_index)

        # print('Action')
        # print(action)
        self.move_users()
            

        # # Dynamically adjust the total number of users (increase or decrease)
        # self.adjust_user_count()

        # self.reallocate_users()
        # if current_episode % 1000 == 0:
        #     self.reset_ue_count(current_episode)


        # Calculate the data rate and power based on the current load
        data_rate = self.calculate_load_based_data_rate()
        total_power = self.calculate_load_based_power()

        # Calculate energy efficiency
        energy_efficiency = np.sum(data_rate) / np.sum(total_power) if np.sum(total_power) > 0 else 0

        # Calculate reward based on energy efficiency, load, data rate, and power penalties
        reward, penalty = self.calculate_reward(energy_efficiency)

        # Update state based on user distribution
        self.state = self.get_observation_state()

        # Check if the episode is done (based on some condition like data rate thresholds)
        done = self.check_done_condition(data_rate)

        

        # Info object can contain additional metrics (optional)
        info = {
            'total_power': total_power,
            'energy_efficiency': energy_efficiency,
            'total_penalty': penalty
        }

        return self.state, reward, done, info
    

    def reset_ue_count(self, current_episode):
        """Reset the number of UEs to the min or max threshold every 1000 episodes."""
        min_ue_total = 2   # Minimum allowed UEs
        max_ue_total = 40  # Maximum allowed UEs

        # If the episode count is a multiple of 2000, reset to max UEs
        if current_episode % 2000 == 0:
            self.total_ue = max_ue_total
            self.user_locations = np.random.uniform(0, 100, (self.total_ue, 2))  # Recreate user locations
            print(f"Reset UEs to maximum threshold: {self.total_ue} UEs.")

        # If the episode count is a multiple of 1000 but not 2000, reset to min UEs
        elif current_episode % 1000 == 0:
            self.total_ue = min_ue_total
            self.user_locations = np.random.uniform(0, 100, (self.total_ue, 2))  # Recreate user locations
            print(f"Reset UEs to minimum threshold: {self.total_ue} UEs.")

        # Recalculate user associations after reset
        self.user_associations = np.zeros(self.total_ue, dtype=int)  # Reset user associations with the new size
        self.reallocate_users()


    def move_users(self):
        """Simulate user movement by adding random displacements to user locations."""
        num_users = self.user_locations.shape[0]  # Get the current number of users
        movement_noise = np.random.uniform(-2, 2, (num_users, 2))  # Small movement in a 2D space
        self.user_locations += movement_noise

        # Ensure users stay within the bounded area defined by min_distance and max_distance
        self.user_locations[:, 0] = np.clip(self.user_locations[:, 0], self.min_distance, self.max_distance)
        self.user_locations[:, 1] = np.clip(self.user_locations[:, 1], self.min_distance, self.max_distance)

        # Recalculate the distances and re-associate users to the nearest active SBS
        self.reallocate_users()


    # def reallocate_users(self):
    #     """Reassociate users to the nearest active SBS after they move."""
    #     active_sbs = np.where(self.sbs_state == 1)[0]  # Get indices of active SBSs
    #     if len(active_sbs) == 0:
    #         print("Error: No active SBS to re-associate users.")
    #         return  # Exit early if no SBSs are active
        
    #     if self.num_ue != len(self.user_associations):
    #         self.num_ue = len(self.user_associations)

    #     # Reassociate each user to the nearest active SBS
    #     for i in range(self.num_ue):
    #         # Find the distance to the active SBSs
    #         distances = np.linalg.norm(self.user_locations[i] - self.bs_locations[active_sbs], axis=1)
    #         # Get the nearest active SBS
    #         nearest_active_sbs = active_sbs[np.argmin(distances)]
    #         # Associate the user with the nearest active SBS
    #         self.user_associations[i] = nearest_active_sbs

    def reallocate_users(self):
        """Reassociate users to the nearest active SBS after they move."""
        active_sbs = np.where(self.sbs_state == 1)[0]  # Get indices of active SBSs
        if len(active_sbs) == 0:
            print("Error: No active SBS to re-associate users.")
            return  # Exit early if no SBSs are active

        # Reassociate each user to the nearest active SBS
        for i in range(len(self.user_associations)):  # Ensure we loop within the bounds of user_associations
            # Find the distance to the active SBSs
            distances = np.linalg.norm(self.user_locations[i] - self.bs_locations[active_sbs], axis=1)
            
            if len(distances) > 0:  # Ensure there are distances to compute
                nearest_active_sbs = active_sbs[np.argmin(distances)]
                if nearest_active_sbs < len(self.sbs_state):  # Ensure valid SBS index
                    # Associate the user with the nearest active SBS
                    self.user_associations[i] = nearest_active_sbs
                else:
                    print(f"Error: Invalid SBS index {nearest_active_sbs} for user {i}.")
            else:
                print(f"Error: No valid distances found for user {i}.")


    def adjust_user_count(self):
        """Increase or decrease the number of users dynamically based on max and min UE thresholds."""
        
        # Define minimum and maximum UE limits
        min_ue_total = 2   # Minimum allowed UEs
        max_ue_total = 40  # Maximum allowed UEs
        
        # Check if we are increasing or decreasing UEs
        if self.increasing:
            # Increase UE count until we reach the max threshold
            users_to_add = np.random.randint(1, 5)  # Randomly add 1-5 UEs
            if self.total_ue + users_to_add <= max_ue_total:
                self.add_users(users_to_add)
                print(f"Added {users_to_add} users. Total UEs: {self.total_ue}")
            else:
                # If we hit the max, stop increasing and start decreasing
                self.total_ue = max_ue_total
                self.increasing = False
                print(f"Reached max UE threshold: {self.total_ue} UEs. Switching to decrease mode.")
                # self.user_associations = np.zeros(self.total_ue, dtype=int)
        
        else:
            # Decrease UE count until we reach the min threshold
            users_to_remove = np.random.randint(1, 5)  # Randomly remove 1-5 UEs
            if self.total_ue - users_to_remove >= min_ue_total:
                self.remove_users(users_to_remove)
                print(f"Removed {users_to_remove} users. Total UEs: {self.total_ue}")
            else:
                # If we hit the min, stop decreasing and start increasing
                self.total_ue = min_ue_total
                self.increasing = True
                print(f"Reached min UE threshold: {self.total_ue} UEs. Switching to increase mode.")
                # self.user_associations = np.zeros(self.total_ue, dtype=int)

        # self.user_associations = np.zeros(self.total_ue, dtype=int)

    def add_users(self, num_users):
        """Add new users randomly into the environment and associate them with the nearest SBS."""
        new_user_locations = np.random.uniform(0, 100, (num_users, 2))
        self.user_locations = np.vstack([self.user_locations, new_user_locations])
        new_user_associations = np.zeros(num_users, dtype=int)

        for i in range(num_users):
            distances = np.linalg.norm(new_user_locations[i] - self.bs_locations, axis=1)
            nearest_sbs = np.argmin(distances)
            new_user_associations[i] = nearest_sbs

        self.user_associations = np.hstack([self.user_associations, new_user_associations])
        self.total_ue += num_users

    def remove_users(self, num_users):
        """Randomly remove users from the environment."""
        if self.total_ue > num_users:
            users_to_remove = np.random.choice(range(self.total_ue), num_users, replace=False)
            self.user_locations = np.delete(self.user_locations, users_to_remove, axis=0)
            self.user_associations = np.delete(self.user_associations, users_to_remove)
            self.total_ue -= num_users
        
    # def step(self, action):
    #     # Handle action (0: no handover, 1: handover)
       
    #     if action == 1:
    #         # handover_successful = self.handover_users()
    #         # if handover_successful:
    #         self.turn_off_idle_sbs()

    #     data_rate = self.get_unifrom_avg_cell_data_rate()
    #     total_power = self.calculate_total_power()
        
    #     energy_efficiency = data_rate / total_power
    #     reward = self.calculate_reward(energy_efficiency)

    #     self.state = self.get_observation_state()

    #     print('State')
    #     # print(data_rate)
    #     print(self.state)

    #     done = self.check_done_condition(data_rate)

    #     # Detect if a mistake was made
    #     mistake_detected = self.detect_mistake(action, data_rate, reward,self.user_associations,self.max_ue_per_sbs,self.min_ue_per_sbs)

    #     info = {
    #         'total_power': total_power,
    #         'energy_efficiency': energy_efficiency,
    #         'mistake_detected': mistake_detected
    #     }
    #     # Return step information
    #     return self.state, reward, done, info
    def calculate_load_based_data_rate(self):
        """
        Calculate data rate for each SBS based on the load (number of users).
        The more users, the lower the data rate.
        """
        data_rates = np.zeros(self.num_sbs)
        for sbs_index in range(self.num_sbs):
            load = np.sum(self.user_associations == sbs_index)
            if load > 0:
                # Data rate decreases as load increases
                data_rates[sbs_index] = self.max_avg_datarate / (1 + 0.05 * load)  # Adjust factor 0.05 as needed
            else:
                data_rates[sbs_index] = 0  # If no users, data rate is 0
        return data_rates
    
    def calculate_load_based_power(self):
        """
        Calculate power consumption based on the load (number of users).
        More users lead to higher power consumption.
        """
        power_consumption = np.zeros(self.num_sbs)
        for sbs_index in range(self.num_sbs):
            load = np.sum(self.user_associations == sbs_index)
            if load > 0:
                # Power consumption increases with load
                power_consumption[sbs_index] = self.min_avg_power + load * 2  # Each user adds 2 watts of consumption
            else:
                power_consumption[sbs_index] = 0  # If no users, power is 0 (SBS turned off)
        return power_consumption
    
    # def calculate_reward(self, energy_efficiency, data_rate, total_power):
    #     """
    #     Calculate reward based on energy efficiency and apply penalties based on load, data rate, and power.
    #     The higher the load, the larger the penalty; the lower the data rate, the larger the penalty; 
    #     inefficient power usage also leads to penalties.
    #     """
    #     load_penalty = 0
    #     data_rate_penalty = 0
    #     power_penalty = 0
    #     load_reward = 0

    #     for sbs_index in range(self.num_sbs):
    #         load = np.sum(self.user_associations == sbs_index)

    #         # Penalty if the load exceeds the max limit per SBS
    #         if load > self.max_ue_per_sbs:
    #             load_penalty += (load - self.max_ue_per_sbs) * 2.0  # Penalty for overloading SBS
    #         else:
    #             # Reward if turning on SBS helped reduce load
    #             if self.sbs_state[sbs_index] == 1 and load < self.max_ue_per_sbs:
    #                 load_reward += 10
            
    #         # Penalty if the data rate falls below the minimum threshold
    #         if data_rate[sbs_index] < self.min_avg_datarate:
    #             data_rate_penalty += (self.min_avg_datarate - data_rate[sbs_index]) * 2.0  # More severe penalty for data rate issues

    #         # Penalty if power consumption is inefficient (too high for the data rate)
    #         if total_power[sbs_index] > self.max_avg_power:
    #             power_penalty += (total_power[sbs_index] - self.max_avg_power) * 0.5  # Penalty for power overuse

    #     # Total penalty is a combination of load, data rate, and power penalties
    #     total_penalty = load_penalty + data_rate_penalty + power_penalty
        
    #     # Reward is based on energy efficiency minus the penalties
    #     reward = energy_efficiency - total_penalty + load_reward

    #     return reward, total_penalty
    
    # def calculate_reward(self, energy_efficiency, data_rate, total_power):
    #     """
    #     Calculate reward and penalties based on SBS load and energy efficiency.
    #     """
    #     load_penalty = 0
    #     data_rate_penalty = 0
    #     power_penalty = 0
    #     load_reward = 0
    #     sbs_on_off_penalty = 0

    #     total_active_users = np.sum([np.sum(self.user_associations == cell_index) for cell_index in range(self.num_sbs)])

    #     for sbs_index in range(self.num_sbs):
    #         load = np.sum(self.user_associations == sbs_index)

    #         # Penalty if load exceeds max UEs (20 UEs max)
    #         if load > self.max_ue_per_sbs:
    #             load_penalty += (load - self.max_ue_per_sbs) * 2.0  # Penalty for overloading SBS
    #         else:
    #             if self.sbs_state[sbs_index] == 1 and load < self.max_ue_per_sbs:
    #                 load_reward += 10
    #             elif  self.sbs_state[sbs_index] == 1 and load < self.min_ue_per_sbs:
    #                 sbs_on_off_penalty += (self.min_ue_per_sbs - load) * 1.5

    #         # **Update**: Penalty for turning off SBS when there are more than 20 UEs
    #         if self.sbs_state[sbs_index] == 0 and load > self.max_ue_per_sbs:
    #             sbs_on_off_penalty += (load - self.max_ue_per_sbs) * 2.0

    #     # Scenario: Turn off SBS if total UEs < 20, turn on SBS if total UEs ≥ 20
    #     if total_active_users < 20:
    #         for sbs_index in range(self.num_sbs):
    #             if self.sbs_state[sbs_index] == 1:
    #                 sbs_on_off_penalty += 5  # Penalty for keeping unnecessary SBS on

    #     if total_active_users >= 20:
    #         if np.sum(self.sbs_state) < 2:
    #             sbs_on_off_penalty += 10  # Penalty for not turning on enough SBSs

    #     total_penalty = load_penalty + data_rate_penalty + power_penalty + sbs_on_off_penalty
    #     reward = energy_efficiency - total_penalty + load_reward
    #     return reward, total_penalty
    
    def calculate_reward(self, energy_efficiency):
        """
        Calculate reward and penalties based on SBS load, underloading, and overloading.
        The penalties and rewards increase if the agent keeps making the same mistake or correct decision.
        """
        load_penalty = 0
        load_reward = 0
        sbs_on_off_penalty = 0

        total_active_users = np.sum([np.sum(self.user_associations == cell_index) for cell_index in range(self.num_sbs)])

        for sbs_index in range(self.num_sbs):
            load = np.sum(self.user_associations == sbs_index)

            # **Condition 1: Penalize SBS if it's ON and has fewer than 5 UEs (Underload)**
            if load < 5 and self.sbs_state[sbs_index] == 1:
                self.penalty_counter[sbs_index] += 1  # Increase penalty counter
                penalty_multiplier = self.penalty_counter[sbs_index]  # Escalating penalty for repeated mistakes
                load_penalty += penalty_multiplier * 10  # Increasing penalty for keeping underloaded SBS ON

            else:
                self.penalty_counter[sbs_index] = 0  # Reset penalty counter if the mistake is corrected

            # **Condition 2: Severe penalty if turning off SBS causes overloading of others**
            if self.sbs_state[sbs_index] == 0:
                # Get the load on all SBSs that are currently active
                active_sbs_loads = [np.sum(self.user_associations == idx) for idx in range(self.num_sbs) if self.sbs_state[idx] == 1]
                total_active_load = np.sum(active_sbs_loads)
                active_sbs_count = np.sum(self.sbs_state)

                # Check if any of the active SBSs exceed their capacity
                if active_sbs_count > 0 and total_active_load > active_sbs_count * self.max_ue_per_sbs:
                    self.penalty_counter[sbs_index] += 1  # Increase penalty counter
                    penalty_multiplier = self.penalty_counter[sbs_index]  # Escalating penalty
                    sbs_on_off_penalty += penalty_multiplier * 50  # Severe penalty for overloading active SBSs
                else:
                    self.penalty_counter[sbs_index] = 0  # Reset penalty counter if no overload

            # **Condition 3: Penalty if SBS has more than 20 UEs (Overload)**
            if load > self.max_ue_per_sbs:
                self.penalty_counter[sbs_index] += 1  # Increase penalty counter
                penalty_multiplier = self.penalty_counter[sbs_index]  # Escalating penalty
                load_penalty += penalty_multiplier * (load - self.max_ue_per_sbs) * 5.0  # Higher penalty for overloading SBS
            else:
                self.penalty_counter[sbs_index] = 0  # Reset penalty counter if load is managed properly

            # **Condition 4: Reward for efficiently managing load on active SBSs**
            if self.sbs_state[sbs_index] == 1 and load <= self.max_ue_per_sbs and load >= 5:
                self.reward_counter[sbs_index] += 1  # Increase reward counter
                reward_multiplier = self.reward_counter[sbs_index]  # Escalating reward for consecutive good decisions
                load_reward += reward_multiplier * 2  # Increasing reward for keeping load within limits
            else:
                self.reward_counter[sbs_index] = 0  # Reset reward counter if not efficient

            # **Condition 5: Ensure the load on an active SBS is never zero**
            if self.sbs_state[sbs_index] == 1 and load == 0:
                self.penalty_counter[sbs_index] += 1  # Increase penalty counter
                penalty_multiplier = self.penalty_counter[sbs_index]  # Escalating penalty
                load_penalty += penalty_multiplier * 10  # Penalty for having an active SBS with no UEs assigned

        # **Global Penalty**: If total UEs exceed 20 and fewer than 2 SBSs are active
        if total_active_users >= 20 and np.sum(self.sbs_state) < 2:
            sbs_on_off_penalty += 20  # Penalty for not turning on enough SBSs to handle the load

        total_penalty = load_penalty + sbs_on_off_penalty
        reward = energy_efficiency - total_penalty + load_reward

        return reward, total_penalty

    def get_observation_state(self):
        """
        Get the current state based on the number of users per SBS,
        the data rate, and the power consumption.
        """
        num_active_users = [
        np.sum(self.user_associations == cell_index) if self.sbs_state[cell_index] == 1 else 0
        for cell_index in range(self.num_sbs)
        ]   
        data_rate = [
        self.calculate_load_based_data_rate()[cell_index] if self.sbs_state[cell_index] == 1 else 0
        for cell_index in range(self.num_sbs)
        ]

        # Calculate the power consumption for each SBS. If the SBS is off, set the power consumption to 0
        total_power = [
            self.calculate_load_based_power()[cell_index] if self.sbs_state[cell_index] == 1 else 0
            for cell_index in range(self.num_sbs)
        ]
        sbs_status = self.sbs_state
        # total_power = self.calculate_load_based_power()
        state = np.concatenate((num_active_users, sbs_status, data_rate, total_power))
        return state

    def check_done_condition(self, data_rate):
        """Check if the episode should end based solely on average throughput (data rate)."""
        
        # Calculate the average throughput across all SBSs
        avg_throughput = np.mean(data_rate)
        print(f"Average throughput: {avg_throughput}")

        # Done condition: End episode if throughput falls between min and max demand thresholds
        if self.config.demand_min <= avg_throughput <= self.config.demand_max:
            print(f"Done: Average throughput is within the acceptable range: {avg_throughput}")
            return True

        # Otherwise, the episode is not done
        return False



    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.user_locations = self.generate_user_locations()
        # self.increasing = True
        # Calculate distances and associate users with the nearest cell
        self.distances, self.user_associations = self.calculate_distances_and_associations()
        self.state = self.get_observation_state()

        self.reset_penalty_and_reward_counters()
        return self.state

    # def turn_off_sbs(self, sbs_index):
    #     """Turn off a specified SBS and reassign its users to other active SBSs."""
    #     if np.sum(self.sbs_state) > 1:  # Only turn off if more than one SBS is active
    #         if self.sbs_state[sbs_index] == 1:  # If SBS is on, turn it off
    #             self.sbs_state[sbs_index] = 0
    #             # Reassign users who were associated with the SBS that is being turned off
    #             users_to_reassign = np.where(self.user_associations == sbs_index)[0]
    #             for user in users_to_reassign:
    #                 # Get active SBSs
    #                 active_sbs = np.where(self.sbs_state == 1)[0]
    #                 if len(active_sbs) > 0:
    #                     # Find the nearest active SBS for each user
    #                     distances = np.linalg.norm(self.user_locations[user] - self.bs_locations[active_sbs], axis=1)
    #                     nearest_active_sbs = active_sbs[np.argmin(distances)]
    #                     # Reassign the user to the nearest active SBS
    #                     self.user_associations[user] = nearest_active_sbs
    #     else:
    #         print(f"Cannot turn off SBS {sbs_index}, as it is the last active SBS.")

    def turn_off_sbs(self, sbs_index):
        """Turn off a specified SBS and reassign its users to other active SBSs."""
        if np.sum(self.sbs_state) > 1:  # Only turn off if more than one SBS is active
            if self.sbs_state[sbs_index] == 1:  # If SBS is on, turn it off
                self.sbs_state[sbs_index] = 0
                # Reassign users who were associated with the SBS that is being turned off
                users_to_reassign = np.where(self.user_associations == sbs_index)[0]
                for user in users_to_reassign:
                    # Get active SBSs
                    active_sbs = np.where(self.sbs_state == 1)[0]
                    if len(active_sbs) > 0:
                        # Find the nearest active SBS for each user
                        distances = np.linalg.norm(self.user_locations[user] - self.bs_locations[active_sbs], axis=1)
                        nearest_active_sbs = active_sbs[np.argmin(distances)]
                        # Reassign the user to the nearest active SBS
                        self.user_associations[user] = nearest_active_sbs
                    else:
                        print(f"Error: No active SBS to reassign UE {user}.")
        else:
            print(f"Cannot turn off SBS {sbs_index}, as it is the last active SBS.")

    def turn_on_sbs(self, sbs_index):
        """Turn on a specified SBS and potentially reassign users."""
        if self.sbs_state[sbs_index] == 0:  # Only turn on if it's currently off
            self.sbs_state[sbs_index] = 1  # Turn on the SBS
            
            # Optionally reassign users to the newly turned-on SBS based on proximity
            for user in range(self.num_ue):
                distance_to_new_sbs = np.linalg.norm(self.user_locations[user] - self.bs_locations[sbs_index])
                current_sbs = self.user_associations[user]
                distance_to_current_sbs = np.linalg.norm(self.user_locations[user] - self.bs_locations[current_sbs])
                
                if distance_to_new_sbs < distance_to_current_sbs:
                    self.user_associations[user] = sbs_index  # Reassign user to the new SBS

    # def check_done_condition(self, data_rate):
    #     """
    #     Checks if the done condition is met for each BS based on its average data rate.
    #     Returns a numpy array where each element indicates whether the condition is met for the corresponding BS.
    #     """
    #     # Initialize the done array with False
    #     done_array = np.zeros(self.num_sbs, dtype=bool)

    #     # Check if any base station's average data rate falls below the minimum threshold
    #     for cell_index in range(self.num_sbs):
    #         if data_rate[cell_index] > self.config.demand_min and data_rate[cell_index] < self.config.demand_max :
    #             done_array[cell_index] = True

    #     return done_array


    def generate_bs_locations(self):
        # Assuming the cells are placed uniformly in a grid within the area
        grid_size = int(np.ceil(np.sqrt(self.num_sbs)))
        cell_locations = []
        for i in range(self.num_sbs):
            x = (i % grid_size) * (self.max_distance - self.min_distance) / grid_size + self.min_distance
            y = (i // grid_size) * (self.max_distance - self.min_distance) / grid_size + self.min_distance
            cell_locations.append([x, y])
        return np.array(cell_locations)
    
    def generate_user_locations(self):
        user_locations = []
        for _ in range(self.num_sbs):
            cell_user_locations = np.random.uniform(low=self.min_distance, high=self.max_distance, size=(self.num_ue, 2))
            user_locations.append(cell_user_locations)
        return np.vstack(user_locations)
    
    # def calculate_distances_and_associations(self):
    #     distances = np.zeros((self.num_ue, self.num_sbs))
    #     user_associations = np.zeros(self.num_ue, dtype=int)

    #     for ue_index in range(self.num_ue):
    #         for cell_index in range(self.num_sbs):
    #             distances[ue_index, cell_index] = np.linalg.norm(self.user_locations[ue_index] - self.bs_locations[cell_index])
    #         nearest_cell = np.argmin(distances[ue_index])
    #         user_associations[ue_index] = nearest_cell

    #     return distances, user_associations
    

    def calculate_distances_and_associations(self):
        """
        Calculate distances between UEs and SBSs, and associate UEs to SBSs ensuring that
        no SBS exceeds the maximum limit of 20 UEs.
        """
        distances = np.zeros((self.num_ue, self.num_sbs))
        user_associations = np.zeros(self.num_ue, dtype=int)
        sbs_loads = np.zeros(self.num_sbs, dtype=int)  # Track load on each SBS (number of UEs)

        for ue_index in range(self.num_ue):
            # Calculate distances between this UE and all SBSs
            for cell_index in range(self.num_sbs):
                distances[ue_index, cell_index] = np.linalg.norm(self.user_locations[ue_index] - self.bs_locations[cell_index])
            
            # Sort SBS indices based on proximity (nearest to farthest)
            nearest_sbs_indices = np.argsort(distances[ue_index])

            # Try to assign this UE to the nearest SBS that has space (fewer than 20 UEs)
            for sbs_index in nearest_sbs_indices:
                if sbs_loads[sbs_index] < self.max_ue_per_sbs:
                    # Assign the UE to this SBS if it has space
                    user_associations[ue_index] = sbs_index
                    sbs_loads[sbs_index] += 1  # Update the SBS load
                    break
                else:
                    # If no SBS has space, handle the case (could be an error or log it)
                    print(f"UE {ue_index} could not be assigned to any SBS because all are full.")

        return distances, user_associations

    # def calculate_total_power(self):
    #     return np.random.uniform(self.min_avg_power, self.max_avg_power, self.num_sbs)


    # def reset(self):
    #     # Randomly generate user locations within the area
    #     self.user_locations = self.generate_user_locations()

    #     # Calculate distances and associate users with the nearest cell
    #     self.distances, self.user_associations = self.calculate_distances_and_associations()

    #     self.state =  self.get_observation_state() # All SBSs are initially on
    #     self.sbs_state = np.ones(self.num_sbs)
    #     return self.state
    

    # def get_unifrom_avg_cell_data_rate(self):
    #     return np.random.uniform(self.min_avg_datarate, self.max_avg_datarate, self.num_sbs)
    
    # # def calculate_energy_efficiency(self):
    # #     _,_, total_data_rate = self.calculate_data_rate()
    # #     _,_,total_power = self.calculate_total_power()
    # #     EE = total_data_rate / total_power
    # #     return EE
    

    # #check this reward function if it makes sense or is feasible 
    # def calculate_reward(self, EE):
    #     Xj, Xk = self.check_constraints()
    #     U = len(self.user_associations)
    #     C = self.num_sbs
    #     penalty = EE * ((1/U) * sum(Xj) + (1/C) * sum(Xk))
    #     print('Penalty to the AI')
    #     print(penalty)
    #     return EE - penalty

    # def check_constraints(self):
    #     # Initialize constraint violation indicators
    #     Xj = np.zeros(len(self.user_associations), dtype=int)  # For users
    #     Xk = np.zeros(self.num_sbs, dtype=int)  # For cells

    #     # Calculate average data rate for each cell
    #     data_rate = self.get_unifrom_avg_cell_data_rate()
        
    #     # Initialize a list to count user allocations per cell
    #     user_allocations = [[] for _ in range(self.num_sbs)]
        
    #     # Populate user allocations based on user associations
    #     for ue_index, cell_index in enumerate(self.user_associations):
    #         user_allocations[cell_index].append(ue_index)
        
    #     # Check constraints
    #     for cell_index in range(self.num_sbs):
    #         # Check data rate constraint
    #         if data_rate[cell_index] < self.config.demand_min:
    #             for ue_index in user_allocations[cell_index]:
    #                 Xj[ue_index] = 1
            
    #         # Check maximum number of users constraint
    #         if len(user_allocations[cell_index]) > self.config.max_num_usr:
    #             Xk[cell_index] = 1

    #     return Xj, Xk

    
    # def turn_off_idle_sbs(self):
    #     for sbs in range(self.num_sbs):
    #         if np.sum(self.user_associations == sbs) == 0:
    #             self.sbs_state[sbs] = 0
    #             self.done = True

    # def get_observation_state(self):
    #     num_active_users = [np.sum(self.user_associations == cell_index) for cell_index in range(self.num_sbs)]
    #     num_active_users_flat = np.array(num_active_users).flatten()
    #     data_rate = self.get_unifrom_avg_cell_data_rate()
    #     data_rate_flattened = data_rate.flatten()
    #     total_power = self.calculate_total_power()
    #     total_power_flattened = total_power.flatten()
    #     state =  np.concatenate((num_active_users_flat, data_rate_flattened, total_power_flattened))
    #     return state
    
    # def detect_mistake(self, actions, data_rate, reward, user_associations, load_threshold_high, load_threshold_low):
    #     """
    #     Detect if the agent has made a mistake based on its action, data rate, reward, and cell load.
    #     Returns True if a mistake is detected, False otherwise.
    #     """

    #     # Initialize mistake flag
    #     mistake_detected = False

    #     # Calculate the load of each cell (number of users associated with each cell)
    #     cell_loads = np.zeros(self.num_sbs, dtype=int)
    #     for ue_index, cell_index in enumerate(user_associations):
    #         cell_loads[cell_index] += 1

    #     # Iterate over each cell and check for mistakes based on load and action
    #     for cell_index in range(self.num_sbs):
    #         # action = actions[cell_index]  # Action for the current cell
            
    #         # Check data rate condition
    #         if data_rate[cell_index] < self.config.demand_min:
    #             mistake_detected = True  # Mistake detected: data rate constraint violated
            
    #         # Action validation based on cell load
    #         if actions == 1:  # Action to turn off the cell
    #             if cell_loads[cell_index] > load_threshold_high:  # Cell load is too high to turn off
    #                 mistake_detected = True  # Mistake detected: trying to turn off a heavily loaded cell

    #         if actions == 0:  # Action to keep the cell on
    #             if cell_loads[cell_index] < load_threshold_low:  # Cell load is too low to keep on
    #                 mistake_detected = True  # Mistake detected: keeping a lightly loaded cell on

    #     # Example 2: Check if the reward is below a certain threshold
    #     if np.any(reward < 0):  # You can set a custom threshold for reward
    #         mistake_detected = True  # Mistake detected: reward is too low

    #     return mistake_detected

