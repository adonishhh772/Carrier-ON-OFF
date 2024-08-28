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
        self.max_ue_per_sbs = self.config.max_num_usr
        self.min_ue_per_sbs = self.config.min_num_usr
        self.min_distance = self.config.min_distance
        self.max_distance = self.config.max_distance
        self.min_avg_datarate = self.config.avg_datarate_min
        self.max_avg_datarate = self.config.avg_datarate_max
        self.min_avg_power = self.config.avg_pwr_min
        self.max_avg_power = self.config.avg_pwr_max

        # Fixed BS locations
        self.bs_locations = self.generate_bs_locations()
        # Randomly generate user locations within the area
        self.user_locations = self.generate_user_locations()
       
        # Calculate distances and associate users with the nearest BS
        self.distances, self.user_associations = self.calculate_distances_and_associations()

        self.action_space = Discrete(self.num_sbs)
        self.sbs_state = np.ones(self.num_sbs)
        self.state = self.get_observation_state()
        # print(self.state)

        
    def step(self, action):
        # Handle action (0: no handover, 1: handover)
       
        if action == 1:
            # handover_successful = self.handover_users()
            # if handover_successful:
            self.turn_off_idle_sbs()

        data_rate = self.get_unifrom_avg_cell_data_rate()
        total_power = self.calculate_total_power()
        
        energy_efficiency = data_rate / total_power
        reward = self.calculate_reward(energy_efficiency)

        self.state = self.get_observation_state()

        print('Action')
        print(action)
        print(self.state)

        done = self.check_done_condition(data_rate)

        # Detect if a mistake was made
        mistake_detected = self.detect_mistake(action, data_rate, reward,self.user_associations,self.max_ue_per_sbs,self.min_ue_per_sbs)

        info = {
            'total_power': total_power,
            'energy_efficiency': energy_efficiency,
            'mistake_detected': mistake_detected
        }
        # Return step information
        return self.state, reward, done, info

    def check_done_condition(self, data_rate):
        """
        Checks if the done condition is met for each BS based on its average data rate.
        Returns a numpy array where each element indicates whether the condition is met for the corresponding BS.
        """
        # Initialize the done array with False
        done_array = np.zeros(self.num_sbs, dtype=bool)

        # Check if any base station's average data rate falls below the minimum threshold
        for cell_index in range(self.num_sbs):
            if data_rate[cell_index] < self.config.demand_min:
                done_array[cell_index] = True

        return done_array


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
            cell_user_locations = np.random.uniform(low=self.min_distance, high=self.max_distance, size=(self.total_ue, 2))
            user_locations.append(cell_user_locations)
        return np.vstack(user_locations)
    
    def calculate_distances_and_associations(self):
        distances = np.zeros((self.total_ue, self.num_sbs))
        user_associations = np.zeros(self.total_ue, dtype=int)

        for ue_index in range(self.total_ue):
            for cell_index in range(self.num_sbs):
                distances[ue_index, cell_index] = np.linalg.norm(self.user_locations[ue_index] - self.bs_locations[cell_index])
            nearest_cell = np.argmin(distances[ue_index])
            user_associations[ue_index] = nearest_cell

        return distances, user_associations
    
    def calculate_total_power(self):
        return np.random.uniform(self.min_avg_power, self.max_avg_power, self.num_sbs)


    def reset(self):
        # Randomly generate user locations within the area
        self.user_locations = self.generate_user_locations()

        # Calculate distances and associate users with the nearest cell
        self.distances, self.user_associations = self.calculate_distances_and_associations()

        self.state =  self.get_observation_state() # All SBSs are initially on
        self.sbs_state = np.ones(self.num_sbs)
        return self.state
    

    def get_unifrom_avg_cell_data_rate(self):
        return np.random.uniform(self.min_avg_datarate, self.max_avg_datarate, self.num_sbs)
    
    # def calculate_energy_efficiency(self):
    #     _,_, total_data_rate = self.calculate_data_rate()
    #     _,_,total_power = self.calculate_total_power()
    #     EE = total_data_rate / total_power
    #     return EE
    

    #check this reward function if it makes sense or is feasible 
    def calculate_reward(self, EE):
        Xj, Xk = self.check_constraints()
        U = len(self.user_associations)
        C = self.num_sbs
        penalty = EE * ((1/U) * sum(Xj) + (1/C) * sum(Xk))
        print('Penalty to the AI')
        print(penalty)
        return EE - penalty

    def check_constraints(self):
        # Initialize constraint violation indicators
        Xj = np.zeros(len(self.user_associations), dtype=int)  # For users
        Xk = np.zeros(self.num_sbs, dtype=int)  # For cells

        # Calculate average data rate for each cell
        data_rate = self.get_unifrom_avg_cell_data_rate()
        
        # Initialize a list to count user allocations per cell
        user_allocations = [[] for _ in range(self.num_sbs)]
        
        # Populate user allocations based on user associations
        for ue_index, cell_index in enumerate(self.user_associations):
            user_allocations[cell_index].append(ue_index)
        
        # Check constraints
        for cell_index in range(self.num_sbs):
            # Check data rate constraint
            if data_rate[cell_index] < self.config.demand_min:
                for ue_index in user_allocations[cell_index]:
                    Xj[ue_index] = 1
            
            # Check maximum number of users constraint
            if len(user_allocations[cell_index]) > self.config.max_num_usr:
                Xk[cell_index] = 1

        return Xj, Xk

    
    def turn_off_idle_sbs(self):
        for sbs in range(self.num_sbs):
            if np.sum(self.user_associations == sbs) == 0:
                self.sbs_state[sbs] = 0
                self.done = True

    def get_observation_state(self):
        num_active_users = [np.sum(self.user_associations == cell_index) for cell_index in range(self.num_sbs)]
        num_active_users_flat = np.array(num_active_users).flatten()
        data_rate = self.get_unifrom_avg_cell_data_rate()
        data_rate_flattened = data_rate.flatten()
        total_power = self.calculate_total_power()
        total_power_flattened = total_power.flatten()
        state =  np.concatenate((num_active_users_flat, data_rate_flattened, total_power_flattened))
        return state
    
    def detect_mistake(self, actions, data_rate, reward, user_associations, load_threshold_high, load_threshold_low):
        """
        Detect if the agent has made a mistake based on its action, data rate, reward, and cell load.
        Returns True if a mistake is detected, False otherwise.
        """

        # Initialize mistake flag
        mistake_detected = False

        # Calculate the load of each cell (number of users associated with each cell)
        cell_loads = np.zeros(self.num_sbs, dtype=int)
        for ue_index, cell_index in enumerate(user_associations):
            cell_loads[cell_index] += 1

        # Iterate over each cell and check for mistakes based on load and action
        for cell_index in range(self.num_sbs):
            # action = actions[cell_index]  # Action for the current cell
            
            # Check data rate condition
            if data_rate[cell_index] < self.config.demand_min:
                mistake_detected = True  # Mistake detected: data rate constraint violated
            
            # Action validation based on cell load
            if actions == 1:  # Action to turn off the cell
                if cell_loads[cell_index] > load_threshold_high:  # Cell load is too high to turn off
                    mistake_detected = True  # Mistake detected: trying to turn off a heavily loaded cell

            if actions == 0:  # Action to keep the cell on
                if cell_loads[cell_index] < load_threshold_low:  # Cell load is too low to keep on
                    mistake_detected = True  # Mistake detected: keeping a lightly loaded cell on

        # Example 2: Check if the reward is below a certain threshold
        if np.any(reward < 0):  # You can set a custom threshold for reward
            mistake_detected = True  # Mistake detected: reward is too low

        return mistake_detected

