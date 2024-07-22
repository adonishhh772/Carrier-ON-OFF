from gym import Env
from gym.spaces import Discrete, Box
import random
import math
from matplotlib import pyplot as plt
import numpy as np
from RANParser import RANParser

class CarrierEnv(Env):
    def __init__(self):
        self.parser = RANParser()
        self.config = self.parser.parse_args()
        self.beta_t = 1
        self.gTx = self.config.tx_gain
        self.gRx = self.config.rx_gain

        #look into the distribution of random arrays
        self.wavelength = self.config.wavelength
        self.path_loss_exponent = self.config.path_loss
        self.num_sbs = self.config.num_sbs
        self.total_ue = self.config.total_user

        # self.num_ue = self.config.num_usr
        self.num_chnl = self.config.num_channel
        self.num_time_slots = 1
        self.diameter = self.config.diameter
        self.min_sinr_threshold = self.config.min_sinr
        self.max_ue_per_sbs = self.config.max_num_usr
        self.beta = 1
        # self.distance = self.randomize_distances()

        self.max_trnsm_power = self.config.max_transm_power
        self.min_trnsm_power = self.config.min_transm_power
        self.min_distance = self.config.min_distance
        self.max_distance = self.config.max_distance
        self.noise_power = self.config.noise_power
        self.B = self.config.band
        self.max_sinr = []
        self.max_datarate = []
        self.ALL_POWER = []

        self.d0 = self.generate_far_field_distance()


        
        # Fixed BS locations
        self.bs_locations = self.generate_bs_locations()
    
        # Randomly generate user locations within the area
        self.user_locations = self.generate_user_locations()
       
        # Calculate distances and associate users with the nearest BS
        self.distances, self.user_associations = self.calculate_distances_and_associations()

       
        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels()
        self.p = self.calculate_transmission_power()

        self.action_space = Discrete(2)
        self.sbs_state = np.ones(self.num_sbs)
        self.state = self.get_observation_state()
        # print(self.state)
       

        

        
    def step(self, action,count):
        # Handle action (0: no handover, 1: handover)
        if action == 1:
            # handover_successful = self.handover_users()
            # if handover_successful:
            self.turn_off_idle_sbs()

        _,_, total_data_rate = self.calculate_data_rate()
        total_data_rate_mb = self.convert_datarate(total_data_rate)
        active_power, sleep_power, total_power = self.calculate_total_power()
        
        energy_efficiency = total_data_rate_mb / total_power
        reward = self.calculate_reward(energy_efficiency)
        done = np.all(self.calculate_done())
       
        if done:
            self.plot_max('SINR')
            self.plot_max('Data')
            self.plot_max('Power')
        # else:
        #     if count > 300:
        #         done = True
           
        
        # if not done and count > 300:
        #     done = True

        self.state = self.get_observation_state()

        info = {
            'active_power': active_power,
            'sleep_power': sleep_power,
            'total_power': total_power,
            'energy_efficiency': energy_efficiency
        }
        return self.state, reward, done, info
        
        # Return step information
        # return self.state, reward, done, info

    def plot_max(self,plot_type):
        if plot_type == 'SINR':
            plot_name = 'MAX SINR'
            self.save_plot(self.max_sinr,' SINR (db)',plot_name)
        if plot_type == 'Data':
            plot_name = 'MAX DATARATE'
            self.save_plot(self.max_datarate,' Data Rate (mb)',plot_name)
        if plot_type == 'Power':
            plot_name = 'Power Consumption'
            self.save_plot(self.ALL_POWER,'Power Consumption',plot_name)


    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Randomly generate user locations within the area
        self.user_locations = self.generate_user_locations()

        # Calculate distances and associate users with the nearest cell
        self.distances, self.user_associations = self.calculate_distances_and_associations()

        self.g = self.calculate_channel_gain()
        self.alpha = self.allocate_channels()
        self.p = self.calculate_transmission_power()

        self.state =  self.get_observation_state() # All SBSs are initially on
        self.sbs_state = np.ones(self.num_sbs)
        return self.state
    

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
    
    def generate_far_field_distance(self):
        # Generate a random antenna dimension (D) between 0.1 meters and 3 meters
        wavelength = self.wavelength
        # Calculate the far-field distance using the formula R >= 2D^2 / λ
        far_field_distance = (2 * self.diameter**2) / wavelength
        return far_field_distance

   
    def calculate_transmission_power(self):
        transmission_powers = []
        for cell_index in range(self.num_sbs):
            num_ue_in_cell = np.sum(self.user_associations == cell_index)
            transmission_powers.append(np.random.uniform(low=self.min_trnsm_power, high=self.max_trnsm_power, size=num_ue_in_cell))
        return transmission_powers
        

    def calculate_channel_gain(self):
        gains = []
        for cell_index in range(self.num_sbs):
            gTx_linear = 10 ** (self.gTx / 10)       # Convert dB to linear scale if necessary
            gRx_linear = 10 ** (self.gRx / 10)
            ue_indices = np.where(self.user_associations == cell_index)[0]
            distances = self.distances[ue_indices, cell_index]
            numerator = self.beta_t * gTx_linear * gRx_linear * (self.wavelength ** 2)
            denominator = 16 * (math.pi ** 2) * ((distances / self.d0) ** self.path_loss_exponent)
            gains.append(numerator / denominator)

        return gains
      
    def convert_datarate(self,datarate):
        """
        Convert bits to megabits.
        
        Parameters:
        bits (int or float): Number of bits to be converted.
        
        Returns:
        float: Number of megabits.
        """
        mb_arr = [bits / (1024 * 1024) for bits in datarate]
        return mb_arr
    
    
    def allocate_channels(self):
        # Static allocation: all UEs are served using the same channel
        alpha = []
        for cell_index in range(self.num_sbs):
            num_ue_in_cell = np.sum(self.user_associations == cell_index)
            ue_alpha = np.ones((num_ue_in_cell, 1), dtype=int)
            alpha.append(ue_alpha)
        return alpha

    # the r is the channel resource allocation which is not used as we are considering the ue are allocated in the same channel
    def calculate_sinr(self,c,u):
         # Convert power levels from dBm to linear scale (mW)
        p_c_u_linear = 10 ** (self.p[c][u] / 10)
        g_c_u_linear = self.g[c][u]
        
        # Calculate the desired signal power (in mW)
        numerator = self.beta * p_c_u_linear * g_c_u_linear
        
        interference = 0.0
        # for i in range(self.num_sbs):
        #     if i != c:
        #         interfering_ue_indices = np.where(self.user_associations == i)[0]
        #         for j in interfering_ue_indices:
        #             if self.alpha[i][j % len(self.alpha[i]), 0] == 1:
        #                 p_i_j_linear = 10 ** (self.p[i][j % len(self.alpha[i])] / 10)
        #                 g_i_j_linear = self.g[i][j % len(self.alpha[i])]
        #                 interference += self.beta * p_i_j_linear * g_i_j_linear

        # Convert noise power from dBm to linear scale (mW)
        noise_power_linear = 10 ** (self.noise_power / 10)
        
        # Calculate the denominator (interference + noise, in mW)
        denominator = interference + noise_power_linear
        sinr = numerator / denominator
        # Optionally convert SINR to dB for better interpretability
        sinr_db =  np.log10(sinr)
        return sinr_db


    def calculate_data_rate(self):
        data_rate = []
        sinr_ue = []
        # sbs_data_rate = []
        total_data_rate = np.zeros(self.num_sbs)
        for c in range(self.num_sbs):
            sbs_data_rate = np.zeros((len(self.alpha[c]), 1))
            sinr_values = np.zeros((len(self.alpha[c]), 1))
            for u in range(len(self.alpha[c])):
                # if self.alpha[c][u] == 1:
                sinr = self.calculate_sinr(c, u)
                sinr_values[u,0] = sinr
                sbs_data_rate[u,0] = self.B * np.log2(1 + sinr)
                        # sinr_val.append(sinr)
            data_rate.append(sbs_data_rate)
            sinr_ue.append(sinr_values)
            total_data_rate[c] = np.sum(sbs_data_rate)
        return data_rate,sinr_ue, total_data_rate

    def calculate_total_power(self):
        active_power = 0
        sleep_power = 0
        #need to recheck this logic for total power consumption for active mode and sleep mode
        for i in range(self.num_sbs):
            if np.sum(self.alpha[i]) > 0:
                active_power += self.sbs_state[i] * (np.sum(self.alpha[i] * self.p[i][:, np.newaxis]) + self.config.power_active)
            else:
                sleep_power += self.config.power_sleep
        total_power = active_power + sleep_power
        return active_power, sleep_power, total_power
    
    def generate_random_base_station_power(self):
        # Power ranges in watts
        min_power_watts = self.config.min_power  # Small cells and indoor base stations
        max_power_watts = self.config.max_power # High-power base stations
        
        # Generate a random power value in watts
        random_power_watts = np.random.uniform(min_power_watts, max_power_watts,size=(self.num_sbs))
        return random_power_watts
    

    #check this one as well if this or the above one makes more sense, which one makes more makes sense
    def calculate_base_station_power_consumption(self, efficiency=0.2, fixed_power=100):
        snr = self.calculate_sinr()
        # Calculate the signal power (S) using the SNR and noise power (N)
        signal_power = snr * self.noise_power()
        # Calculate the transmission power (P_tx) using the signal power (S) and channel gain (G)
        transmission_power = signal_power / self.calculate_channel_gain()
        # Calculate the total power consumption (P_total) of the base station
        total_power_consumption = (transmission_power / efficiency) + fixed_power
        return total_power_consumption

    def calculate_energy_efficiency(self):
        _,_, total_data_rate = self.calculate_data_rate()
        _,_,total_power = self.calculate_total_power()
        EE = total_data_rate / total_power
        return EE
    

    #check this reward function if it makes sense or is feasible 
    def calculate_reward(self, EE):
        Xj, Xk = self.check_constraints()
        U = len(self.user_associations)
        C = self.num_sbs
        penalty = EE * ((1/U) * sum(Xj) + (1/C) * sum(Xk))
        return EE - penalty

    def check_constraints(self):
        Xj = np.zeros(len(self.user_associations), dtype=int)
        Xk = np.zeros(self.num_sbs, dtype=int)
        
        data_rate, _,_ = self.calculate_data_rate()
        for cell_index in range(self.num_sbs):
            for ue_index in range(len(self.alpha[cell_index])):
                if data_rate[cell_index][ue_index].sum() < self.config.demand_min:
                    Xj[ue_index] = 1
            if len(self.alpha[cell_index]) > self.config.max_num_usr:
                Xk[cell_index] = 1

        return Xj, Xk
    
    
    def calculate_done(self):
        """
        Calculate the 'done' signal for each SBS based on the described logic.
        
        Returns:
        np.ndarray: Boolean array indicating whether 'done' is True or False for each SBS.
        """
        done = np.zeros(self.num_sbs, dtype=bool)
        for i in range(self.num_sbs):
            # Get the indices of users associated with the current SBS
            associated_users = np.where(self.user_associations == i)[0]
    
            # Calculate SINR for associated users
            if len(associated_users) != 0:
                sinr_values = [self.calculate_sinr(i, u) for u in range(0,len(associated_users))]
                if len(sinr_values) != 0:
                    # Calculate the average SINR for the current SBS
                    sinr = np.mean(sinr_values)
                    
                    # Check if the QoS requirement is met
                    done[i] = self.check_qos(sinr)
            else:
                done[i] = True
        return done


    
    def check_qos(self, sinr):
        """
        Check if all UEs associated with an SBS can meet their minimum required QoS based on SINR.
        Parameters:
        sinr (float): SINR value for the SBS.
        Returns:
        bool: True if all UEs can meet QoS, False otherwise.
        """
        # Replace with your QoS evaluation logic based on SINR
        return sinr >= self.min_sinr_threshold
    

    #need to look into the distance between UE and SBS for handing over as well. Only handover if it is in the minumum distance range.
    def handover_users(self):
        handover_successful = False
        for sbs_from in range(self.num_sbs):
            if self.sbs_state[sbs_from] == 1 and np.sum(self.alpha[sbs_from]) > 0:
                for u in range(len(self.alpha[sbs_from])):
                    if np.sum(self.alpha[sbs_from][u]) > 0:
                        for sbs_to in range(self.num_sbs):
                            if sbs_from != sbs_to and self.sbs_state[sbs_to] == 1:
                                ue_index = self.num_ue[sbs_from][u]
                                distance_to_new_bs = self.distances[ue_index, sbs_to]
                                if distance_to_new_bs < self.min_distance and len(self.alpha[sbs_to]) < self.max_ue_per_sbs:
                                    sinr_to = self.calculate_sinr(sbs_to, u)
                                    # sinr_to = np.mean([self.calculate_sinr(sbs_to, u, r) for r in range(self.num_chnl) if u < len(self.alpha[sbs_to])])
                                    data_rate_to = np.mean([self.B * np.log2(1 + sinr_to)])
                                    if sinr_to >= self.min_sinr_threshold and data_rate_to >= self.config.demand_min:
                                        self.alpha[sbs_to][u] = self.alpha[sbs_from][u]
                                        self.alpha[sbs_from][u] = 0
                                        handover_successful = True
                                        break
                        if handover_successful:
                            break
        return handover_successful

    def turn_off_idle_sbs(self):
        for sbs in range(self.num_sbs):
            if np.sum(self.alpha[sbs]) == 0:
                self.sbs_state[sbs] = 0

    def get_observation_state(self):
        num_active_users = [np.sum(self.user_associations == cell_index) for cell_index in range(self.num_sbs)]
        num_active_users_flat = np.array(num_active_users).flatten()
        print('Num USer')
        print(num_active_users_flat)
        self.g = self.calculate_channel_gain()
        self.p = self.calculate_transmission_power()
        data_rate,sinr, _ = self.calculate_data_rate()
        data_rate_mb = self.convert_datarate(data_rate)
        flattened_data = np.concatenate([array.flatten() for array in data_rate_mb])
        data_rate_flattened = flattened_data
        sinr_values = np.concatenate([array.flatten() for array in sinr])
        _, _, total_power = self.calculate_total_power()
        self.max_sinr.append(np.max(sinr_values,axis=0))
        self.max_datarate.append(np.max(data_rate_flattened,axis=0))
        self.ALL_POWER.append(total_power)
        # print(sinr_values)
        # print(data_rate_flattened)
        # self.save_plot(sinr_values,data_rate_flattened,'Data Rate(Mb)',' SINR (db)','SINR VS DATA RATE')
        state =  np.concatenate((num_active_users_flat, data_rate_flattened, sinr_values, [total_power]))
        return state
    
    def save_plot(self, x, label_x, plot_name):
        plt.figure(figsize=(10, 6))
        # plt.xlabel(label_x, fontsize=22)
        plt.ylabel(label_x, fontsize=22)
        plt.plot(np.arange(len(x)), x)
        plt.grid(True)
        plt.title('SINR')
        plot_name = plot_name + '.png'
        plt.savefig(plot_name)

