# Project: Adaptive Traffic Signal Control using Deep Q-Learning (DQN)
# Description: This script implements a Deep Q-Network (DQN) agent to learn
#              optimal traffic light phasing in a SUMO simulation environment.
#              The agent's goal is to minimize overall traffic congestion,
#              measured primarily by accumulated vehicle waiting time.
#              It compares the DQN's performance against baseline Fixed-Time
#              and Q-Learning (Q-table) strategies.


# Step 1: Imports
# -------------------------
import os, sys, random, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xml.etree.ElementTree as ET
from collections import deque # For the replay buffer

# -------------------------
# Step 2: SUMO_HOME and Traci
# -------------------------
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

# -------------------------
# Step 3: SUMO config & Output File Paths
# -------------------------
TRIPINFO_OUTPUT_FILE = "tripinfos_dqn_extended_0.5s_step_global_waiting_reward.xml" # Changed filename
EMISSION_OUTPUT_FILE = "emission_output_dqn_extended_0.5s_step_global_waiting_reward.xml" # Changed filename

Sumo_config = [
    'sumo',
    '-c', 'osm.sumocfg',
    '--step-length', '0.5', # *** CONSISTENT 0.5s STEP-LENGTH ***
    '--delay', '1000',
    '--lateral-resolution', '0',
    '--seed', '42',
    '--tripinfo-output', TRIPINFO_OUTPUT_FILE,
    '--emission-output', EMISSION_OUTPUT_FILE
]
traci.start(Sumo_config)
# traci.gui.setSchema("View #0", "real world") # Only active if sumo-gui is used

# -------------------------
# Step 4: TLS & Detectors
# -------------------------
TLS_IDS = ["IKOTUN_IN", "IKOTUN_OUT", "OKOTA_OUT", "ISOLO_J", "R2"]
DETECTOR_IDS = {
    "IKOTUN_IN": ["DET_J5_IKOTUN_IN_0"],
    "IKOTUN_OUT": ["DET_GATE_IKOTUN_OUT_0"],
    "OKOTA_OUT": ["DET_O_OKOTA_OUT_0"],
    "ISOLO_J": ["DET_ISOLO_ISOLOJ_0"],
    "R2": ["DET_J8_R2_0"]
}

# -------------------------
# Step 5: Hyperparameters
# -------------------------
TOTAL_STEPS = 36000 # This now aims for 18000 simulated seconds (5 hours)
ALPHA, GAMMA = 0.1, 0.9 # ALPHA (learning rate) is now managed by Adam optimizer
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.99999 # Slower decay rate
EPSILON = EPSILON_START

ACTIONS = [0, 1] # 0: keep phase, 1: switch phase
MIN_GREEN_STEPS = 180 # *** CORRECTED for 0.5s step-length (90 simulated seconds of green) ***

# Initialize last_switch dictionary
last_switch = {tls: -MIN_GREEN_STEPS for tls in TLS_IDS}

# DQN Specific Hyperparameters
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 500
TRAIN_FREQ = 10

# State representation: (discretized_queue_length, current_phase_index)
state_size = 2 # (discretized_queue, phase_index)
action_size = len(ACTIONS)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Step 6: Build DQN Models
# -------------------------
def build_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='linear') # Linear activation for Q-values
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model

# Policy Network (DQN)
dqn_model = build_model(state_size, action_size)
# Target Network (for stable Q-value targets)
target_dqn_model = build_model(state_size, action_size)
target_dqn_model.set_weights(dqn_model.get_weights())

# -------------------------
# Step 7: Helper Functions
# -------------------------
# Experience Replay Buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

def to_array(state):
    return np.array(state, dtype=np.float32).reshape((1, -1))

def get_queue_length(det):
    try:
        # Using getLastStepHaltingNumber for number of stopped vehicles at the detector
        return traci.lanearea.getLastStepHaltingNumber(det)
    except:
        return 0

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state(tls_id):
    detectors = DETECTOR_IDS[tls_id]
    raw_queue = sum(get_queue_length(d) for d in detectors)
    phase = get_current_phase(tls_id)

    # --- Queue Discretization (for state representation, same as before) ---
    if raw_queue <= 5:
        queue_bin = 0 # Low queue
    elif raw_queue <= 15:
        queue_bin = 1 # Medium queue
    else:
        queue_bin = 2 # High queue
    
    return (queue_bin, phase)

# --- NEW GLOBAL REWARD FUNCTION: Negative Sum of Accumulated Waiting Time ---
def get_global_reward_from_waiting_time():
    total_waiting_time = 0.0
    vehicle_ids = traci.vehicle.getIDList()
    
    # Iterate over all vehicles to sum their accumulated waiting times
    for veh_id in vehicle_ids:
        try:
            # getAccumulatedWaitingTime returns the total time (in seconds)
            # a vehicle has been standing (speed < 0.1 m/s) since its departure.
            total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh_id)
        except traci.exceptions.TraCIException:
            # Handle cases where vehicle might have departed/arrived between calls
            pass
            
    # Reward is the negative sum of waiting times.
    # Lower total waiting time -> less negative reward (better)
    # The agent is incentivized to minimize waiting.
    return -total_waiting_time

def choose_action(state):
    global EPSILON
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    
    Qs = dqn_model.predict(to_array(state), verbose=0)[0]
    return int(np.argmax(Qs))

def apply_action(tls_id, action, step):
    if action == 1 and step - last_switch[tls_id] >= MIN_GREEN_STEPS:
        try:
            current_program = traci.trafficlight.getProgram(tls_id)
            current_phase = traci.trafficlight.getPhase(tls_id)
            
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0] 
            next_phase = (current_phase + 1) % len(logic.phases)
            
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch[tls_id] = step 
        except Exception as e:
            pass 

def train_dqn_batch():
    if len(replay_buffer) < BATCH_SIZE:
        return

    minibatch = random.sample(replay_buffer, BATCH_SIZE)

    states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
    actions = np.array([experience[1] for experience in minibatch])
    rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
    next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)

    current_q_values = dqn_model.predict(states, verbose=0)
    future_q_values = target_dqn_model.predict(next_states, verbose=0)
    
    target_q_values = np.copy(current_q_values)
    
    batch_indices = np.arange(BATCH_SIZE)
    target_q_values[batch_indices, actions] = rewards + GAMMA * np.max(future_q_values, axis=1)

    dqn_model.fit(states, target_q_values, batch_size=BATCH_SIZE, verbose=0)


# --- Analysis Functions (remain the same) ---
def process_tripinfo_output(filename):
    print(f"\n--- Analyzing Tripinfo Output: {filename} ---")
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        total_trips = 0
        total_travel_time = 0.0
        total_speed = 0.0

        for tripinfo in root.findall('tripinfo'):
            total_trips += 1
            duration = float(tripinfo.get('duration'))
            route_length = float(tripinfo.get('routeLength'))
            
            total_travel_time += duration
            
            if duration > 0:
                total_speed += (route_length / duration)

        if total_trips > 0:
            avg_travel_time = total_travel_time / total_trips
            avg_speed = total_speed / total_trips
            print(f"Total Number of Trips Completed: {total_trips}")
            print(f"Average Travel Time: {avg_travel_time:.2f} seconds")
            print(f"Average Speed: {avg_speed:.2f} m/s")
        else:
            print("No trips completed in the tripinfo file.")

    except FileNotFoundError:
        print(f"Error: Tripinfo file '{filename}' not found.")
    except Exception as e:
        print(f"Error processing tripinfo file: {e}")

def process_emission_output(filename):
    print(f"\n--- Analyzing Emission Output: {filename} ---")
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        total_co2 = 0.0
        total_co = 0.0
        total_hc = 0.0
        total_nox = 0.0
        total_pmx = 0.0
        total_fuel = 0.0
        total_noise = 0.0

        for emission_step in root.findall('timestep'):
            for vehicle in emission_step.findall('vehicle'):
                total_co2 += float(vehicle.get('CO2'))
                total_co += float(vehicle.get('CO'))
                total_hc += float(vehicle.get('HC'))
                total_nox += float(vehicle.get('NOx'))
                total_pmx += float(vehicle.get('PMx'))
                total_fuel += float(vehicle.get('fuel'))
                total_noise += float(vehicle.get('noise'))

        print(f"Total CO2: {total_co2:.2f} mg")
        print(f"Total CO: {total_co:.2f} mg")
        print(f"Total HC: {total_hc:.2f} mg")
        print(f"Total NOx: {total_nox:.2f} mg")
        print(f"Total PMx: {total_pmx:.2f} mg")
        print(f"Total fuel: {total_fuel:.2f} mL")
        print(f"Total noise: {total_noise:.2f} dB")

    except FileNotFoundError:
        print(f"Error: Emission output file '{filename}' not found.")
    except Exception as e:
        print(f"Error processing emission output file: {e}")


# -------------------------
# Step 8: Learning Loop
# -------------------------
step_hist, rew_hist, q_hist = [], [], []
cum_rew = 0.0
print("📡 Starting Deep Q-Learning loop with GLOBAL REWARD (negative sum of ACCUMULATED WAITING TIME)...")

step = 0
while step < TOTAL_STEPS and traci.simulation.getMinExpectedNumber() > 0:
    total_q_current_step = 0 
    
    # Decay epsilon
    EPSILON = max(EPSILON_END, EPSILON * EPSILON_DECAY_RATE)

    # Store current states before actions are applied
    old_states = {tls: get_state(tls) for tls in TLS_IDS}
    
    # Action phase for all TLS
    actions = {tls: choose_action(old_states[tls]) for tls in TLS_IDS}
    for tls in TLS_IDS:
        apply_action(tls, actions[tls], step) # Apply actions before advancing simulation

    traci.simulationStep() # Advance SUMO simulation

    # --- Calculate Global Reward AFTER the step ---
    # This reward applies to all agents for the actions taken in this step
    global_reward_for_step = get_global_reward_from_waiting_time()

    # Learning phase: Collect new states and rewards, store experiences
    for tls in TLS_IDS:
        new_state = get_state(tls)
        
        # All agents receive the same global reward for the actions taken in this step
        # For simplicity, each experience in the replay buffer gets the full global reward.
        # In more complex multi-agent setups, this reward might be divided or scaled.
        reward = global_reward_for_step 

        # Store the experience in the replay buffer
        replay_buffer.append((old_states[tls], actions[tls], reward, new_state))
        
        total_q_current_step += new_state[0] # Still track discretized queue for plotting/monitoring

    # Train policy network from replay buffer
    if len(replay_buffer) >= BATCH_SIZE and step % TRAIN_FREQ == 0:
        train_dqn_batch()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_dqn_model.set_weights(dqn_model.get_weights())
        # print(f"--- Target Network Updated at Step {step} ---") 

    # Update cumulative reward and history for plotting
    # The cumulative reward should reflect the total global reward for the network
    cum_rew += global_reward_for_step 
    
    if step % 100 == 0:
        step_hist.append(step)
        rew_hist.append(cum_rew)
        q_hist.append(total_q_current_step)
        print(f"Step {step}: TotalQueue={total_q_current_step}, CumReward={cum_rew:.2f}, Epsilon={EPSILON:.4f}, BufferSize={len(replay_buffer)}")

    step += 1

traci.close()
print("✅ Simulation finished. Model trained.")

# -------------------------
# Step 9: Analysis and Visualization
# -------------------------
process_tripinfo_output(TRIPINFO_OUTPUT_FILE)
process_emission_output(EMISSION_OUTPUT_FILE)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(step_hist, rew_hist, label='Cumulative Reward')
plt.xlabel('Step'); plt.ylabel('Reward'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(step_hist, q_hist, label='Total Queue')
plt.xlabel('Step'); plt.ylabel('Queue'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()