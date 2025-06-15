# Fully Online Q-learning Traffic Signal Control for Your TLS and Detectors

# Step 1: Imports
import os, sys, random, numpy as np, matplotlib.pyplot as plt
import xml.etree.ElementTree as ET # For parsing XML output files

# Step 2: SUMO_HOME and Traci
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

# Step 3: SUMO config
Sumo_config = [
    'sumo-gui', '-c', 'osm.sumocfg',
    '--step-length', '0.5',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Output file names (must match what's configured in your 'osm.sumocfg' for outputs)
TRIPINFO_OUTPUT_FILE = 'tripinfos.xml'
EMISSION_OUTPUT_FILE = 'emission_output.xml'

try:
    traci.start(Sumo_config)
    traci.gui.setSchema("View #0", "real world")
    print(f"Successfully connected to SUMO using config: {Sumo_config[2]}")
except traci.exceptions.TraCIException as e:
    print(f"ERROR: Failed to start SUMO or connect to TraCI: {e}")
    print(f"Please ensure SUMO_HOME is set correctly and '{Sumo_config[2]}' is valid and accessible.")
    sys.exit(1)


# Step 4: TLS & Detectors
TLS_IDS = ["IKOTUN_IN", "IKOTUN_OUT", "OKOTA_OUT", "ISOLO_J", "R2"]
# DETECTOR_IDS now correctly maps TLS_ID to its relevant detectors
DETECTOR_IDS = {
    "IKOTUN_IN": ["DET_J5_IKOTUN_IN_0"],
    "IKOTUN_OUT": ["DET_GATE_IKOTUN_OUT_0"],
    "OKOTA_OUT": ["DET_O_OKOTA_OUT_0"], # Using DET_O_OKOTA_OUT_0 as provided in your script
    "ISOLO_J": ["DET_ISOLO_ISOLOJ_0"],
    "R2": ["DET_J8_R2_0"]
}

# Step 5: Hyperparameters
TOTAL_STEPS = 7200
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.9
ACTIONS = [0, 1] # 0 = keep phase, 1 = switch to next phase

# Adjusted MIN_GREEN_STEPS based on 0.5s step length (e.g., 90 steps * 0.5s/step = 45 simulated seconds)
MIN_GREEN_STEPS = 90

Q_table = {}
last_switch = {tls: -MIN_GREEN_STEPS for tls in TLS_IDS}

# Global cache for TLS program logics to minimize TraCI calls
# Populated after traci.start
TLS_PROGRAM_LOGICS_CACHE = {}
for tls_id in TLS_IDS:
    try:
        # Assuming we only care about the first (default) program logic for simple cycling
        # If your TLS has multiple programs and you need to select them, this logic would need adjustment.
        TLS_PROGRAM_LOGICS_CACHE[tls_id] = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    except Exception as e:
        print(f"Warning: Could not get program logic for TLS '{tls_id}': {e}. This TLS might not switch phases.")
        TLS_PROGRAM_LOGICS_CACHE[tls_id] = None # Mark as unavailable


# Step 6: Helper Functions

def get_queue_length(det):
    """Returns the current queue length (number of vehicles) from a specified SUMO E2 detector."""
    try:
        return traci.lanearea.getLastStepVehicleNumber(det)
    except traci.exceptions.TraCIException:
        # This exception might occur if the detector ID is invalid or not yet known to SUMO
        # It's better to catch the specific TraCI exception.
        return 0 # if detector is missing

def get_current_phase(tls_id):
    """Returns the current phase index of a specified SUMO traffic light."""
    try:
        return traci.trafficlight.getPhase(tls_id)
    except traci.exceptions.TraCIException:
        # Handle cases where TLS ID might not be valid (e.g., typo)
        return -1 # Indicate an error or invalid phase

def get_state(tls_id):
    """
    Captures the current state for a specific TLS.
    The state is a tuple combining:
    1. Queue lengths from its assigned detectors.
    2. Its current phase index.
    """
    detectors = DETECTOR_IDS.get(tls_id, []) # Use .get() to avoid KeyError if TLS_ID not in DETECTOR_IDS
    queues = [get_queue_length(d) for d in detectors]
    phase = get_current_phase(tls_id)
    return tuple(queues + [phase]) # state = (q1, q2, ..., phase)

def reward_fn(state):
    """
    Calculates the reward based on the state's queue lengths.
    Reward is the negative sum of queue lengths, penalizing congestion.
    """
    # The 'state' tuple contains queue lengths first, then the phase.
    # Sum only the queue lengths (all elements except the last one, which is phase).
    return -sum(state[:-1]) # penalize total queue length for THIS TLS

def max_Q(s):
    """Retrieves the maximum Q-value for a given state, initializing if new."""
    Q_table.setdefault(s, np.zeros(len(ACTIONS)))
    return np.max(Q_table[s])

def update_Q(s, a, r, ns):
    """Updates the Q-table using the Q-Learning formula."""
    Q_table.setdefault(s, np.zeros(len(ACTIONS)))
    Q_table[s][a] += ALPHA * (r + GAMMA * max_Q(ns) - Q_table[s][a])

def choose_action(s):
    """Selects an action based on the epsilon-greedy policy."""
    if random.random() < EPSILON: # Exploration
        return random.choice(ACTIONS)
    Q_table.setdefault(s, np.zeros(len(ACTIONS))) # Initialize if state is new
    return int(np.argmax(Q_table[s])) # Exploitation

def apply_action(tls_id, action, step):
    """Applies the chosen action to a specific traffic light, respecting MIN_GREEN_STEPS."""
    global last_switch # Access the global dictionary to track last switch times

    if action == 1: # If the agent decides to switch phases
        # Check if the minimum green time has passed for this specific TLS
        if step - last_switch[tls_id] >= MIN_GREEN_STEPS:
            try:
                current_phase = traci.trafficlight.getPhase(tls_id)
                # Use cached program logic
                logic = TLS_PROGRAM_LOGICS_CACHE.get(tls_id)
                if logic and logic.phases:
                    num_phases = len(logic.phases)
                    next_phase = (current_phase + 1) % num_phases # Cycle to the next phase
                    traci.trafficlight.setPhase(tls_id, next_phase) # Set the new phase
                    last_switch[tls_id] = step # Update the last switch step for this TLS
                else:
                    print(f"⚠️ TLS '{tls_id}' has no valid program logic or phases in cache. Cannot switch phase.")
            except traci.exceptions.TraCIException as e:
                print(f"⚠️ TLS switch failed for {tls_id}: {e}")
            except Exception as e:
                print(f"⚠️ An unexpected error occurred while switching TLS {tls_id}: {e}")
    elif action == 0: # If the agent decides to keep the current phases
        pass # Do nothing, phases naturally continue


# Step 7: Learning Loop (with early stop and network-wide metrics)
step_hist, rew_hist, q_hist = [], [], []
total_cumulative_network_reward = 0.0 # Accumulates reward for the entire network over time

print("\n📡 Starting Q-learning control loop...")
print(f"Simulation configured for {TOTAL_STEPS} steps with step length {Sumo_config[5]}s.")
print(f"Controlling Traffic Lights: {', '.join(TLS_IDS)}")
print(f"Monitoring Detectors: {', '.join([det for sublist in DETECTOR_IDS.values() for det in sublist])}") # List all detectors
print(f"RL Hyperparameters: ALPHA={ALPHA}, GAMMA={GAMMA}, EPSILON={EPSILON}, MIN_GREEN_STEPS={MIN_GREEN_STEPS}")
print("Note: Initial learning might show high congestion due to exploration.")


step = 0
# Continue as long as current step is less than total steps AND there are still vehicles in the network
while step < TOTAL_STEPS and traci.simulation.getMinExpectedNumber() > 0:
    current_step_network_queue = 0 # Total queue length for ALL detectors in the network for THIS step
    current_step_network_reward = 0.0 # Total reward accumulated from ALL TLS for THIS step

    # Dictionary to store (old_state, action) pairs for each TLS before simulation step
    tls_states_actions_pre_step = {}

    for tls_id in TLS_IDS:
        s = get_state(tls_id) # Get state for current TLS
        a = choose_action(s)  # Choose action for current TLS
        apply_action(tls_id, a, step) # Apply action to current TLS
        tls_states_actions_pre_step[tls_id] = (s, a) # Store (old_state, action)

    traci.simulationStep() # Advance the SUMO simulation by one step

    # After simulation step, get new states and rewards for all TLS
    for tls_id in TLS_IDS:
        old_s, action_taken = tls_states_actions_pre_step[tls_id]
        new_s = get_state(tls_id) # Get new state for current TLS
        reward_for_tls = reward_fn(new_s) # Calculate reward for current TLS

        update_Q(old_s, action_taken, reward_for_tls, new_s) # Update Q-table for THIS TLS's experience

        # Accumulate network-wide metrics for the current step
        current_step_network_reward += reward_for_tls
        # Sum all queue lengths from the new_s tuple (excluding the last element which is the phase)
        current_step_network_queue += sum(new_s[:-1])

    # Accumulate total network reward over the entire simulation
    total_cumulative_network_reward += current_step_network_reward

    # Data Recording and Console Output (Optimized Frequency)
    if step % 100 == 0:
        step_hist.append(step)
        rew_hist.append(total_cumulative_network_reward) # Store network-wide cumulative reward
        q_hist.append(current_step_network_queue) # Store network-wide total queue for this step
        print(f"Step {step}: TotalNetworkQueue={current_step_network_queue}, TotalNetworkCumReward={total_cumulative_network_reward:.2f}")

    step += 1

traci.close()
print("✅ Simulation finished. Q-table size:", len(Q_table))

# Step 8: Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(step_hist, rew_hist, label='Cumulative Reward')
plt.xlabel('Simulation Step'); plt.ylabel('Cumulative Reward (Negative Total Queue)'); plt.legend(); plt.grid(True)
plt.title("Q-Learning: Cumulative Reward During Training")

plt.subplot(1, 2, 2)
plt.plot(step_hist, q_hist, label='Total Network Queue'); plt.xlabel('Simulation Step')
plt.ylabel('Total Queue Length (Vehicles)'); plt.legend(); plt.grid(True)
plt.title("Q-Learning: Total Network Queue Length During Training")

plt.tight_layout()
plt.show()

# Step 9: Process Tripinfo Output
def process_tripinfo_output(filename: str, sumo_config_file: str):
    """
    Parses the tripinfo XML file to calculate average travel time, average speed,
    and total number of completed trips.
    """
    total_travel_time = 0.0
    total_travel_speed = 0.0
    completed_trips = 0

    print(f"\n--- Analyzing Tripinfo Output: {filename} ---")
    if not os.path.exists(filename):
        print(f"Error: Tripinfo output file '{filename}' not found. Please ensure it's generated by SUMO.")
        print(f"Check the <output> section of your '{sumo_config_file}'.")
        return

    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        for tripinfo in root.findall('tripinfo'):
            duration_str = tripinfo.get('duration')
            route_length_str = tripinfo.get('routeLength')

            if duration_str is not None and route_length_str is not None:
                duration = float(duration_str)
                route_length = float(route_length_str)

                if duration > 0: # Ensure duration is positive to avoid division by zero
                    total_travel_time += duration
                    speed = route_length / duration # Speed in m/s
                    total_travel_speed += speed
                    completed_trips += 1

        print(f"Total Number of Trips Completed: {completed_trips}")

        if completed_trips > 0:
            avg_travel_time = total_travel_time / completed_trips
            avg_travel_speed = total_travel_speed / completed_trips
            print(f"Average Travel Time: {avg_travel_time:.2f} seconds")
            print(f"Average Speed: {avg_travel_speed:.2f} m/s")
        else:
            print("No completed trips found to calculate average metrics.")

    except ET.ParseError as e:
        print(f"Error parsing tripinfo XML file '{filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during tripinfo processing: {e}")

# Step 10: Process Emission Output
def process_emission_output(filename: str, sumo_config_file: str):
    """
    Parses the emission XML file to calculate total emissions of various pollutants.
    """
    total_emissions = {
        'CO2': 0.0, 'CO': 0.0, 'HC': 0.0, 'NOx': 0.0,
        'PMx': 0.0, 'fuel': 0.0, 'noise': 0.0
    }

    print(f"\n--- Analyzing Emission Output: {filename} ---")
    if not os.path.exists(filename):
        print(f"Error: Emission output file '{filename}' not found. Please ensure it's generated by SUMO.")
        print(f"Check the <output> section of your '{sumo_config_file}'.")
        return

    try:
        root = ET.parse(filename).getroot()
        for timestep_element in root.findall('timestep'):
            for vehicle_element in timestep_element.findall('vehicle'):
                for pollutant in total_emissions.keys():
                    value_str = vehicle_element.get(pollutant)
                    if value_str is not None:
                        total_emissions[pollutant] += float(value_str)

        for pollutant, total_value in total_emissions.items():
            unit = "mg" # Default unit
            if pollutant == 'fuel': unit = "mL"
            elif pollutant == 'noise': unit = "dB"
            print(f"Total {pollutant}: {total_value:.2f} {unit}")
    except ET.ParseError as e:
        print(f"Error parsing emission XML file '{filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during emission processing: {e}")

# Call analysis functions after simulation ends
process_tripinfo_output(TRIPINFO_OUTPUT_FILE, Sumo_config[2]) # Pass the actual config file name
process_emission_output(EMISSION_OUTPUT_FILE, Sumo_config[2]) # Pass the actual config file name

print("\n--- Simulation and Analysis Complete ---")