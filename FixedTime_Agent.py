# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt # Visualization
import xml.etree.ElementTree as ET # NEW: For parsing XML output files

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci # Static network information (such as reading and analyzing network files)

# -------------------------
# Step 4: Configuration Parameters
# -------------------------

# SUMO Simulation Configuration
SUMO_GUI = True # Set to False for non-GUI (headless) simulation
SUMO_CONFIG_FILE = 'osm.sumocfg' # Your SUMO configuration file
STEP_LENGTH = 0.5 # Simulation step length in seconds
GUI_DELAY = 1000 # Milliseconds delay for GUI updates (1000ms = 1s delay per step)
ADDITIONAL_FILE = 'osm.add.xml' # Your additional file for detectors
TRIPINFO_OUTPUT_FILE = 'tripinfos.xml' # NEW: Name of the tripinfo output file
EMISSION_OUTPUT_FILE = 'emission_output.xml' # ADDED: Name of the emission output file

# Traffic Light IDs to be controlled/monitored
TLS_IDS = ["OKOTA_OUT", "ISOLO_J", "R2", "IKOTUN_IN", "IKOTUN_OUT"]

# Detector IDs (corresponding to the osm.add.xml file)
# These IDs are used to read real-time queue data from the simulation.
DETECTOR_IDS = [
    "DET_O_OKOTA_OUT_0",
    "DET_ISOLO_ISOLOJ_0",
    "DET_J8_R2_0",
    "DET_J5_IKOTUN_IN_0",
    "DET_GATE_IKOTUN_OUT_0"
]

# Reinforcement Learning Hyperparameters (currently inactive for fixed timing monitoring)
TOTAL_SIMULATION_STEPS = 7200 # The total number of simulation steps
ALPHA = 0.1          # Learning rate (α)
GAMMA = 0.9          # Discount factor (γ)
EPSILON = 0.1        # Exploration rate (ε) - set to 0.0 for fixed timing
ACTIONS = [0, 1]     # The discrete action space (0 = keep phase, 1 = switch phase)

# Q-table dictionary: key = state tuple, value = numpy array of Q-values for each action
# This table will remain largely empty as Q-Learning is inactive.
Q_table = {}

# Additional Stability Parameters (currently inactive for fixed timing)
MIN_GREEN_STEPS = 90
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 5: SUMO Simulation Setup
# -------------------------

# Construct Sumo command based on configuration
sumo_cmd = ['sumo-gui' if SUMO_GUI else 'sumo',
            '-c', SUMO_CONFIG_FILE,
            '--step-length', str(STEP_LENGTH),
            '--lateral-resolution', '0']

# Add delay only for GUI mode to observe the simulation visually
if SUMO_GUI:
    sumo_cmd.extend(['--delay', str(GUI_DELAY)])

# Open connection between SUMO and Traci
try:
    traci.start(sumo_cmd)
    # Set the GUI schema for better visualization if running with GUI
    if SUMO_GUI:
        traci.gui.setSchema("View #0", "real world")
except traci.exceptions.TraCIException as e:
    print(f"ERROR: Failed to start SUMO or connect to TraCI: {e}")
    print("Please ensure SUMO_HOME is set correctly and your SUMO configuration file ('{SUMO_CONFIG_FILE}') is valid.")
    print("Also, check if another SUMO instance is already running on the same port.")
    sys.exit(1)


# -------------------------
# Step 6: Define Helper Functions
# -------------------------

def get_queue_length(detector_id: str) -> int:
    """Returns the current queue length (number of vehicles) from a specified SUMO E2 detector."""
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id: str) -> int:
    """Returns the current phase index of a specified SUMO traffic light."""
    return traci.trafficlight.getPhase(tls_id)

def get_state() -> tuple:
    """
    Captures the current state of the system for monitoring.
    State includes:
    - Queue lengths from all defined detectors (e.g., [q1, q2, q3, q4, q5]).
    - Current phase index of all defined traffic lights (e.g., [p1, p2, p3, p4, p5]).
    The state is returned as a combined tuple (q1, ..., qN, p1, ..., pM).
    """
    # Collect queue data from all specified detectors
    queue_data = [get_queue_length(det_id) for det_id in DETECTOR_IDS]
    # Collect current phase from all specified traffic lights
    phase_data = [get_current_phase(tls_id) for tls_id in TLS_IDS]
    # Combine queue lengths and phases into a single state tuple
    return tuple(queue_data + phase_data)


def get_reward(state: tuple) -> float:
    """
    Calculates the reward based on the current state.
    For fixed-time monitoring, this helps quantify performance.
    Reward is the negative sum of all queue lengths, encouraging shorter queues.
    """
    # The state tuple contains queue lengths followed by phase indices
    num_tls_phases = len(TLS_IDS)
    # Sum only the queue components from the state tuple
    total_queue = sum(state[:-num_tls_phases])
    return -float(total_queue)

# --- Q-Learning specific functions (currently inactive, included for template completeness) ---
def get_max_Q_value_of_state(s: tuple) -> float:
    """Returns the maximum Q-value for a given state from the Q-table (used in Q-Learning update)."""
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def update_Q_table(old_state: tuple, action: int, reward: float, new_state: tuple):
    """
    Updates the Q-table using the Q-Learning formula.
    This function is currently inactive in the main simulation loop.
    """
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))

    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state: tuple) -> int:
    """
    Selects an action based on the epsilon-greedy policy.
    This function is currently inactive in the main simulation loop.
    """
    if random.random() < EPSILON: # Exploration
        return random.choice(ACTIONS)
    else: # Exploitation
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return np.argmax(Q_table[state]) # Choose action with highest Q-value

def apply_action(action: int, current_step: int):
    """
    Applies the chosen action to the traffic lights.
    This function is currently designed for a single TLS_ID and is inactive.
    It will need significant modification for multi-TLS control when RL is activated.
    """
    # This function would contain the logic to change traffic light phases based on 'action'.
    # For fixed-time monitoring, no action is applied by the script.
    pass # No action applied as RL is commented out

# -------------------------
# Step 7: Simulation Loop (Fixed Timing Monitoring)
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting SUMO Fixed Timing Simulation (Monitoring Oke Afa Roundabout) ===")
print("Monitoring queues on:", ", ".join(DETECTOR_IDS))
print("Monitoring TLS phases for:", ", ".join(TLS_IDS))
print(f"Simulation configured for {TOTAL_SIMULATION_STEPS} steps.")


for step in range(TOTAL_SIMULATION_STEPS):
    traci.simulationStep()  # Advance the SUMO simulation by one step
    
    new_state = get_state() # Always get the new state to monitor conditions
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    # Record and print data periodically for monitoring
    if step % 100 == 0: # Print every 100 simulation steps
        num_tls_phases_in_state = len(TLS_IDS)
        current_total_queue = sum(new_state[:-num_tls_phases_in_state])
        
        print(f"Step {step}: Total Queue = {current_total_queue}, Reward = {reward:.2f}, Cumulative Reward = {cumulative_reward:.2f}")
        
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(current_total_queue)
        
# -------------------------
# Step 8: Close connection between SUMO and Traci
# -------------------------
traci.close()

# -------------------------
# Step 9: Visualization of Results
# -------------------------

print("\n=== Generating Plots ===")
# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Fixed Timing: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Fixed Timing: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Step 10: Process Tripinfo Output
# -------------------------

def process_tripinfo_output(filename: str, sumo_config_file: str): # ADDED sumo_config_file parameter
    """
    Parses the tripinfo XML file to calculate average travel time, average speed,
    and total number of completed trips.
    """
    total_travel_time = 0.0
    total_travel_speed = 0.0
    completed_trips = 0

    if not os.path.exists(filename):
        print(f"\nError: Tripinfo output file '{filename}' not found. Please ensure it's generated by SUMO.")
        print(f"Make sure '{filename}' is specified in the <output> section of your '{sumo_config_file}'.") # Use parameter
        return

    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        for tripinfo in root.findall('tripinfo'):
            # Only consider trips that have a duration and route length, indicating completion
            duration_str = tripinfo.get('duration')
            route_length_str = tripinfo.get('routeLength')

            if duration_str is not None and route_length_str is not None:
                duration = float(duration_str)
                route_length = float(route_length_str)

                if duration > 0: # Ensure valid duration to avoid division by zero
                    total_travel_time += duration
                    speed = route_length / duration
                    total_travel_speed += speed
                    completed_trips += 1

        print("\n=== Tripinfo Analysis Results ===")
        print(f"Total Number of Trips Completed: {completed_trips}")

        if completed_trips > 0:
            avg_travel_time = total_travel_time / completed_trips
            avg_travel_speed = total_travel_speed / completed_trips
            print(f"Average Travel Time: {avg_travel_time:.2f} seconds")
            print(f"Average Speed: {avg_travel_speed:.2f} m/s")
        else:
            print("No completed trips found in the tripinfo output to calculate average metrics.")

    except ET.ParseError as e:
        print(f"\nError parsing tripinfo XML file '{filename}': {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during tripinfo processing: {e}")

# ADDED: Step 11: Process Emission Output
def process_emission_output(filename: str, sumo_config_file: str):
    total_emissions = {
        'CO2': 0.0, 'CO': 0.0, 'HC': 0.0, 'NOx': 0.0,
        'PMx': 0.0, 'fuel': 0.0, 'noise': 0.0
    }
    if not os.path.exists(filename):
        print(f"\nError: Emission output file '{filename}' not found. Please ensure it's generated by SUMO.")
        print(f"Make sure '{filename}' is specified in the <output> section of your '{sumo_config_file}'.")
        return

    try:
        root = ET.parse(filename).getroot()
        for timestep_element in root.findall('timestep'):
            for vehicle_element in timestep_element.findall('vehicle'):
                for pollutant in total_emissions.keys():
                    value_str = vehicle_element.get(pollutant)
                    if value_str is not None:
                        total_emissions[pollutant] += float(value_str)

        print("\n=== Emission Analysis Results ===")
        for pollutant, total_value in total_emissions.items():
            unit = "mg"
            if pollutant == 'fuel': unit = "mL"
            elif pollutant == 'noise': unit = "dB"
            print(f"Total {pollutant}: {total_value:.2f} {unit}")
    except ET.ParseError as e:
        print(f"\nError parsing emission XML file '{filename}': {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during emission processing: {e}")


# Call the function to process tripinfo output after the simulation ends
process_tripinfo_output(TRIPINFO_OUTPUT_FILE, SUMO_CONFIG_FILE) # Pass config file name

# Call the function to process emission output after the simulation ends
process_emission_output(EMISSION_OUTPUT_FILE, SUMO_CONFIG_FILE) # Call new function

print("\nSimulation complete.")
