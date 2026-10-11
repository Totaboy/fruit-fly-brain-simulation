import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. BRAIN ARCHITECTURE SETUP (Compressed Graph Structure)
# =====================================================================
# Let's simulate a network of 150 neurons mimicking a fly's visual lobe
num_neurons = 150  
time_steps = 200   # 200 ms simulation window
dt = 1.0           # 1 millisecond intervals

# Biological properties of Leaky Integrate-and-Fire (LIF) neurons
v_threshold = 1.0   # Membrane potential needed to fire an electrical spike
v_reset = 0.0       # Reset voltage after a spike
tau = 10.0          # Membrane time constant (controls speed of decay)
decay = np.exp(-dt / tau) # Voltage leak factor per time step

# Generate a sparse synaptic matrix (Fruit fly brains are highly sparse)
# Instead of storing 22,500 connections (150x150), we only map active paths
np.random.seed(42) 
sparsity = 0.05  # Only 5% of all possible connections exist
raw_matrix = np.random.randn(num_neurons, num_neurons) * 0.5
raw_matrix[raw_matrix < 0] = 0 # Keep only excitatory links
mask = np.random.rand(num_neurons, num_neurons) < sparsity
synaptic_weights = raw_matrix * mask

# =====================================================================
# 2. SIMULATION MEMORY ALLOCATION
# =====================================================================
# Voltage tracker for all neurons across the current frame
voltages = np.zeros(num_neurons)
# Binary matrix tracking exactly when each neuron spikes (1=Spike, 0=Rest)
spike_history = np.zeros((num_neurons, time_steps))

# =====================================================================
# 3. RUNNING THE NEUROMORPHIC SIMULATION LOOP
# =====================================================================
print("Simulating neural network activity...")

for t in range(1, time_steps):
    # --- Input Stimulus ---
    # Simulate a visual threat (like a bird shadow) hitting neurons 0 to 10
    # The stimulus lasts from time step 20 to time step 60
    stimulus = np.zeros(num_neurons)
    if 20 <= t <= 60:
        stimulus[0:11] = 1.8  # Injecting sensory current into input layer
        
    # --- Synaptic Transmission ---
    # Check which neurons fired in the previous millisecond
    previous_spikes = spike_history[:, t-1]
    
    # Mathematical dot product: calculates the total incoming charge to
    # every neuron based on what spiked and how heavily they are connected.
    incoming_current = np.dot(previous_spikes, synaptic_weights)
    
    # --- Membrane Potential Update ---
    # Apply the leak equation: New Voltage = (Old Voltage * Decay) + Inputs
    voltages = (voltages * decay) + incoming_current + stimulus
    
    # --- Thresholding & Firing Action ---
    # Identify every neuron that broke the threshold limit
    spiked_now = voltages >= v_threshold
    
    # Record the active spikes into history
    spike_history[spiked_now, t] = 1.0
    
    # Reset the fired neurons back to their base resting voltage
    voltages[spiked_now] = v_reset

print("Simulation complete! Processing visualization maps...")

# =====================================================================
# 4. VISUALIZATION & DATA ANCHORS
# =====================================================================
# Creating a Raster Plot (Standard neuroscience way to look at brain data)
plt.figure(figsize=(10, 6))
# Find coordinates of every spike to plot them as distinct points
neurons_idx, times_idx = np.where(spike_history == 1.0)
plt.scatter(times_idx, neurons_idx, color='black', s=2, marker='|')

plt.title("Neuromorphic Simulation: 150-Neuron Connectome Sub-Circuit", fontsize=12)
plt.xlabel("Time Steps (ms)", fontsize=10)
plt.ylabel("Neuron Index ID", fontsize=10)
plt.xlim(0, time_steps)
plt.ylim(-1, num_neurons)
plt.grid(True, alpha=0.2)
plt.show()
