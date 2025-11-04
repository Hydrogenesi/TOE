import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
# Let's start with a single hydrogen atom
initial_hydrogen_atoms = 1
# We'll simulate the process over 1000 time steps
time_steps = 1000
# We'll define a variable to represent the state of our nano-star
# 0: Gas, 1: Metallic, 2: Brown Dwarf
star_state = 0

# --- Physical Constants (simplified for this narrative simulation) ---
# A simplified "attraction force" for hydrogen atoms.
# This will be used to model the growth of the cluster.
hydrogen_attraction = 1.5
# The pressure and voltage threshold for the metallic state transformation.
metallic_threshold = 100
# The time steps required in the metallic state before becoming a brown dwarf
brown_dwarf_threshold = 200

# We'll store the number of atoms at each time step to visualize the growth
atom_count_history = []
# We'll also store the state of the star at each time step
state_history = []
# We'll store the magnetic field and conductivity for the brown dwarf phase
magnetic_field_history = []
conductivity_history = []


def simulate_protostar_phase(current_atoms):
    """
    Simulates the growth of the nano-star by attracting more hydrogen atoms.
    The growth is modeled as an exponential increase with some randomness.
    """
    # The number of new atoms is proportional to the current number of atoms
    # and the attraction force. We add a random factor to make it more dynamic.
    new_atoms = int(current_atoms * (hydrogen_attraction - 1) * np.random.uniform(0.5, 1.5))
    return current_atoms + new_atoms


def simulate_brown_dwarf_phase(current_atoms, time_step):
    """
    Simulates the brown dwarf phase, with fluctuating magnetic and conductive properties.
    """
    # The magnetic field and conductivity are modeled as sine waves with some noise
    # to represent the "unusual" properties.
    magnetic_field = np.sin(time_step / 50) * (current_atoms / metallic_threshold) + np.random.normal(0, 0.1)
    conductivity = np.cos(time_step / 50) * (current_atoms / metallic_threshold) + np.random.normal(0, 0.1)
    return magnetic_field, conductivity


# --- Simulation Loop ---
current_atoms = initial_hydrogen_atoms
for t in range(time_steps):
    atom_count_history.append(current_atoms)
    state_history.append(star_state)

    if star_state == 0:  # Gaseous State (Protostar Phase)
        current_atoms = simulate_protostar_phase(current_atoms)
        # Check for transition to the metallic state
        if current_atoms >= metallic_threshold:
            star_state = 1
        magnetic_field_history.append(0)
        conductivity_history.append(0)

    elif star_state == 1:  # Metallic State
        # In the metallic state, the cluster's properties change.
        # For this simulation, we'll assume it stops growing and prepares for the next stage.
        if t > state_history.index(1) + brown_dwarf_threshold:
            star_state = 2
        magnetic_field_history.append(0)
        conductivity_history.append(0)

    elif star_state == 2:  # Brown Dwarf State
        magnetic_field, conductivity = simulate_brown_dwarf_phase(current_atoms, t)
        magnetic_field_history.append(magnetic_field)
        conductivity_history.append(conductivity)


# --- Visualization ---
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot Atom Count
axs[0].plot(atom_count_history, label='Number of Hydrogen Atoms')
axs[0].axhline(y=metallic_threshold, color='r', linestyle='--', label='Metallic Threshold')
axs[0].set_xlabel('Time Steps')
axs[0].set_ylabel('Number of Atoms')
axs[0].set_title('Nano-Star Growth')
axs[0].legend()
axs[0].grid(True)

# Plot Star State
axs[1].plot(state_history, label='Star State', drawstyle='steps-post')
axs[1].set_yticks([0, 1, 2])
axs[1].set_yticklabels(['Gas', 'Metallic', 'Brown Dwarf'])
axs[1].set_xlabel('Time Steps')
axs[1].set_ylabel('State')
axs[1].set_title('Nano-Star State Evolution')
axs[1].legend()
axs[1].grid(True)

# Plot Brown Dwarf Properties
axs[2].plot(magnetic_field_history, label='Magnetic Field')
axs[2].plot(conductivity_history, label='Conductivity')
axs[2].set_xlabel('Time Steps')
axs[2].set_ylabel('Magnitude')
axs[2].set_title('Brown Dwarf Properties')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('nano_star_evolution.png')
