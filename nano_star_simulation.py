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

# --- Quantumonix Parameters ---
class Proton:
    def __init__(self, id, lineage):
        self.id = id
        self.lineage = lineage
        self.entangled_with = None

proton_lineage = []
bifurcation_depth = 3  # The maximum depth of the recursive split

def hydrogen_split(proton, current_depth, max_depth):
    """
    Recursively splits a proton, creating a lineage of new protons.
    """
    if current_depth >= max_depth:
        return

    # Create two new entangled protons
    proton1 = Proton(f"{proton.id}-1", proton.lineage + [proton.id])
    proton2 = Proton(f"{proton.id}-2", proton.lineage + [proton.id])

    # Entangle the protons
    proton1.entangled_with = proton2
    proton2.entangled_with = proton1

    proton_lineage.append(proton1)
    proton_lineage.append(proton2)

    # Recursively split the new protons
    hydrogen_split(proton1, current_depth + 1, max_depth)
    hydrogen_split(proton2, current_depth + 1, max_depth)


def print_mythic_insert(state, time_step):
    """
    Prints narrative inserts based on the current state of the simulation.
    """
    if state == 0 and time_step == 0:
        print("--- CUE CARD: The Genesis Atom ---")
        print("In the beginning, there was a single hydrogen atom, a protostar's seed.")
    elif state == 1 and state_history[time_step - 1] == 0:
        print("\n--- CUE CARD: The Metallic Transformation ---")
        print("Under immense pressure, the hydrogen cluster transforms into a metallic state.")
    elif state == 2 and state_history[time_step - 1] == 1:
        print("\n--- CUE CARD: The Brown Dwarf Awakens ---")
        print("The nano-star evolves into a brown dwarf, a crucible of quantum phenomena.")
        print("\n--- MYTHIC INSERT: The Phoenix Quantumonix ---")
        print("The hydrogen atom splits, birthing entangled protons in a recursive dance.")

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
    print_mythic_insert(star_state, t)
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

        # Initiate the hydrogen split when the brown dwarf phase begins
        if len(proton_lineage) == 0:
            initial_proton = Proton("P0", [])
            proton_lineage.append(initial_proton)
            hydrogen_split(initial_proton, 0, bifurcation_depth)


# --- Visualization ---
def plot_fractal_tree(ax, proton, x, y, angle, length, depth):
    if depth == 0:
        return

    x_end = x + length * np.cos(np.radians(angle))
    y_end = y + length * np.sin(np.radians(angle))

    ax.plot([x, x_end], [y, y_end], 'w-')

    children = [p for p in proton_lineage if p.lineage[-1] == proton.id]
    if len(children) == 2:
        plot_fractal_tree(ax, children[0], x_end, y_end, angle - 30, length * 0.7, depth - 1)
        plot_fractal_tree(ax, children[1], x_end, y_end, angle + 30, length * 0.7, depth - 1)

fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# Plot Atom Count
axs[0].plot(atom_count_history, label='Number of Hydrogen Atoms')
axs[0].axhline(y=metallic_threshold, color='r', linestyle='--', label='Metallic Threshold')
axs[0].text(time_steps / 2, metallic_threshold + 50, 'Transformation Threshold', color='r')
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

# Plot Fractal Tree
if proton_lineage:
    axs[3].set_facecolor('black')
    axs[3].set_title('Bifurcation Lineage')
    initial_proton = proton_lineage[0]
    plot_fractal_tree(axs[3], initial_proton, 0, 0, 90, 1, bifurcation_depth)
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].axis('off')

plt.tight_layout()
plt.savefig('nano_star_evolution.png')
