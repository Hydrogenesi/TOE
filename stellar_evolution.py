"""
Stellar Evolution Physics Module

This module contains functions for stellar structure equations, nuclear reaction rates,
opacity calculations, and stellar evolution modeling.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
G = 6.67408e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458    # Speed of light (m/s)
a = 7.5657e-16   # Radiation constant (J/m^3K^4)

# Stellar Structure Equations
def hydrostatic_equilibrium(rho: float, M: float, r: float) -> float:
    """
    Compute hydrostatic equilibrium.
    
    Args:
        rho (float): Density (kg/m^3)
        M (float): Mass (kg)
        r (float): Radius (m)
    
    Returns:
        float: Hydrostatic equilibrium value (Pa/m)
    """
    return -rho * G * M / r**2

def energy_generation(rho: float, epsilon: float, dM: float) -> float:
    """
    Compute energy generation.
    
    Args:
        rho (float): Density (kg/m^3)
        epsilon (float): Energy generation rate (W/kg)
        dM (float): Mass change (kg)
    
    Returns:
        float: Energy generation value (W)
    """
    return epsilon * rho * dM

def radiative_transfer(kappa: float, rho: float, L: float, r: float, T: float) -> float:
    """
    Compute radiative transfer.
    
    Args:
        kappa (float): Opacity (m^2/kg)
        rho (float): Density (kg/m^3)
        L (float): Luminosity (W)
        r (float): Radius (m)
        T (float): Temperature (K)
    
    Returns:
        float: Radiative transfer value (W/m^2)
    """
    return -3 * kappa * rho * L / (16 * np.pi * a * c * T**3)

# Nuclear Reaction Rates
def proton_proton_chain(T: float, rho: float) -> float:
    """
    Compute proton-proton chain reaction rate.
    
    Args:
        T (float): Temperature (K)
        rho (float): Density (kg/m^3)
    
    Returns:
        float: Proton-proton chain reaction rate (1/s)
    """
    return T**4 * rho**2

def CNO_cycle(T: float, rho: float) -> float:
    """
    Compute CNO cycle reaction rate.
    
    Args:
        T (float): Temperature (K)
        rho (float): Density (kg/m^3)
    
    Returns:
        float: CNO cycle reaction rate (1/s)
    """
    return T**19 * rho

# Opacity
def kappa(rho: float, T: float) -> float:
    """
    Compute opacity.
    
    Args:
        rho (float): Density (kg/m^3)
        T (float): Temperature (K)
    
    Returns:
        float: Opacity value (m^2/kg)
    """
    return 1e-2 * rho**0.5 * T**-3

# Stellar Evolution Timescales
def main_sequence_lifetime(M: float, L: float) -> float:
    """
    Compute main sequence lifetime.
    
    Args:
        M (float): Mass (kg)
        L (float): Luminosity (W)
    
    Returns:
        float: Main sequence lifetime (s)
    """
    return M / L

# Differential Equations
def stellar_evolution(state: list, t: float, M: float) -> list:
    """
    Compute stellar evolution differential equations.
    
    Args:
        state (list): Current state [density, luminosity, temperature]
        t (float): Time (s)
        M (float): Mass (kg)
    
    Returns:
        list: Derivatives [density, luminosity, temperature]
    """
    rho, L, T = state
    
    # Handle t=0 case to avoid division by zero
    if t == 0:
        t = 1e-10  # Small time to avoid singularity
    
    dMdt = hydrostatic_equilibrium(rho, M, t)
    dLdt = energy_generation(rho, 1e-6, dMdt)
    dTdt = radiative_transfer(kappa(rho, T), rho, L, t, T)
    drhodt = -dMdt / (4 * np.pi * t**2)
    return [drhodt, dLdt, dTdt]

# Initial Conditions
def initial_conditions(M: float) -> list:
    """
    Compute initial conditions.
    
    Args:
        M (float): Mass (kg)
    
    Returns:
        list: Initial conditions [density, luminosity, temperature]
    """
    return [1e3, 1e26, 1e7]

# Simulation function
def run_stellar_evolution_simulation(M_values=None, t_points=None, plot=True):
    """
    Run stellar evolution simulation for given masses and time points.
    
    Args:
        M_values (list): List of stellar masses in kg
        t_points (array): Time points for simulation
        plot (bool): Whether to generate plots
    
    Returns:
        dict: Results dictionary with solutions for each mass
    """
    if M_values is None:
        M_values = [1e30, 5e30, 1e31]
    
    if t_points is None:
        t_points = np.linspace(0, 1e10, 1000)
    
    results = {}
    
    for M in M_values:
        state0 = initial_conditions(M)
        solution = odeint(stellar_evolution, state0, t_points, args=(M,))
        results[M] = solution
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(t_points, solution[:, 0], label='Density')
            plt.plot(t_points, solution[:, 1], label='Luminosity')
            plt.plot(t_points, solution[:, 2], label='Temperature')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Stellar Evolution (M={M})')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    return results

if __name__ == "__main__":
    # Run the simulation
    run_stellar_evolution_simulation()