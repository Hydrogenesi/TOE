#!/usr/bin/env python3
"""
Demonstration script for the TOE stellar evolution module.

This script shows how to use the stellar evolution functions and runs 
a basic simulation to demonstrate the functionality.
"""

import stellar_evolution as se
import numpy as np

def main():
    print("=== TOE Stellar Evolution Demonstration ===\n")
    
    # Display physical constants
    print("Physical Constants:")
    print(f"  Gravitational constant G = {se.G} m³ kg⁻¹ s⁻²")
    print(f"  Speed of light c = {se.c} m/s")
    print(f"  Radiation constant a = {se.a} J m⁻³ K⁻⁴")
    print()
    
    # Test stellar structure equations
    print("Stellar Structure Equations:")
    rho = 1e3  # kg/m³
    M = 2e30   # kg (solar mass)
    r = 7e8    # m (solar radius)
    L = 3.8e26 # W (solar luminosity)
    T = 1e7    # K
    
    he = se.hydrostatic_equilibrium(rho, M, r)
    print(f"  Hydrostatic equilibrium: {he:.2e} Pa/m")
    
    eg = se.energy_generation(rho, 1e-6, 1e20)
    print(f"  Energy generation: {eg:.2e} W")
    
    kappa_val = se.kappa(rho, T)
    rt = se.radiative_transfer(kappa_val, rho, L, r, T)
    print(f"  Radiative transfer: {rt:.2e} W/m²")
    print()
    
    # Test nuclear reaction rates
    print("Nuclear Reaction Rates:")
    pp = se.proton_proton_chain(T, rho)
    cno = se.CNO_cycle(T, rho)
    print(f"  Proton-proton chain: {pp:.2e} s⁻¹")
    print(f"  CNO cycle: {cno:.2e} s⁻¹")
    print()
    
    # Test opacity
    print(f"Opacity: {kappa_val:.2e} m²/kg")
    print()
    
    # Test stellar evolution timescales
    ms_lifetime = se.main_sequence_lifetime(M, L)
    print(f"Main sequence lifetime: {ms_lifetime:.2e} s ({ms_lifetime/(365.25*24*3600):.2e} years)")
    print()
    
    # Test initial conditions
    ic = se.initial_conditions(M)
    print(f"Initial conditions: density={ic[0]:.1e} kg/m³, luminosity={ic[1]:.1e} W, temperature={ic[2]:.1e} K")
    print()
    
    # Run a small simulation (without plotting to avoid display issues)
    print("Running stellar evolution simulation...")
    M_values = [1e30, 2e30]  # Smaller set for demo
    t_points = np.linspace(1e6, 1e9, 100)  # Start from 1e6 to avoid t=0 issues
    
    results = se.run_stellar_evolution_simulation(
        M_values=M_values, 
        t_points=t_points, 
        plot=False
    )
    
    print(f"Simulation completed for {len(M_values)} stellar masses")
    for mass in M_values:
        solution = results[mass]
        print(f"  Mass {mass:.1e} kg: {solution.shape[0]} time steps, {solution.shape[1]} variables")
        print(f"    Final state: density={solution[-1,0]:.2e}, luminosity={solution[-1,1]:.2e}, temperature={solution[-1,2]:.2e}")
    
    print("\n=== Demonstration Complete ===")

if __name__ == "__main__":
    main()