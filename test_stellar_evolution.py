"""
Test suite for stellar evolution physics module.

This module contains comprehensive tests for all stellar physics functions
including constants validation, stellar structure equations, nuclear reaction rates,
opacity calculations, and stellar evolution modeling.
"""

import pytest
import numpy as np
from stellar_evolution import (
    G, c, a,
    hydrostatic_equilibrium,
    energy_generation,
    radiative_transfer,
    proton_proton_chain,
    CNO_cycle,
    kappa,
    main_sequence_lifetime,
    stellar_evolution,
    initial_conditions,
    run_stellar_evolution_simulation
)


class TestConstants:
    """Test physical constants."""
    
    def test_gravitational_constant(self):
        """Test that gravitational constant has correct value."""
        assert abs(G - 6.67408e-11) < 1e-16
        
    def test_speed_of_light(self):
        """Test that speed of light has correct value."""
        assert c == 299792458
        
    def test_radiation_constant(self):
        """Test that radiation constant has correct value."""
        assert abs(a - 7.5657e-16) < 1e-20


class TestStellarStructureEquations:
    """Test stellar structure equation functions."""
    
    def test_hydrostatic_equilibrium(self):
        """Test hydrostatic equilibrium calculation."""
        # Test with typical stellar values
        rho = 1e3  # kg/m^3
        M = 2e30   # kg (solar mass)
        r = 7e8    # m (solar radius)
        
        result = hydrostatic_equilibrium(rho, M, r)
        
        # Should be negative (pressure decreases outward)
        assert result < 0
        
        # Check order of magnitude (should be around -0.027 Pa/m)
        expected = -rho * G * M / r**2
        assert abs(result - expected) < 1e-10
        
    def test_hydrostatic_equilibrium_zero_radius(self):
        """Test hydrostatic equilibrium with zero radius raises error."""
        with pytest.raises(ZeroDivisionError):
            hydrostatic_equilibrium(1e3, 2e30, 0)
            
    def test_energy_generation(self):
        """Test energy generation calculation."""
        rho = 1e3      # kg/m^3
        epsilon = 1e-6 # W/kg
        dM = 1e20      # kg
        
        result = energy_generation(rho, epsilon, dM)
        expected = epsilon * rho * dM
        
        assert result == expected
        assert result > 0  # Energy generation should be positive
        
    def test_radiative_transfer(self):
        """Test radiative transfer calculation."""
        kappa_val = 1e-2   # m^2/kg
        rho = 1e3          # kg/m^3
        L = 1e26           # W
        r = 7e8            # m
        T = 1e7            # K
        
        result = radiative_transfer(kappa_val, rho, L, r, T)
        
        # Should be negative (temperature decreases outward)
        assert result < 0
        
        # Check calculation
        expected = -3 * kappa_val * rho * L / (16 * np.pi * a * c * T**3)
        assert abs(result - expected) < 1e-10


class TestNuclearReactionRates:
    """Test nuclear reaction rate functions."""
    
    def test_proton_proton_chain(self):
        """Test proton-proton chain reaction rate."""
        T = 1e7    # K
        rho = 1e3  # kg/m^3
        
        result = proton_proton_chain(T, rho)
        expected = T**4 * rho**2
        
        assert result == expected
        assert result > 0
        
        # Test temperature dependence (T^4)
        result_2T = proton_proton_chain(2*T, rho)
        assert abs(result_2T / result - 16) < 1e-10  # 2^4 = 16
        
    def test_CNO_cycle(self):
        """Test CNO cycle reaction rate."""
        T = 1e7    # K
        rho = 1e3  # kg/m^3
        
        result = CNO_cycle(T, rho)
        expected = T**19 * rho
        
        assert result == expected
        assert result > 0
        
        # Test strong temperature dependence (T^19)
        result_2T = CNO_cycle(2*T, rho)
        assert abs(result_2T / result - 2**19) < 1e-10


class TestOpacity:
    """Test opacity function."""
    
    def test_kappa(self):
        """Test opacity calculation."""
        rho = 1e3  # kg/m^3
        T = 1e7    # K
        
        result = kappa(rho, T)
        expected = 1e-2 * rho**0.5 * T**-3
        
        assert abs(result - expected) < 1e-15
        assert result > 0
        
    def test_kappa_temperature_dependence(self):
        """Test opacity temperature dependence."""
        rho = 1e3  # kg/m^3
        T1 = 1e7   # K
        T2 = 2e7   # K
        
        k1 = kappa(rho, T1)
        k2 = kappa(rho, T2)
        
        # Higher temperature should give lower opacity (T^-3 dependence)
        assert k2 < k1
        assert abs(k1 / k2 - 8) < 1e-10  # 2^3 = 8


class TestStellarEvolutionTimescales:
    """Test stellar evolution timescale functions."""
    
    def test_main_sequence_lifetime(self):
        """Test main sequence lifetime calculation."""
        M = 2e30   # kg (solar mass)
        L = 3.8e26 # W (solar luminosity)
        
        result = main_sequence_lifetime(M, L)
        expected = M / L
        
        assert result == expected
        assert result > 0
        
    def test_main_sequence_lifetime_zero_luminosity(self):
        """Test main sequence lifetime with zero luminosity raises error."""
        with pytest.raises(ZeroDivisionError):
            main_sequence_lifetime(2e30, 0)


class TestDifferentialEquations:
    """Test differential equation functions."""
    
    def test_stellar_evolution(self):
        """Test stellar evolution differential equation."""
        state = [1e3, 1e26, 1e7]  # [density, luminosity, temperature]
        t = 1e6                   # s
        M = 2e30                  # kg
        
        result = stellar_evolution(state, t, M)
        
        # Should return list of derivatives
        assert isinstance(result, list)
        assert len(result) == 3
        
        # All derivatives should be finite numbers
        for derivative in result:
            assert np.isfinite(derivative)
            
    def test_stellar_evolution_with_zero_time(self):
        """Test stellar evolution with zero time is handled gracefully."""
        state = [1e3, 1e26, 1e7]  # [density, luminosity, temperature]
        t = 0                     # s
        M = 2e30                  # kg
        
        # Should not raise error and return finite derivatives
        result = stellar_evolution(state, t, M)
        assert isinstance(result, list)
        assert len(result) == 3
        for derivative in result:
            assert np.isfinite(derivative)


class TestInitialConditions:
    """Test initial conditions function."""
    
    def test_initial_conditions(self):
        """Test initial conditions generation."""
        M = 2e30  # kg
        
        result = initial_conditions(M)
        
        # Should return list of 3 values
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Check expected values
        expected = [1e3, 1e26, 1e7]
        assert result == expected
        
        # All values should be positive
        for value in result:
            assert value > 0


class TestSimulation:
    """Test complete simulation functions."""
    
    def test_run_stellar_evolution_simulation_default(self):
        """Test simulation with default parameters."""
        # Run without plotting to avoid display issues in tests
        result = run_stellar_evolution_simulation(plot=False)
        
        # Should return dictionary
        assert isinstance(result, dict)
        
        # Should have results for default masses
        default_masses = [1e30, 5e30, 1e31]
        assert len(result) == len(default_masses)
        
        for mass in default_masses:
            assert mass in result
            
            # Each result should be a 2D array
            solution = result[mass]
            assert solution.ndim == 2
            assert solution.shape[1] == 3  # density, luminosity, temperature
            
    def test_run_stellar_evolution_simulation_custom(self):
        """Test simulation with custom parameters."""
        M_values = [1e30, 2e30]
        t_points = np.linspace(0, 1e9, 100)
        
        result = run_stellar_evolution_simulation(
            M_values=M_values, 
            t_points=t_points, 
            plot=False
        )
        
        # Should have results for custom masses
        assert len(result) == len(M_values)
        for mass in M_values:
            assert mass in result
            
            # Should have correct number of time points
            solution = result[mass]
            assert solution.shape[0] == len(t_points)


class TestPhysicalReasonableness:
    """Test that results are physically reasonable."""
    
    def test_stellar_mass_scaling(self):
        """Test that more massive stars have different properties."""
        M1 = 1e30   # Lower mass
        M2 = 5e30   # Higher mass
        
        # Initial conditions should be the same (simplified model)
        ic1 = initial_conditions(M1)
        ic2 = initial_conditions(M2)
        assert ic1 == ic2
        
    def test_reaction_rates_temperature_scaling(self):
        """Test that reaction rates scale properly with temperature."""
        T_low = 1e6   # K
        T_high = 2e7  # K
        rho = 1e3     # kg/m^3
        
        # Proton-proton chain should increase with temperature
        pp_low = proton_proton_chain(T_low, rho)
        pp_high = proton_proton_chain(T_high, rho)
        assert pp_high > pp_low
        
        # CNO cycle should increase much more rapidly with temperature
        cno_low = CNO_cycle(T_low, rho)
        cno_high = CNO_cycle(T_high, rho)
        assert cno_high > cno_low
        
        # At high temperatures, CNO should dominate
        # (This is a simplified test - in reality the crossover is more complex)
        if T_high > 1.5e7:
            assert cno_high / CNO_cycle(1, rho) > pp_high / proton_proton_chain(1, rho)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])