# Orbital Dynamics and Stellar Irradiation of TRAPPIST-1e

This project models the orbital motion of the exoplanet TRAPPIST-1e around its ultracool M-dwarf host star using numerical integration techniques in Python. The orbit was simulated as a two-body gravitational system and solved using both the first-order Forward Euler (FE) method and the fourth-order Runge–Kutta (RK4) method.

The project compares the numerical stability and accuracy of both integrators by analyzing:

- Orbital trajectories
- Energy conservation
- Angular momentum conservation
- Time-dependent stellar irradiation

Results show that the RK4 method maintains a stable closed orbit with significantly improved conservation properties, while the FE method accumulates numerical error over time, producing nonphysical orbital expansion.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Author

Madeline Maldonado Gutierrez  
Barnard College of Columbia University  
ASTR-UN3273: High-Energy Astrophysics
