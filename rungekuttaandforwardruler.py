import numpy as np

class RungeKutta4:
    """
    Runge-Kutta 4th-order stepper (RK4) for first-order ordinary differential equations (ODEs).
    systems dy/dr = f(t, y)
    
    Provides:
        - step(f, r0, y0, h): advance one RK4 step from (r0, y0) to (r0 + h, y1)
        - orbit_rhs(t, y): RHS for orbital motion with state vector [x, y, vx, vy]
    """
    @staticmethod
    def step(f, r0, y0, h):
        """
        Performs RK4 for the average of four vector directions in first-order 
        forward Euler steps, respectively. This function counts the number of iterations
        using step Euler size or step height h 

        Args:
            f (callable): function that computes dy/dt at (t0, y0)
            r0 (float): current value of the independent variable, here time
            y0 (np.ndarray): current state vector
            h (float): timestep size
        
        Returns:
            np.ndarray: updated state after one RK4 Euler step (approx. to y at r0 + h)
        """
        k1 = f(r0, y0) # at base point
        k2 = f(r0 + 0.5*h, y0 + 0.5*h*k1) # evaluate state vector after forward euler step in k1
        k3 = f(r0 + 0.5*h, y0 + 0.5*h*k2) # evaluate state vector after forward euler step in k2
        k4 = f(r0 + h, y0+h*k3) # evaluate state vector after ffull euler step in k3
        
        # avg. of the four vector directions for 4th-order accuracy
        result = y0 + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4) # update next value of y
        return result

    @staticmethod
    def orbit_rhs(t, y):
        """
        Right-hand side (RHS) of the two-body orbital motion equations
        for the TRAPPIST-1e system in the x-y plane.

        The state vector is defined as:
            y = [x, y_pos, vx, vy]

        where:
            x, y_pos: orbital position coordinates (m)
            vx, vy: velocity components (m/s)

        The function computes the derivatives:
            dx/dt  = vx
            dy/dt  = vy
            dvx/dt = ax
            dvy/dt = ay

        using Newtonian gravity for a test particle orbiting
        a central M-dwarf star.

        Notes:
            - The orbital radius is computed as:
                r = sqrt(x^2 + y^2)

            - Gravitational acceleration follows the inverse-square law:
                a = -GM/r^2

            - The acceleration components are:
                ax = -GMx/r^3
                ay = -GMy/r^3

        Args:
            t (float): Current simulation time (s).
            y (np.ndarray): Current state vector [x, y_pos, vx, vy].

        Returns:
            np.ndarray: Array containing the time derivatives: [vx, vy, ax, ay].
        """
        x, y_pos, vx, vy = y

        G = 6.67430e-11
        M_star = 0.0898 * 1.98847e30 # mass of star in kg (0.0898 solar masses)

        r = np.sqrt(x**2 + y_pos**2)

        ax = -G * M_star * x / r**3
        ay = -G * M_star * y_pos / r**3

        return np.array([vx, vy, ax, ay], dtype=float)
    

def euler_step(f, t, y, h):
    """
    First-order Forward Euler integrator for systems of
    first-order ordinary differential equations (ODEs).

    The Forward Euler method advances the solution using
    the local derivative evaluated at the current timestep:

        y_(n+1) = y_n + h*f(t_n, y_n)

    where:
        y_n: current state vector
        h: timestep size
        f(t, y): derivative function (RHS of the ODE system)

    Notes:
        - This is an explicit first-order numerical method.
        - The method is simple and computationally inexpensive,
          but accumulates truncation errors over long integrations.
        - For orbital motion problems, these errors can produce
          nonphysical changes in quantities such as energy and
          angular momentum.

    Args:
        f (callable): unction that computes dy/dt for the system.
        t (float): Current simulation time.
        y (np.ndarray): Current state vector.
        h (float): Timestep size.

    Returns:
        np.ndarray: Updated state vector after one Euler step.
    """
    return y + h * f(t, y)

