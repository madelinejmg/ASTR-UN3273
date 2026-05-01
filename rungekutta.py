import numpy as np

# constants + EOS (based Seager+2007)
G = 6.67430e-11 # m^3 kg^-1 s^-2
rho0 = 8300.0 # kg m^-3
c_eos = 0.00349 # kg m^-3 Pa^-n
n_eos = 0.528 # dimensionless

class RungeKutta4:
    """
    Runge-Kutta 4th-order stepper (RK4) for first-order ordinary differential equations (ODEs).
    systems dy/dr = f(r, y)
    
    Provides:
        - step(f, r0, y0, h): advance one RK4 step from (r0, y0) to (r0 + h, y1)
        - rhoP(P): density from pressure via EOS (Seager+07, eq. 11)
        - rhs(r, y): right-hand side for [P, m]^T: returns [dP/dr, dm/dr]
        - RK4_planet(Pc, step_size, r_max): integrate outward until p goes to 0 and return (R, M)
    """
    @staticmethod
    def step(f, r0, y0, h):
        """
        Performs RK4 for the average of four vector directions in full 
        forward Euler steps, respectively. This function counts the number of iterations
        using step Euler size or step height h 

        Args:
            f (callable): function that computes dy/dr at (r, y)
            r (float): current value of the independent variable (e.g., radius)
            y (np.ndarray): current state vector (P, m), where P(r) and m(r)
            h (float): step size (dr)
        
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
    def rhoP(P):
        """
        The equation of state (EOS) of a relationship that states how density (rho) depends on the pressure (P)
        inside the solid planet based on Equation 11 of Seager+07.

        Notes:
            P = np.maximum(P, 0.0) - during numerical integration, due to floating-point errors, pressure can sometimes dip
            slightly below zero before the stopping condition triggers. Therefore, negative pressure is unphysical 
            here, so it is "clipped" to zero

        Returns:
            float: computes the density at a given pressure

        """
        P = np.maximum(P, 0.0)
        result = RungeKutta4.rho0 + RungeKutta4.c_eos*(P**RungeKutta4.n_eos)
        return result

    @staticmethod
    def rhs(r, y):
        """
        Right-hand side (RHS) of the given system of differential equations. 
        Here the unknowns are: the pressure, P(r) and the enclosed mass m(r). 
        These equations also depend on the density, rho(r).

        Function:
            r_eff = max(r, 1e-9) - at exact center (r=0), formulas like 1/r**2 would 
            blow up. 
                - This is a trick that prevents diving by zero by repalcing r with 
                a tiny value (1e-9) when necessary.
        Args:
            r (float): current value of the independent variable (e.g., radius)
            y (np.ndarray): current state vector (P, m), where P(r) and m(r)

        Returns:
            np.ndarray: derivatives, dP/dr, dm/dr evaluated at (r, y)
        """
        P, m = y
        r_eff = max(float(r), 1e-9) # avoid r=0
        rho = RungeKutta4.rhoP(P) # density from EOS
        dPdr = - G * m * rho / (r_eff * r_eff) # hydrostatic equilibrium
        dmdr = 4.0 * np.pi * (r_eff * r_eff) * rho # mass continuty of enclosed mass, m(r)
        result = np.array([dPdr, dmdr], dtype=float)
        return result
    
    @staticmethod
    def RK4_planet(Pc, step_size=1000.0, r_max = 2.0e7, return_profiles=False):
        """
        Integrating outward from the planet's center until pressur (P) drops to zero.

        Notes:
            - We start at r = step_size with a regular-center approximation for m:
                m0 ≈ (4/3) * pi * rho_c * r^3
            where rho_c = rho(Pc).
            - We linearly interpolate between the last two steps to estimate the
            surface R where P crosses zero, and M there.
        Args:
            Pc (float): central pressure (Pa) for an Earth-sized rocky planet
            step_size (float): 1000 m (1 km) is ~6,000 steps for planetary radii (~6,000 km)
                smaller = more accurate, but more steps
            r_max (float): safety cutoff radius (m). If the integration hasn't hit P=0 by this
            radius, stop anyway to avoid infinite loops

        Returns:
            (R, M) : tuple of floats
                R (float): final planet radius where surface pressure P = 0
                M (float): total enclosed mass enclosed at that radius
        """
        # start one step out from center
        r = float(step_size)
        rho_c = RungeKutta4.rhoP(Pc) # central density from EOS
        m0 = (4.0/3.0) * np.pi * rho_c * r**3 # Equation 25 from Seager+07
        y = np.array([float(Pc), m0], dtype=float) # state vector [P, m]

        # keep previous point for final interpolation
        r_prev, y_prev = 0.0, np.array([float(Pc), 0.0]) # center: P = Pc, m = 
        rs, Ps, Ms = [0.0, r], [float(Pc), y[0]], [0.0, y[1]]

        # integrate outward with fixed steps until P <=0 or r exceeds r_max
        while y[0] > 0.0 and r < r_max:
            r_prev, y_prev = r, y # save last step
            y = RungeKutta4.step(RungeKutta4.rhs, r, y, step_size) # advance one RK4 step
            r += step_size  # increment radius
            rs.append(r); Ps.append(float(y[0])); Ms.append(float(y[1]))

        # suggested: if pressure never reached zero (safety stop)
        if y[0] > 0.0:
            return np.nan, np.nan
        
        # linear interpolation between last two points to estimate where P = 0 surface
        P1, M1 = y_prev[0], y_prev[1]
        P2, M2 = y[0], y[1]

        # fractional distance from (r_prev) to (r) where P hits zero
        frac = 0.0 if P2 == P1 else P1 / (P1 - P2)

        # compute interpolated radius and mass
        R = r_prev + frac * (r - r_prev)
        M_at_R = M1 + frac * (M2 - M1)

        if return_profiles:
            return R, M_at_R, np.array(rs), np.array(Ps), np.array(Ms)
        
        return R, M_at_R
    
    @staticmethod
    def pendulum_rhs(t, y):
        """
        RHS for nonlinear pendulum:
            theta' = omega
            omega' = -(g/L) * sin(theta)
        y = [theta, omega]
        """
        theta, omega = y   # y must be a 2-component array
        g = 9.81
        L = 1.0

        dtheta = omega
        domega = -(g/L) * np.sin(theta)

        return np.array([dtheta, domega], dtype=float)
    
    @staticmethod
    def orbit_rhs(t, y):
        x, y_pos, vx, vy = y

        G = 6.67430e-11
        M_star = 0.0898 * 1.98847e30 # mass of star in kg (0.0898 solar masses)

        r = np.sqrt(x**2 + y_pos**2)

        ax = -G * M_star * x / r**3
        ay = -G * M_star * y_pos / r**3

        return np.array([vx, vy, ax, ay], dtype=float)

