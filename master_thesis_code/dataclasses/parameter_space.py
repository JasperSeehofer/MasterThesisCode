from dataclasses import dataclass

@dataclass
class ParameterSpace():
    """
    Dataclass to manage the parameter space of a simulation.
    """
    M: float  # mass of the MBH (massive black hole) in solar masses
    µ: float  # mass of the CO (compact object) in solar masses
    a: float  # dimensionless spin of the MBH
    a2_vec: tuple[float]  # 3-dimensional spin angular momentum of the CO
    p_0: float  # Kepler-orbit parameter: separation
    e_0: float  # Kepler-orbit parameter: eccentricity
    x_I0: float  # Kepler-orbit parameter: x_I0=cosI (I is the inclination)
    d_L: float  # luminosity distance
    theta_S: float  # polar skylocalization (solar system barycenter frame)
    phi_S: float  # azimuthal skylocalization (solar system barycenter frame)
    theta_K: float  # polar orientation of spin angular momentum 
    phi_K: float  # azimuthal orientation of spin angular momentum 
    Phi_theta0: float  # polar phase
    Phi_phi0: float  # azimuthal phase
    Phi_r0: float  # radial phase

    def assume_schwarzschild(self):
        self.a = 0
        self.x_I0 = 1
        self.Phi_theta0 = 0


