import numpy as np
import cupy as cp
import tqdm
import os
from scipy.optimize import curve_fit, minimize, OptimizeWarning
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import concurrent.futures
warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", DeprecationWarning)
'''

███╗   ███╗██╗   ██╗██████╗     ██╗      █████╗ ██████╗ ████████╗██╗███╗   ███╗███████╗    ███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗ ██████╗ ██████╗ 
████╗ ████║██║   ██║██╔══██╗    ██║     ██╔══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
██╔████╔██║██║   ██║██████╔╝    ██║     ███████║██████╔╝   ██║   ██║██╔████╔██║█████╗      ███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║   ██║██████╔╝
██║╚██╔╝██║██║   ██║██╔══██╗    ██║     ██╔══██║██╔═══╝    ██║   ██║██║╚██╔╝██║██╔══╝      ╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║   ██║██╔══██╗
██║ ╚═╝ ██║╚██████╔╝██║  ██║    ███████╗██║  ██║██║        ██║   ██║██║ ╚═╝ ██║███████╗    ███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                                                                                                                                                                    
developed by Flynn Cassells, 2025
'''



# Constants
g = 9.81
rho = 1.225

# Vehicle parameters
car_data = {
    "mass": 370,                   # Mass of the car (kg)
    "engine_torque": 100,          # Motor torque (Nm)
    "max_motor_rpm": 6000,         # Maximum motor RPM
    "wheel_radius": 0.25,          # Wheel radius (m)
    "drag_coefficient": 0.9,       # Drag coefficient
    "frontal_area": 1.5,           # Frontal area (m^2)
    "rolling_resistance_coeff": 0.015,
    "brake_torque": 200,           # Brake torque (Nm)
    "drivetrain_efficiency": 0.95, # Drivetrain efficiency
    "wheelbase": 1.5,              # Wheelbase (m)
    "track_width": 1.2,            # Track width (m)
    "cg_height": 0.3,              # CG height (m)
    "cg_position": 0.75,           # CG position from front axle (m)
    "roll_centre": 0.75,           # Roll Centre Position from front axle (m)
    "toe_angle": 0,                # Toe Angle (deg)
    "down_force": 0.8,               # Coefficient of downforce
    "roll_stiffness_front": 25000, # Roll Stiffness for front (Nm/deg)
    "roll_stiffness_rear": 20000,  # Roll Stiffness for rear (Nm/deg)
    "chassis_torsional_stiff": 40000,
    "camber_gain": 0.1,
    "suspension_stiffness": 45,    # Suspension stiffness (N/m)
    "gamma_front": -0.1,           # Camber in the front (deg)
    "gamma_rear": -0.1,            # Camber in the rear (deg)
}


class TireCornering:
    def __init__(self, run):
        #for given run, load the data file, and set self parameters accordingly, then run the fitting functions.
        run = str(run)
        self.run = run
        param_file = f"tire_data/corner/B2356params{run}.csv"
        
        # Check if parameters file exists
        if os.path.exists(param_file):
            # Load parameters from CSV file
            params_df = pd.read_csv(param_file)
            self.a = cp.array(params_df["value"].values)
            print(f"Loaded parameters from {param_file}")
        else:
            # Load raw data and perform fitting
            file_path = "tire_data/corner/B2356raw" + run + ".dat"
            df = pd.read_csv(file_path, delimiter="\t", skiprows=1, header=0, low_memory=False)
            self.units = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            # Convert data to CuPy arrays
            self.Fz_data = -cp.array(pd.to_numeric(df["FZ"], errors='coerce').values)
            self.Fy_data = cp.array(pd.to_numeric(df["FY"], errors='coerce').values)
            self.alpha_data = cp.array(pd.to_numeric(df["SA"], errors='coerce').values)
            self.gamma_data = cp.array(pd.to_numeric(df["IA"], errors='coerce').values)
            
            # Fit parameters and save to CSV
            self.a = self.fit_Fy()
            self.save_parameters()
            print(f"Fitted parameters and saved to {param_file}")
            
    def save_parameters(self):
        # Create a DataFrame with parameter names and values
        param_names = [f"a{i}" for i in range(18)]
        params_df = pd.DataFrame({
            "parameter": param_names,
            "value": self.a.get()  # Convert CuPy array to NumPy for saving
        })
        
        # Ensure directory exists
        os.makedirs("tire_data/corner", exist_ok=True)
        
        # Save to CSV
        param_file = f"tire_data/corner/B2356params{self.run}.csv"
        params_df.to_csv(param_file, index=False)

    def pacejka_formula(self, X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17): #Defines the Pacejka Magic Formal for the lateral case
        Fz, alpha, gamma = X
        C = a0
        D = Fz*(a1*Fz+a2)*(1-a15*gamma**2)
        BCD = a3*cp.sin(cp.arctan(2*Fz/a4))*(1-a5*cp.abs(gamma))
        B = BCD/(C*D)
        H = a8*Fz + a9 + a10*gamma
        E = (a6*Fz + a7) * (1 - (a16 * gamma + a17) * cp.sign(cp.asarray(alpha + H)))
        V = a11*Fz+a12+(a13*Fz+a14)*gamma*Fz
        Bx1 = B*(alpha+H)
        return D*cp.sin(C*cp.arctan(Bx1-E*(Bx1-cp.arctan(Bx1)))) + V

    def fit_Fy(self): #Find params for given data
        #intial Guesses
        a = cp.zeros(18)
        a[0] = 1.4
        a[2] = 1100
        a[3] = 1100
        a[4] = 10
        a[7] = -2
        
        # For curve_fit, convert CuPy arrays to NumPy arrays
        X = (self.Fz_data.get(), self.alpha_data.get(), self.gamma_data.get())
        Y = self.Fy_data.get()
        
        # For initial guesses, convert CuPy array to NumPy
        popt, _ = curve_fit(self.pacejka_formula_numpy, X, Y, p0=a.get())
        
        # Convert the result back to CuPy array
        return cp.array(popt)
    
    # NumPy version for curve_fit (which doesn't support CuPy)
    def pacejka_formula_numpy(self, X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17):
        Fz, alpha, gamma = X
        C = a0
        D = Fz*(a1*Fz+a2)*(1-a15*gamma**2)
        BCD = a3*np.sin(np.arctan(2*Fz/a4))*(1-a5*np.abs(gamma))
        B = BCD/(C*D)
        H = a8*Fz + a9 + a10*gamma
        E = (a6*Fz + a7) * (1 - (a16 * gamma + a17) * np.sign(np.asarray(alpha + H)))
        V = a11*Fz+a12+(a13*Fz+a14)*gamma*Fz
        Bx1 = B*(alpha+H)
        return D*np.sin(C*np.arctan(Bx1-E*(Bx1-np.arctan(Bx1)))) + V
    
    def plot_fit(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Check if we have data loaded, if not, load it
        if not hasattr(self, 'Fz_data'):
            file_path = "tire_data/corner/B2356raw" + self.run + ".dat"
            df = pd.read_csv(file_path, delimiter="\t", skiprows=1, header=0, low_memory=False)
            self.units = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            # Convert data to CuPy arrays
            self.Fz_data = -cp.array(pd.to_numeric(df["FZ"], errors='coerce').values)
            self.Fy_data = cp.array(pd.to_numeric(df["FY"], errors='coerce').values)
            self.alpha_data = cp.array(pd.to_numeric(df["SA"], errors='coerce').values)
            self.gamma_data = cp.array(pd.to_numeric(df["IA"], errors='coerce').values)
        
        # Convert CuPy arrays to NumPy for plotting
        Fz = self.Fz_data.get()
        alpha = self.alpha_data.get()
        Fy = self.Fy_data.get()
        
        Fz_mesh, alpha_mesh = np.meshgrid(
            np.linspace(Fz.min(), Fz.max(), 50),
            np.linspace(alpha.min(), alpha.max(), 50)
        )
        
        # Get mean of gamma data
        gamma = float(cp.mean(self.gamma_data).get())
        Fy_mesh = -np.zeros_like(Fz_mesh)
        
        for i in range(Fz_mesh.shape[0]):
            for j in range(Fz_mesh.shape[1]):
                # Use CuPy version for calculations, but convert result to NumPy for plotting
                Fy_mesh[i,j] = -float(self.neg_fitted_function(Fz_mesh[i,j], alpha_mesh[i,j], gamma).get())

        # Create the scatter plot
        ax.scatter(Fz, alpha, Fy, color='blue', marker='o', label='Data')
        
        surf = ax.plot_surface(Fz_mesh, alpha_mesh, Fy_mesh, cmap="viridis", alpha=0.5, label="Fitted Surface")

        ax.set_xlabel('Normal Force (Fz)')
        ax.set_ylabel('Slip Angle (α)')
        ax.set_zlabel('Lateral Force (Fy)')
        ax.set_title('Pacejka Magic Formula Fit')
        plt.show()
        
    def neg_fitted_function(self, Fz, alpha, gamma):
        # Ensure inputs are CuPy arrays
        if not isinstance(Fz, cp.ndarray):
            Fz = cp.array(Fz)
        if not isinstance(alpha, cp.ndarray):
            alpha = cp.array(alpha)
        if not isinstance(gamma, cp.ndarray):
            gamma = cp.array(gamma)
            
        return -self.pacejka_formula((Fz, alpha, gamma), self.a[0], self.a[1], self.a[2], self.a[3], self.a[4], self.a[5], self.a[6], self.a[7], self.a[8], self.a[9], self.a[10], self.a[11], self.a[12], self.a[13], self.a[14], self.a[15], self.a[16], self.a[17])
    
    def calculate_maximum_steer(self, Fz, gamma):
        # Convert to NumPy for compatibility with minimize
        Fz_np = float(Fz) if isinstance(Fz, cp.ndarray) else Fz
        gamma_np = float(gamma) if isinstance(gamma, cp.ndarray) else gamma
        
        def objective(alpha):
            # Convert to CuPy for computation, then back to NumPy for minimize
            alpha_cp = cp.array(alpha)
            Fz_cp = cp.array(Fz_np)
            gamma_cp = cp.array(gamma_np)
            result = self.neg_fitted_function(Fz_cp, alpha_cp, gamma_cp)
            return float(result.get())
            
        result = minimize(objective, x0=0, bounds=[(-12.5,12.5)])
        maximum_Fy = -result.fun
        return maximum_Fy
    
    def calculate_maximum_rear(self, Fz, alpha, gamma):
        # Ensure inputs are CuPy arrays
        if not isinstance(Fz, cp.ndarray):
            Fz = cp.array(Fz)
        if not isinstance(alpha, cp.ndarray):
            alpha = cp.array(alpha)
        if not isinstance(gamma, cp.ndarray):
            gamma = cp.array(gamma)
            
        result = self.pacejka_formula((Fz, alpha, gamma), self.a[0], self.a[1], self.a[2], self.a[3], self.a[4], self.a[5], self.a[6], self.a[7], self.a[8], self.a[9], self.a[10], self.a[11], self.a[12], self.a[13], self.a[14], self.a[15], self.a[16], self.a[17])
        
        # Convert result to NumPy scalar if needed
        if isinstance(result, cp.ndarray):
            return float(result.get())
        return result


class TireDrive:
    def __init__(self, run):
        run = str(run)
        self.run = run
        param_file = f"tire_data/drive/B2356params{run}.csv"
        
        # Check if parameters file exists
        if os.path.exists(param_file):
            # Load parameters from CSV file
            params_df = pd.read_csv(param_file)
            self.b = cp.array(params_df["value"].values)
            print(f"Loaded parameters from {param_file}")
        else:
            # Load raw data and perform fitting
            file_path = "tire_data/drive/B2356raw" + run + ".dat"
            df = pd.read_csv(file_path, delimiter="\t", skiprows=1, header=0, low_memory=False)
            self.units = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            # Convert data to CuPy arrays
            self.Fx_data = cp.array(pd.to_numeric(df["FX"], errors='coerce').values)
            self.Fz_data = -cp.array(pd.to_numeric(df["FZ"], errors='coerce').values)
            self.slip_data = cp.array(pd.to_numeric(df["SR"], errors='coerce').values)
            
            # Fit parameters and save to CSV
            self.b = self.fit_Fx()
            self.save_parameters()
            print(f"Fitted parameters and saved to {param_file}")
    
    def save_parameters(self):
        # Create a DataFrame with parameter names and values
        param_names = [f"b{i}" for i in range(5)]
        params_df = pd.DataFrame({
            "parameter": param_names,
            "value": self.b.get()  # Convert CuPy array to NumPy for saving
        })
        
        # Ensure directory exists
        os.makedirs("tire_data/drive", exist_ok=True)
        
        # Save to CSV
        param_file = f"tire_data/drive/B2356params{self.run}.csv"
        params_df.to_csv(param_file, index=False)
        
    def pacejka_formula(self, X, B, C, D, E, V): #Defines the Pacejka Magic Formal for the longitudinal case
        Fz, slip = X
        return Fz*D*cp.sin(C*cp.arctan(B*slip-E*(B*slip-cp.arctan(B*slip)))) + V
    
    # NumPy version for curve_fit
    def pacejka_formula_numpy(self, X, B, C, D, E, V):
        Fz, slip = X
        return Fz*D*np.sin(C*np.arctan(B*slip-E*(B*slip-np.arctan(B*slip)))) + V
        
    def fit_Fx(self): #Find params
        #initial guesses
        b = cp.zeros(5)
        b[0] = 10
        b[1] = 1.65
        b[2] = 1
        b[3] = 0.97
        
        # Convert to NumPy for curve_fit
        X = (self.Fz_data.get(), self.slip_data.get())
        Y = self.Fx_data.get()
        popt, _ = curve_fit(self.pacejka_formula_numpy, X, Y, p0=b.get())
        
        # Convert back to CuPy
        return cp.array(popt)
    
    def neg_fitted_function(self, Fz, slip):
        # Ensure inputs are CuPy arrays
        if not isinstance(Fz, cp.ndarray):
            Fz = cp.array(Fz)
        if not isinstance(slip, cp.ndarray):
            slip = cp.array(slip)
            
        return -self.pacejka_formula((Fz, slip), self.b[0], self.b[1], self.b[2], self.b[3], self.b[4])
    
    def calculate_maximum(self, Fz):
        # Convert to NumPy for compatibility with minimize
        Fz_np = float(Fz) if isinstance(Fz, cp.ndarray) else Fz
        
        def objective(slip):
            # Convert to CuPy for computation, then back to NumPy for minimize
            slip_cp = cp.array(slip)
            Fz_cp = cp.array(Fz_np)
            result = self.neg_fitted_function(Fz_cp, slip_cp)
            return float(result.get())
            
        result = minimize(objective, x0=0, bounds=[(-1,1)])
        maximum_Fx = -result.fun
        return maximum_Fx
    
    def plot_fit(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Check if we have data loaded, if not, load it
        if not hasattr(self, 'Fz_data'):
            file_path = "tire_data/drive/B2356raw" + self.run + ".dat"
            df = pd.read_csv(file_path, delimiter="\t", skiprows=1, header=0, low_memory=False)
            self.units = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            # Convert data to CuPy arrays
            self.Fx_data = cp.array(pd.to_numeric(df["FX"], errors='coerce').values)
            self.Fz_data = -cp.array(pd.to_numeric(df["FZ"], errors='coerce').values)
            self.slip_data = cp.array(pd.to_numeric(df["SR"], errors='coerce').values)
        
        # Convert CuPy arrays to NumPy for plotting
        Fz = self.Fz_data.get()
        slip = self.slip_data.get()
        Fx = self.Fx_data.get()

        # Use NumPy for mesh creation
        Fz_mesh, slip_mesh = np.meshgrid(
            np.linspace(Fz.min(), Fz.max(), 50),
            np.linspace(slip.min(), slip.max(), 50)
        )

        Fx_mesh = -np.zeros_like(Fz_mesh)
        for i in range(Fz_mesh.shape[0]):
            for j in range(Fz_mesh.shape[1]):
                # Use CuPy version for calculations, but convert result to NumPy for plotting
                Fx_mesh[i,j] = -float(self.neg_fitted_function(Fz_mesh[i,j], slip_mesh[i,j]).get())

        ax.scatter(Fz, slip, Fx, color='blue', marker='o', label='Data')
        surf = ax.plot_surface(Fz_mesh, slip_mesh, Fx_mesh, cmap="viridis", alpha=0.5, label="Fitted Surface")
        plt.show()


class TireModel:
    def __init__(self):
        #chosen tire data sets --- LATER SET THESE UP WITH SOME KIND OF LOOKUP TABLE OR SMT
        self.cnr_run = 4
        self.drv_run = 69
        self.TireCornering = TireCornering(self.cnr_run)
        self.TireDrive = TireDrive(self.drv_run)
        
    def calculate(self, loads, gamma):
        #calculate the max lateral and longitudnial forces based on the given loads
        #front tires:
        max_lateral = [
            self.TireCornering.calculate_maximum_steer(loads[0], gamma[0]), 
            self.TireCornering.calculate_maximum_steer(loads[1], gamma[1]), 
            self.TireCornering.calculate_maximum_rear(loads[2], car_data["toe_angle"], gamma[2]), 
            self.TireCornering.calculate_maximum_rear(loads[3], car_data["toe_angle"], gamma[3])
        ]
        #max longitudinal:
        max_longitudinal = [
            self.TireDrive.calculate_maximum(loads[0]),
            self.TireDrive.calculate_maximum(loads[1]),
            self.TireDrive.calculate_maximum(loads[2]),
            self.TireDrive.calculate_maximum(loads[3])
        ]
        return max_lateral, max_longitudinal
class LoadTransfer:
    def __init__(self):
        self.mass = car_data["mass"]
        self.cg_height = car_data['cg_height']
        self.track_width = car_data["track_width"]
        self.wheel_base = car_data["wheelbase"]
        self.K_chassis = (car_data["chassis_torsional_stiff"]*(self.track_width/self.wheel_base)**2)/2
        self.K_front = (1/car_data["roll_stiffness_front"] + 1/self.K_chassis)**(-1)
        self.K_rear = (1/car_data["roll_stiffness_rear"]+ 1/self.K_chassis)**(-1)
        self.K_total = (1/(car_data["roll_stiffness_rear"]+car_data["roll_stiffness_front"])+ 1/self.K_chassis)**(-1)
        #calculate static weight distrobution:
        mass_static_front = self.mass*car_data["cg_position"]/self.wheel_base
        self.Fz_static = g*np.array([mass_static_front, (self.mass-mass_static_front)])
    def calculate_lateral(self, Fy):
        t = self.track_width
        h = self.cg_height
        W = Fy*h/(t/2) # total lateral weight transfer
        theta = W*h/self.K_total
        dFz_front = cp.array([W*self.K_front/self.K_total, -W*self.K_front/self.K_total])/2
        dFz_rear = cp.array([W*self.K_rear/self.K_total, -W*self.K_rear/self.K_total])/2
        def camber_change(dFz,K):
            dz = dFz/car_data["suspension_stiffness"]
            theta = dFz*h/K
            d_gamma_comp = dz*car_data["camber_gain"]
            d_gamma_roll = np.sign(dFz)*car_data["camber_gain"]*theta
            return d_gamma_comp + d_gamma_roll
        dGamma_front = camber_change(dFz_front, self.K_front)
        dGamma_rear = camber_change(dFz_rear, self.K_rear)
        return [dFz_front, dFz_rear, dGamma_front, dGamma_rear]
    def calculate_longitudinal(self, Fx):
        L = self.wheel_base
        h = self.cg_height
        dFz = cp.array([-Fx*h/L, Fx*h/L])/2 # front, rear
        def camber_change(dFz):
            dz = dFz/car_data["suspension_stiffness"]
            return dz*car_data["camber_gain"]
        dgamma_front = camber_change(dFz[0])
        dgamma_rear = camber_change(dFz[1])
        return (dFz, dgamma_front, dgamma_rear)
    def simulate(self, Fx, Fy):
        #Fx, Fy = F
        dFz_front_lat, dFz_rear_lat, dGamma_front_lat, dGamma_rear_lat = self.calculate_lateral(Fy)
        dFz_long, dGamma_front_long, dGamma_rear_long = self.calculate_longitudinal(Fx)
        dFz_front = dFz_front_lat + dFz_long[0]
        dFz_rear = dFz_rear_lat + dFz_long[1]
        dGamma_front = dGamma_front_lat+dGamma_front_long
        dGamma_rear = dGamma_rear_lat+dGamma_rear_long
        return ([dFz_front[0], dFz_front[1], dFz_rear[0], dFz_rear[1]], [dGamma_front[0], dGamma_front[1], dGamma_rear[0], dGamma_rear[1]])


class VehicleDynamics:
    def __init__(self, car_data, track):
        self.car = car_data
        self.track = track  # Store the track object
        self.tire_model = TireModel()
        self.load_transfer = LoadTransfer()
        
        # Initialize state variables
        self.velocity = 0  # m/s
        self.velocity_lateral = 0  # m/s
        self.position = cp.array([0, 0])  # x, y coordinates
        self.heading = 0  # radians
        self.yaw_rate = 0  # rad/s

    def calculate_slip_angles(self, steer_angle):
        """
        Calculate slip angles for all four wheels.
        """
        if abs(self.velocity) < 0.1:  # Prevent division by zero at very low speeds
            return [0, 0, 0, 0]
        
        # Distance from CG to front and rear axles
        a = self.car["cg_position"]  # Distance from front axle to CG
        b = self.car["wheelbase"] - a  # Distance from CG to rear axle
        
        # Calculate slip angles for front and rear
        front_slip = cp.arctan2(
            (self.velocity_lateral + self.yaw_rate * a),
            abs(self.velocity)
        ) - steer_angle
        
        rear_slip = cp.arctan2(
            (self.velocity_lateral - self.yaw_rate * b),
            abs(self.velocity)
        )
        
        return [front_slip, front_slip, rear_slip, rear_slip]

    def calculate_aerodynamic_drag(self, velocity):
        """
        Calculate aerodynamic drag force.
        """
        return 0.5 * self.car["drag_coefficient"] * self.car["frontal_area"] * rho * velocity**2

    def apply_friction_ellipse(self, Fx, Fy, Fz, mu_x, mu_y):
        """
        Apply friction ellipse model to combine longitudinal and lateral forces.
        """
        Fx_max = mu_x * Fz
        Fy_max = mu_y * Fz
        
        if (Fx / Fx_max)**2 + (Fy / Fy_max)**2 <= 1:
            return Fx, Fy
        
        scaling = cp.sqrt((Fx / Fx_max)**2 + (Fy / Fy_max)**2)
        return Fx / scaling, Fy / scaling
    def apply_friction_ellipse_parallel(self, Fx_requested, Fy_requested, loads, mu_x, mu_y):
        """Vectorized version of friction ellipse calculation"""
        # Calculate the normalized forces
        fx_norm = Fx_requested / (loads * mu_x)
        fy_norm = Fy_requested / (loads * mu_y)
        
        # Calculate the combined normalized force magnitude
        f_norm_magnitude = cp.sqrt(fx_norm**2 + fy_norm**2)
        
        # Find where the combined force exceeds the friction ellipse
        exceeds_limit = f_norm_magnitude > 1.0
        
        # Create scaling factors (1.0 for within limits, less than 1.0 for beyond)
        scaling = cp.ones_like(f_norm_magnitude)
        scaling[exceeds_limit] = 1.0 / f_norm_magnitude[exceeds_limit]
        
        # Apply scaling to get actual forces
        Fx_actual = Fx_requested * scaling
        Fy_actual = Fy_requested * scaling
        
        # Return combined forces as a 2D array
        return cp.column_stack((Fx_actual, Fy_actual))
    def calculate_forces(self, throttle, brake, steer_angle):
	    # Start with static loads
	    loads = [self.load_transfer.Fz_static[0] / 2, 
	             self.load_transfer.Fz_static[0] / 2,
	             self.load_transfer.Fz_static[1] / 2, 
	             self.load_transfer.Fz_static[1] / 2]
	    
	    # Initial camber angles
	    gamma = [self.car["gamma_front"], self.car["gamma_front"],
	             self.car["gamma_rear"], self.car["gamma_rear"]]
	    
	    # Calculate slip angles
	    slip_angles = self.calculate_slip_angles(steer_angle)
	    
	    # Add speed-dependent grip reduction
	    speed_grip_factor = max(1.0 - (self.velocity / 45.0) * 0.2, 0.8)
	    
	    # Prepare calculations that can be done once outside the loop
	    rolling_resistance = self.car["rolling_resistance_coeff"] * self.car["mass"] * g
	    drag_force = self.calculate_aerodynamic_drag(self.velocity)
	    total_resistance = rolling_resistance + drag_force
	    
	    # Engine force calculation with power curve
	    engine_force = throttle * self.car["engine_torque"] * self.car["drivetrain_efficiency"] / self.car["wheel_radius"]
	    power_factor = max(1.0 - (self.velocity / 30.0) ** 2, 0.1)
	    engine_force *= power_factor
	    
	    brake_force = brake * self.car["brake_torque"] / self.car["wheel_radius"]
	    
	    # Fixed components of force distribution
	    resistance_per_wheel = total_resistance / 4
	    front_brake = -brake_force / 2 - resistance_per_wheel
	    rear_force_base = engine_force / 2 - brake_force / 2 - resistance_per_wheel
	    
	    # Base Fx requested values (will be updated in parallel)
	    Fx_requested = [front_brake, front_brake, rear_force_base, rear_force_base]
	    
	    # Iterative solution
	    for _ in cp.range(3):
	        # Parallel computation of max tire forces
	        max_lat_long = cp.array(
	            self.tire_model.calculate_parallel(cp.array(loads), cp.array(gamma))
	        )
	        max_lateral = max_lat_long[0] * speed_grip_factor
	        max_longitudinal = max_lat_long[1] * speed_grip_factor
	        
	        # Calculate lateral forces based on slip angles (vectorized)
	        slip_params = cp.column_stack((loads, slip_angles, gamma))
	        Fy_requested = self.tire_model.TireCornering.pacejka_formula_vectorized(
	            slip_params, *self.tire_model.TireCornering.a
	        )
	        
	        # Apply friction ellipse to each wheel (parallel)
	        mu_x = max_longitudinal / cp.array(loads)
	        mu_y = max_lateral / cp.array(loads)
	        combined_forces = self.apply_friction_ellipse_parallel(
	            cp.array(Fx_requested), Fy_requested, cp.array(loads), mu_x, mu_y
	        )
	        
	        # Extract Fx and Fy components
	        Fx = combined_forces[:, 0]
	        Fy = combined_forces[:, 1]
	        
	        # Calculate totals
	        Fx_total = cp.sum(Fx)
	        Fy_total = cp.sum(Fy)
	        
	        # Update loads and camber due to weight transfer
	        delta_loads, delta_gamma = self.load_transfer.simulate(Fx_total, Fy_total)
	        
	        # Update loads and gamma (vectorized)
	        loads = cp.array([
	            self.load_transfer.Fz_static[0] / 2 + delta_loads[0],
	            self.load_transfer.Fz_static[0] / 2 + delta_loads[1],
	            self.load_transfer.Fz_static[1] / 2 + delta_loads[2],
	            self.load_transfer.Fz_static[1] / 2 + delta_loads[3]
	        ])
	        
	        gamma = cp.array([
	            self.car["gamma_front"] + delta_gamma[0],
	            self.car["gamma_front"] + delta_gamma[1],
	            self.car["gamma_rear"] + delta_gamma[2],
	            self.car["gamma_rear"] + delta_gamma[3]
	        ])
	    
	    return float(Fx_total), float(Fy_total)
	def _update_vehicle_state(self, vehicle_state, Fx, Fy, dt):
	    """Helper method to update vehicle state in parallel processing"""
	    # Extract current state
	    position = vehicle_state['position']
	    velocity = vehicle_state['velocity']
	    heading = vehicle_state['heading']
	    
	    # Calculate acceleration (simplified from vehicle.update_state)
	    mass = self.vehicle.car["mass"]
	    accel_x = Fx / mass
	    accel_y = Fy / mass
	    
	    # Update velocity (magnitude)
	    velocity += accel_x * dt
	    
	    # Update heading based on lateral forces
	    yaw_rate = Fy * self.vehicle.car["wheelbase"] / (mass * velocity) if velocity > 0.1 else 0
	    heading += yaw_rate * dt
	    
	    # Update position
	    position[0] += velocity * cp.cos(heading) * dt
	    position[1] += velocity * cp.sin(heading) * dt
	    
	    # Update the state dictionary
	    vehicle_state['position'] = position
	    vehicle_state['velocity'] = velocity
	    vehicle_state['heading'] = heading
    def update_state(self, Fx, Fy, dt):
        # Calculate accelerations
        ax = Fx / self.car["mass"]
        ay = Fy / self.car["mass"]
        
        # Update velocities in body frame
        self.velocity += ax * dt
        self.velocity_lateral += ay * dt
        
        # Enforce maximum speed based on curvature
        nearest_point = self.track.get_nearest_point(self.position[0], self.position[1])
        max_speed = self.track.calculate_max_speed(1 / (nearest_point["curvature"]+1e-25) , self)
        self.velocity = min(self.velocity, max_speed)
        
        # Update yaw rate and heading
        moment_arm = self.car["wheelbase"] / 2
        yaw_moment = Fy * moment_arm
        I_zz = self.car["mass"] * (self.car["wheelbase"]**2 + self.car["track_width"]**2) / 12
        self.yaw_rate += (yaw_moment / I_zz) * dt
        self.heading += self.yaw_rate * dt
        
        # Update position
        velocity_global_x = self.velocity * cp.cos(self.heading) - self.velocity_lateral * cp.sin(self.heading)
        velocity_global_y = self.velocity * cp.sin(self.heading) + self.velocity_lateral * cp.cos(self.heading)
        self.position[0] += velocity_global_x * dt
        self.position[1] += velocity_global_y * dt


class Track:
    def __init__(self, scale=100, track_name="figure8", load_from_cache=True):
        self.scale = scale
        self.track_width = 10  # meters
        self.resolution = 100  # points to define track
        self.track_name = track_name
        
        # CSV filenames for caching
        self.track_data_csv = f"{self.track_name}_track_data.csv"
        self.racing_line_csv = f"{self.track_name}_racing_line.csv"
        
        # Try to load track data from cache if requested
        if load_from_cache and self.load_track_from_csv():
            print(f"Loaded track data from {self.track_data_csv}")
        else:
            # Generate centerline points
            self.generate_centerline()
            
            # Calculate track properties
            self.calculate_curvature()
            self.generate_boundaries()
            
            # Save track data
            self.save_track_to_csv()
        
        # Racing line properties (to be loaded or calculated later)
        self.racing_line_x = None
        self.racing_line_y = None
        self.racing_line_speed = None
    
    def generate_centerline(self):
        """Generate centerline points for figure-8 track using parametric equations"""
        t = cp.linspace(0, 2*cp.pi, self.resolution)
        
        # Use lemniscate of Gerono equations (figure-8 curve)
        self.centerline_x = self.scale * cp.sin(t)
        self.centerline_y = self.scale * cp.sin(t) * cp.cos(t)
        
        # Calculate tangent vectors
        dx_dt = self.scale * cp.cos(t)
        dy_dt = self.scale * (cp.cos(2*t))
        
        # Normalize to get direction vectors
        magnitude = cp.sqrt(dx_dt**2 + dy_dt**2)
        self.tangent_x = dx_dt / magnitude
        self.tangent_y = dy_dt / magnitude
        
        # Calculate normal vectors (perpendicular to tangent)
        self.normal_x = -self.tangent_y
        self.normal_y = self.tangent_x
        
    def calculate_curvature(self):
        """Calculate track curvature at each point"""
        t = cp.linspace(0, 2*cp.pi, self.resolution)
        
        # Second derivatives
        d2x_dt2 = -self.scale * cp.sin(t)
        d2y_dt2 = -self.scale * (2 * cp.sin(2*t))
        
        # First derivatives
        dx_dt = self.scale * cp.cos(t)
        dy_dt = self.scale * (cp.cos(2*t))
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)
        
        self.curvature = numerator / denominator
        
    def generate_boundaries(self):
        """Generate inner and outer track boundaries"""
        half_width = self.track_width / 2
        
        # Inner boundary
        self.inner_x = self.centerline_x - half_width * self.normal_x
        self.inner_y = self.centerline_y - half_width * self.normal_y
        
        # Outer boundary
        self.outer_x = self.centerline_x + half_width * self.normal_x
        self.outer_y = self.centerline_y + half_width * self.normal_y
    
    def save_track_to_csv(self):
        """Save track data to CSV file"""
        try:
            # Convert CuPy arrays to NumPy for saving
            track_data = {
                'centerline_x': cp.asnumpy(self.centerline_x),
                'centerline_y': cp.asnumpy(self.centerline_y),
                'tangent_x': cp.asnumpy(self.tangent_x),
                'tangent_y': cp.asnumpy(self.tangent_y),
                'normal_x': cp.asnumpy(self.normal_x),
                'normal_y': cp.asnumpy(self.normal_y),
                'curvature': cp.asnumpy(self.curvature),
                'inner_x': cp.asnumpy(self.inner_x),
                'inner_y': cp.asnumpy(self.inner_y),
                'outer_x': cp.asnumpy(self.outer_x),
                'outer_y': cp.asnumpy(self.outer_y)
            }
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(track_data)
            df.to_csv(self.track_data_csv, index=False)
            print(f"Saved track data to {self.track_data_csv}")
            return True
        except Exception as e:
            print(f"Error saving track data: {e}")
            return False
    
    def load_track_from_csv(self):
        """Load track data from CSV file if it exists"""
        try:
            if not os.path.exists(self.track_data_csv):
                return False
                
            # Load data from CSV
            df = pd.read_csv(self.track_data_csv)
            
            # Convert NumPy arrays to CuPy
            self.centerline_x = cp.array(df['centerline_x'].values)
            self.centerline_y = cp.array(df['centerline_y'].values)
            self.tangent_x = cp.array(df['tangent_x'].values)
            self.tangent_y = cp.array(df['tangent_y'].values)
            self.normal_x = cp.array(df['normal_x'].values)
            self.normal_y = cp.array(df['normal_y'].values)
            self.curvature = cp.array(df['curvature'].values)
            self.inner_x = cp.array(df['inner_x'].values)
            self.inner_y = cp.array(df['inner_y'].values)
            self.outer_x = cp.array(df['outer_x'].values)
            self.outer_y = cp.array(df['outer_y'].values)
            
            # Set resolution based on loaded data
            self.resolution = len(self.centerline_x)
            
            return True
        except Exception as e:
            print(f"Error loading track data: {e}")
            return False
    
    def save_racing_line_to_csv(self):
        """Save racing line data to CSV file"""
        try:
            if self.racing_line_x is None or self.racing_line_y is None or self.racing_line_speed is None:
                print("No racing line data to save")
                return False
                
            # Convert CuPy arrays to NumPy for saving
            racing_line_data = {
                'racing_line_x': cp.asnumpy(self.racing_line_x),
                'racing_line_y': cp.asnumpy(self.racing_line_y),
                'racing_line_speed': self.racing_line_speed  # This might already be a Python list
            }
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(racing_line_data)
            df.to_csv(self.racing_line_csv, index=False)
            print(f"Saved racing line data to {self.racing_line_csv}")
            return True
        except Exception as e:
            print(f"Error saving racing line data: {e}")
            return False
    
    def load_racing_line_from_csv(self):
        """Load racing line data from CSV file if it exists"""
        try:
            if not os.path.exists(self.racing_line_csv):
                return False
                
            # Load data from CSV
            df = pd.read_csv(self.racing_line_csv)
            
            # Convert NumPy arrays to CuPy
            self.racing_line_x = cp.array(df['racing_line_x'].values)
            self.racing_line_y = cp.array(df['racing_line_y'].values)
            
            # racing_line_speed might be a list or numpy array
            self.racing_line_speed = df['racing_line_speed'].tolist()
            
            print(f"Loaded racing line data from {self.racing_line_csv}")
            return True
        except Exception as e:
            print(f"Error loading racing line data: {e}")
            return False
    
    def calculate_track_length(self):
        """Calculate the total length of the track."""
        dx = cp.diff(self.centerline_x)
        dy = cp.diff(self.centerline_y)
        distances = cp.sqrt(dx**2 + dy**2)
        return cp.sum(distances).get()  # Convert to CPU for native Python

    def get_nearest_point(self, x, y):
        """
        Find the nearest point on the centerline and its properties
        
        Args:
            x, y (float): Query point coordinates
        
        Returns:
            dict: Information about nearest track point
        """
        # Calculate distances to all centerline points
        distances = cp.sqrt((self.centerline_x - x)**2 + (self.centerline_y - y)**2)
        
        # Find index of nearest point
        idx = cp.argmin(distances).get()  # Convert to CPU
        
        return {
            'x': self.centerline_x[idx].get(),
            'y': self.centerline_y[idx].get(),
            'tangent_x': self.tangent_x[idx].get(),
            'tangent_y': self.tangent_y[idx].get(),
            'normal_x': self.normal_x[idx].get(),
            'normal_y': self.normal_y[idx].get(),
            'curvature': self.curvature[idx].get(),
            'distance': distances[idx].get(),
            'index': idx
        }
    
    def is_within_track(self, x, y):
        """
        Check if a point is within track boundaries
        
        Args:
            x, y (float): Point coordinates
        
        Returns:
            bool: True if point is within track boundaries
        """
        nearest = self.get_nearest_point(x, y)
        
        # Vector from nearest centerline point to query point
        dx = x - nearest['x']
        dy = y - nearest['y']
        
        # Project onto normal vector to get signed distance from centerline
        distance = dx * nearest['normal_x'] + dy * nearest['normal_y']
        
        return abs(distance) <= self.track_width / 2

    def calculate_theoretical_top_speed(self, vehicle):
        """
        Calculate theoretical top speed based on motor RPM and resistive forces
        """
        # Calculate maximum wheel speed from motor RPM
        max_wheel_rpm = vehicle.car["max_motor_rpm"] * vehicle.car["drivetrain_efficiency"]
        max_wheel_speed = (max_wheel_rpm * 2 * cp.pi * vehicle.car["wheel_radius"]) / 60  # m/s

        # Iteratively solve for top speed considering all forces
        v = max_wheel_speed
        tolerance = 0.1
        max_iterations = 10
        
        #with tqdm.tqdm(total=max_iterations, desc="Calculating top speed") as pbar:
        for iteration in range(max_iterations):
            # Calculate resistive forces
            F_drag = 0.5 * rho * vehicle.car["drag_coefficient"] * vehicle.car["frontal_area"] * v**2
            F_roll = vehicle.car["rolling_resistance_coeff"] * vehicle.car["mass"] * g
            F_down = 0.5 * rho * vehicle.car["down_force"] * vehicle.car["frontal_area"] * v**2

            # Calculate available engine force at current speed
            omega = (v / vehicle.car["wheel_radius"]) * 60 / (2 * cp.pi)  # Current RPM
            rpm_ratio = omega / vehicle.car["max_motor_rpm"]
            available_torque = vehicle.car["engine_torque"] * (1 - rpm_ratio)  # Simple linear torque falloff
            F_engine = (available_torque * vehicle.car["drivetrain_efficiency"]) / vehicle.car["wheel_radius"]

            # Net force
            F_net = F_engine - F_drag - F_roll

            if abs(F_net) < tolerance:
                break

            # Adjust speed estimate
            v = v + cp.sign(F_net) * 0.1
            #pbar.update(1)

        return v

    def calculate_max_speed(self, radius, vehicle):
        """
        Calculate maximum speed through a corner with tire model and aero effects
        """
        if abs(radius) < 1e-6:  # Nearly straight line
            return self.calculate_theoretical_top_speed(vehicle)
        
        # Get static loads
        mass_front = vehicle.car["mass"] * vehicle.car["cg_position"] / vehicle.car["wheelbase"]
        mass_rear = vehicle.car["mass"] - mass_front
        static_loads = [
            mass_front * g / 2,  # Front left
            mass_front * g / 2,  # Front right
            mass_rear * g / 2,   # Rear left
            mass_rear * g / 2    # Rear right
        ]

        def force_balance(v):
            # Calculate aero forces
            F_down = 0.5 * rho * vehicle.car["down_force"] * vehicle.car["frontal_area"] * v**2
            F_drag = 0.5 * rho * vehicle.car["drag_coefficient"] * vehicle.car["frontal_area"] * v**2

            # Calculate load transfer
            Fy = vehicle.car["mass"]*v**2 / radius #lateral force
            loadmodel = LoadTransfer()
            [dFz_front, dFz_rear, dGamma_front, dGamma_rear] = loadmodel.calculate_lateral(Fy) #vehicle.car["mass"] * lat_accel * vehicle.car["cg_height"] / vehicle.car["track_width"]

            # Update individual wheel loads with aero and load transfer
            aero_per_wheel = F_down / 4
            loads = [
                static_loads[0] + aero_per_wheel + dFz_front[0]/2,  # Front left
                static_loads[1] + aero_per_wheel + dFz_front[1]/2,  # Front right
                static_loads[2] + aero_per_wheel + dFz_rear[0]/2,  # Rear left
                static_loads[3] + aero_per_wheel + dFz_rear[1]/2   # Rear right
            ]

            # Calculate camber angles including load transfer effects
            gammas = [
                vehicle.car["gamma_front"] + dGamma_front[0],
                vehicle.car["gamma_front"] + dGamma_front[1],
                vehicle.car["gamma_rear"] + dGamma_rear[0],
                vehicle.car["gamma_rear"] + dGamma_rear[1]
            ]

            # Use tire model to calculate maximum lateral force
            max_lat_forces = []
            for i in range(4):
                # Skip if load is negative (wheel lift)
                if loads[i] <= 0:
                    return -float('inf')

                # Calculate slip angle for maximum lateral force
                if i < 2:  # Front wheels
                    max_lat = vehicle.tire_model.TireCornering.calculate_maximum_steer(loads[i], gammas[i])
                else:  # Rear wheels
                    max_lat = vehicle.tire_model.TireCornering.calculate_maximum_rear(loads[i], 0, gammas[i])
                max_lat_forces.append(max_lat)

            # Total available lateral force
            F_lat_available = sum(max_lat_forces)

            # Required lateral force for cornering
            F_lat_required = vehicle.car["mass"] * v**2 / radius

            return F_lat_available - F_lat_required

        # Find maximum speed using binary search
        v_low = 0
        v_high = self.calculate_theoretical_top_speed(vehicle)
        tolerance = 0.1
        
        #with tqdm.tqdm(desc="Finding max corner speed") as pbar:
        while (v_high - v_low) > tolerance:
            v_mid = (v_low + v_high) / 2
            if force_balance(v_mid) > 0:
                v_low = v_mid
            else:
                v_high = v_mid
                #pbar.update(1)

        return v_low
    
    def calculate_racing_line(self, vehicle, num_points=50, use_cache=True):
        """
        Calculate optimal racing line using optimization with multithreading
        
        Args:
            vehicle (VehicleDynamics): Vehicle object with properties
            num_points (int): Number of points to optimize
            use_cache (bool): Whether to load from cache if available
        """
        # Check if racing line data exists in cache and use it if requested
        if use_cache and self.load_racing_line_from_csv():
            print("Using cached racing line data")
            return
            
        import concurrent.futures
        
        with tqdm.tqdm(total=3, desc="Calculating racing line") as pbar:
            # Create equally spaced points along centerline
            indices = cp.linspace(0, len(self.centerline_x)-1, num_points, dtype=int)
            
            # Initial guess: points halfway between center and outer edge
            track_offset = cp.zeros(num_points)  # 0 = centerline, 1 = outer edge, -1 = inner edge
            
            def objective(offsets):
                # Convert to GPU array if not already
                offsets = cp.asarray(offsets)
                
                # Convert offsets to x,y coordinates
                x = self.centerline_x[indices] + offsets * self.normal_x[indices] * (self.track_width/2)
                y = self.centerline_y[indices] + offsets * self.normal_y[indices] * (self.track_width/2)
                
                # Calculate path length and curvature
                dx = cp.diff(x)
                dy = cp.diff(y)
                distances = cp.sqrt(dx**2 + dy**2)
                
                # Approximate curvature using three points
                curvatures = []
                for i in range(1, len(x)-1):
                    x1, y1 = x[i-1], y[i-1]
                    x2, y2 = x[i], y[i]
                    x3, y3 = x[i+1], y[i+1]
                    
                    # Calculate radius using circumscribed circle
                    d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if abs(d) < 1e-6:
                        curvatures.append(0)
                    else:
                        ux = ((x1*x1 + y1*y1)*(y2-y3) + (x2*x2 + y2*y2)*(y3-y1) + (x3*x3 + y3*y3)*(y1-y2))/d
                        uy = ((x1*x1 + y1*y1)*(x3-x2) + (x2*x2 + y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1))/d
                        r = cp.sqrt((x1-ux)**2 + (y1-uy)**2)
                        curvatures.append(1/r if r > 0 else 0)
                
                # Convert curvatures to GPU array
                curvatures = cp.array(curvatures)
                
                # Move arrays to CPU for compatibility with calculate_max_speed
                curvatures_cpu = cp.asnumpy(curvatures)
                
                # Calculate maximum speed at each point - multithreaded
                speeds = parallelize_speed_calculation(curvatures_cpu, vehicle, self.calculate_max_speed)
                
                speeds = cp.array([speeds[0]] + speeds + [speeds[-1]])  # Add endpoints
                
                # Estimate lap time using trapezoidal integration
                times = distances / ((speeds[:-1] + speeds[1:]) / 2)
                
                return cp.sum(times).get()  # Return scalar CPU value
            
            # Helper function for parallelizing curvature calculation
            def calculate_curvature(point_indices, points_x, points_y):
                results = []
                for i in point_indices:
                    if i <= 0 or i >= len(points_x) - 1:
                        continue
                        
                    x1, y1 = points_x[i-1], points_y[i-1]
                    x2, y2 = points_x[i], points_y[i]
                    x3, y3 = points_x[i+1], points_y[i+1]
                    
                    d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if abs(d) < 1e-6:
                        results.append((i, 0))
                    else:
                        ux = ((x1*x1 + y1*y1)*(y2-y3) + (x2*x2 + y2*y2)*(y3-y1) + (x3*x3 + y3*y3)*(y1-y2))/d
                        uy = ((x1*x1 + y1*y1)*(x3-x2) + (x2*x2 + y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1))/d
                        r = np.sqrt((x1-ux)**2 + (y1-uy)**2)
                        results.append((i, 1/r if r > 0 else 0))
                
                return results
            
            # Helper function for parallelizing speed calculation
            def calculate_speed_chunk(curvatures_chunk, vehicle, speed_func):
                return [speed_func(1/c if c != 0 else float('inf'), vehicle) for c in curvatures_chunk]
            
            # Function to parallelize speed calculation
            def parallelize_speed_calculation(curvatures, vehicle, speed_func, num_workers=None):
                if num_workers is None:
                    import multiprocessing
                    num_workers = multiprocessing.cpu_count()
                
                # Split curvatures into chunks
                chunk_size = max(1, len(curvatures) // num_workers)
                chunks = [curvatures[i:i + chunk_size] for i in range(0, len(curvatures), chunk_size)]
                
                speeds = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(calculate_speed_chunk, chunk, vehicle, speed_func) for chunk in chunks]
                    for future in concurrent.futures.as_completed(futures):
                        speeds.extend(future.result())
                
                return speeds
            
            # Function to parallelize curvature calculation
            def parallelize_curvature_calculation(points_x, points_y, num_workers=None):
                if num_workers is None:
                    import multiprocessing
                    num_workers = multiprocessing.cpu_count()
                
                # Create chunks of indices
                indices = list(range(1, len(points_x) - 1))
                chunk_size = max(1, len(indices) // num_workers)
                chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
                
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(calculate_curvature, chunk, points_x, points_y) for chunk in chunks]
                    for future in concurrent.futures.as_completed(futures):
                        results.extend(future.result())
                
                # Sort by index and extract curvatures
                results.sort(key=lambda x: x[0])
                return [r[1] for r in results]
            
            # Constraint: keep points within track bounds
            bounds = [(-1, 1) for _ in range(num_points)]
            
            # We need to convert track_offset to CPU for scipy optimize
            track_offset_cpu = cp.asnumpy(track_offset)
            
            # Optimize racing line
            result = minimize(objective, track_offset_cpu, bounds=bounds, method='SLSQP')
            pbar.update(1)  # Update progress (1/3 complete)
            
            # Store optimized racing line
            optimized_offsets = cp.array(result.x)
            self.racing_line_x = self.centerline_x[indices] + optimized_offsets * self.normal_x[indices] * (self.track_width/2)
            self.racing_line_y = self.centerline_y[indices] + optimized_offsets * self.normal_y[indices] * (self.track_width/2)
            
            # Move racing line to CPU for multithreaded curvature calculation
            racing_line_x_cpu = cp.asnumpy(self.racing_line_x)
            racing_line_y_cpu = cp.asnumpy(self.racing_line_y)
            
            # Calculate curvatures using multithreading
            curvatures = parallelize_curvature_calculation(racing_line_x_cpu, racing_line_y_cpu)
            pbar.update(1)  # Update progress (2/3 complete)
            
            # Calculate speeds for each curvature using multithreading
            self.racing_line_speed = parallelize_speed_calculation(curvatures, vehicle, self.calculate_max_speed)
                
            # Pad speeds to match number of points
            self.racing_line_speed = [self.racing_line_speed[0]] + self.racing_line_speed + [self.racing_line_speed[-1]]
            pbar.update(1)  # Update progress (3/3 complete)
            
            # Save racing line to CSV for future use
            self.save_racing_line_to_csv()
    
    def plot_track(self, show_racing_line=True, ax=None, save_figure=False, filename=None):
        """Visualize the track and racing line"""
        import matplotlib.pyplot as plt
        
        # Move data to CPU for plotting
        outer_x = cp.asnumpy(self.outer_x)
        outer_y = cp.asnumpy(self.outer_y)
        inner_x = cp.asnumpy(self.inner_x)
        inner_y = cp.asnumpy(self.inner_y)
        centerline_x = cp.asnumpy(self.centerline_x)
        centerline_y = cp.asnumpy(self.centerline_y)
        
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        
        # Plot boundaries
        ax.plot(outer_x, outer_y, 'k-', label='Track boundary')
        ax.plot(inner_x, inner_y, 'k-')
        
        # Plot centerline
        ax.plot(centerline_x, centerline_y, 'r--', label='Centerline')
        
        # Plot racing line if available
        if show_racing_line and self.racing_line_x is not None:
            racing_line_x = cp.asnumpy(self.racing_line_x)
            racing_line_y = cp.asnumpy(self.racing_line_y)
            scatter = ax.scatter(racing_line_x, racing_line_y, 
                            c=self.racing_line_speed, cmap='viridis',
                            s=50, label='Racing Line')
            plt.colorbar(scatter, label='Maximum Speed (m/s)')
        
        ax.axis('equal')
        ax.set_title('Figure-8 Track Layout with Racing Line')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if save_figure:
            if filename is None:
                filename = f"{self.track_name}_visualization.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved track visualization to {filename}")

class LapSimulator:
    def __init__(self, track, vehicle):
        self.track = track
        self.vehicle = vehicle
        self.dt = 0.01  # simulation timestep
        
    def simulate_optimal_lap(self):
        """
        Find optimal lap time using optimal control
        """
        # First, calculate racing line if not already done
        if self.track.racing_line_x is None:
            self.track.calculate_racing_line(self.vehicle)
        
        # Get racing line points and target speeds
        points = list(zip(self.track.racing_line_x, self.track.racing_line_y))
        target_speeds = self.track.racing_line_speed
        
        # Initialize arrays to store results
        self.time = []
        self.position_x = []
        self.position_y = []
        self.velocity = []
        self.steering = []
        self.throttle = []
        self.brake = []
        
        # Start from first point with zero velocity
        self.vehicle.position = cp.array([points[0][0], points[0][1]])
        self.vehicle.velocity = 0
        self.vehicle.velocity_lateral = 0
        self.vehicle.heading = cp.arctan2(
            points[1][1] - points[0][1],
            points[1][0] - points[0][0]
        )
        self.vehicle.yaw_rate = 0
        
        current_time = 0
        lap_distance = 0
        target_idx = 1
        
        # Calculate track length for progress tracking
        track_length = self.track.calculate_track_length()
        # Initialize lap distance and progress bar
        lap_distance = 0
        progress_bar = tqdm(total=int(track_length), desc="Lap Progress", unit="m")
        
        # Continue until we complete a lap
        while lap_distance < track_length:
            target_point = points[target_idx]
            target_speed = target_speeds[target_idx]
            
            # Calculate distance to target point
            dist_to_target = cp.sqrt(
                (self.vehicle.position[0] - target_point[0])**2 +
                (self.vehicle.position[1] - target_point[1])**2
            )
            
            # Find optimal controls for this segment
            controls = self.optimize_segment_controls(target_point, target_speed, dist_to_target)
            
            # Simulate segment with optimal controls
            segment_states, segment_distance = self.simulate_segment(controls, dist_to_target, target_point)
            
            # Update lap distance and progress bar
            previous_lap_distance = lap_distance
            lap_distance += segment_distance
            progress_bar.update(int(lap_distance - previous_lap_distance))
            
            # Calculate actual time for each state based on distance and velocity
            prev_pos = self.vehicle.position
            prev_time = current_time
            
            for state in segment_states:
                # Get current position and velocity from state
                current_pos = state['position']
                current_vel = state['velocity']
                
                if current_vel > 0:  # Avoid division by zero
                    # Calculate distance between previous and current positions
                    step_distance = cp.sqrt(
                        (current_pos[0] - prev_pos[0])**2 + 
                        (current_pos[1] - prev_pos[1])**2
                    )
                    
                    # Calculate time based on average velocity
                    # Use average velocity between previous and current states for better accuracy
                    avg_velocity = (self.velocity[-1] if self.velocity else 0 + current_vel) / 2
                    if avg_velocity > 0:  # Avoid division by zero
                        step_time = step_distance / avg_velocity
                    else:
                        step_time = self.dt  # Fallback to dt if velocity is zero
                    
                    current_time += step_time
                else:
                    # If velocity is zero, use acceleration model or fallback to dt
                    current_time += self.dt
                
                # Store results
                self.time.append(current_time)
                self.position_x.append(float(current_pos[0]))
                self.position_y.append(float(current_pos[1]))
                self.velocity.append(float(current_vel))
                self.steering.append(state['steering'])
                self.throttle.append(state['throttle'])
                self.brake.append(state['brake'])
                
                # Update previous position for next iteration
                prev_pos = current_pos
            
            # Move to next target point
            target_idx = (target_idx + 1) % len(points)
        
        # Close the progress bar when done
        progress_bar.close()
        
        return current_time  # Total lap time
    
    def optimize_segment_controls(self, target_point, target_speed, dist_to_target):
        """
        Find optimal controls to reach next racing line point
        """
        def objective(controls):
            # Unpack control inputs
            throttle, brake, steering = controls
            
            # Simulate vehicle response
            Fx, Fy = self.vehicle.calculate_forces(throttle, brake, steering)
            
            # Store initial state
            init_pos = self.vehicle.position.copy()
            init_vel = self.vehicle.velocity
            
            # Update vehicle state
            self.vehicle.update_state(Fx, Fy, self.dt)
            
            # Calculate error terms
            pos_error = cp.sqrt(
                (self.vehicle.position[0] - target_point[0])**2 +
                (self.vehicle.position[1] - target_point[1])**2
            )
            vel_error = abs(self.vehicle.velocity - target_speed)
            
            # Add time penalty
            time_penalty = float(pos_error) / max(float(self.vehicle.velocity), 1.0)  # Approximate time to target
            
            # Restore initial state
            self.vehicle.position = init_pos
            self.vehicle.velocity = init_vel
            
            # Return CPU values for scipy.optimize (which doesn't support GPU arrays)
            return float(time_penalty + pos_error + vel_error)
        
        # Initial guess
        x0 = [0.5, 0, 0]  # throttle, brake, steering
        
        # Bounds for controls
        bounds = [(0, 1), (0, 1), (-0.5, 0.5)]  # throttle, brake, steering limits
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        
        return result.x
    
    def simulate_segment(self, controls, dist_to_target, target_point):
	    """
	    Simulate vehicle motion for one segment with given controls - parallelized version
	    
	    Args:
	        controls (tuple): Throttle, brake, and steering inputs
	        dist_to_target (float): Distance to target point
	        target_point (tuple): X,Y coordinates of target point
	    """
	    throttle, brake, steering = controls
	    max_steps = int(5.0 / self.dt)  # Maximum 5 seconds per segment
	    
	    # Pre-allocate arrays for better performance
	    positions = cp.zeros((max_steps + 1, 2))
	    velocities = cp.zeros(max_steps + 1)
	    steerings = cp.ones(max_steps + 1) * steering
	    throttles = cp.ones(max_steps + 1) * throttle
	    brakes = cp.ones(max_steps + 1) * brake
	    
	    # Set initial position
	    positions[0] = self.vehicle.position
	    velocities[0] = self.vehicle.velocity
	    
	    # Target point as array for vectorized distance calculation
	    target_array = cp.array(target_point)
	    
	    # Prepare for parallel force calculation
	    num_processes = min(8, os.cpu_count() or 4)  # Use at most 8 processes
	    
	    # Create a pool executor
	    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
	        # Prepare batches for parallel processing
	        batch_size = max(10, max_steps // num_processes)
	        batches = [(i, min(i + batch_size, max_steps)) for i in range(0, max_steps, batch_size)]
	        
	        # Function to process a batch of simulation steps
	        def process_batch(start_idx, end_idx):
	            batch_positions = []
	            batch_velocities = []
	            last_vehicle_state = None
	            
	            # Initialize with the last known state or initial state
	            if start_idx == 0:
	                vehicle_state = {
	                    'position': self.vehicle.position.copy(),
	                    'velocity': self.vehicle.velocity,
	                    'heading': self.vehicle.heading
	                }
	            else:
	                # Use the state from the end of the previous batch
	                vehicle_state = last_vehicle_state
	            
	            # Process each step in the batch
	            for step in range(start_idx, end_idx):
	                # Calculate forces
	                Fx, Fy = self.vehicle.calculate_forces(throttle, brake, steering)
	                
	                # Store current position and velocity
	                batch_positions.append(vehicle_state['position'].copy())
	                batch_velocities.append(vehicle_state['velocity'])
	                
	                # Update vehicle state
	                self._update_vehicle_state(vehicle_state, Fx, Fy, self.dt)
	                
	                # Check if we've reached the target point
	                if cp.sqrt(cp.sum((vehicle_state['position'] - target_array)**2)) < 1.0:
	                    break
	            
	            return {
	                'positions': batch_positions,
	                'velocities': batch_velocities,
	                'last_state': vehicle_state
	            }
	        
	        # Submit all batches for parallel processing
	        future_to_batch = {executor.submit(process_batch, start, end): (start, end) 
	                          for start, end in batches}
	        
	        # Process completed batches
	        step_count = 0
	        for future in concurrent.futures.as_completed(future_to_batch):
	            start, end = future_to_batch[future]
	            result = future.result()
	            
	            # Copy results to our pre-allocated arrays
	            batch_len = len(result['positions'])
	            positions[start:start+batch_len] = result['positions']
	            velocities[start:start+batch_len] = result['velocities']
	            
	            step_count = max(step_count, start + batch_len)
	            
	            # Check if we reached the target in this batch
	            if batch_len < (end - start):
	                break
	    
	    # Trim arrays to actual step count
	    positions = positions[:step_count]
	    velocities = velocities[:step_count]
	    steerings = steerings[:step_count]
	    throttles = throttles[:step_count]
	    brakes = brakes[:step_count]
	    
	    # Calculate total distance
	    position_diffs = positions[1:] - positions[:-1]
	    step_distances = cp.sqrt(cp.sum(position_diffs**2, axis=1))
	    total_distance = float(cp.sum(step_distances))
	    
	    # Convert to list of state dictionaries for compatibility
	    states = [
	        {
	            'position': positions[i].copy(),
	            'velocity': float(velocities[i]),
	            'steering': float(steerings[i]),
	            'throttle': float(throttles[i]),
	            'brake': float(brakes[i])
	        }
	        for i in range(step_count)
	    ]
	    
	    return states, total_distance
    
    def plot_results(self):
        """
        Visualize simulation results
        """
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot track and vehicle path
        self.track.plot_track(show_racing_line=True, ax=ax1)
        ax1.plot(self.position_x, self.position_y, 'g-', label='Simulated Path')
        ax1.set_title('Track and Vehicle Path')
        
        # Plot velocity profile
        ax2.plot(self.time, self.velocity)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        
        # Plot control inputs
        ax3.plot(self.time, self.throttle, 'g-', label='Throttle')
        ax3.plot(self.time, self.brake, 'r-', label='Brake')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Input')
        ax3.set_title('Throttle and Brake Inputs')
        ax3.legend()
        ax3.grid(True)
        
        # Plot steering input
        ax4.plot(self.time, self.steering, 'b-')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Steering Angle (rad)')
        ax4.set_title('Steering Input')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()


# Create track and vehicle
track = Track(scale=100, track_name="figure8", load_from_cache=True)
print("track data generated")
track_length = track.calculate_track_length()
print(f"Track length: {track_length:.2f} meters")



vehicle = VehicleDynamics(car_data, track)  # Pass the track object
print("vehicle model generated")


# Create simulator
simulator = LapSimulator(track, vehicle)
print("simulation created")

# Run simulation
lap_time = simulator.simulate_optimal_lap()
print(f"Optimal lap time: {lap_time:.2f} seconds")

# Visualize results
simulator.plot_results()    
