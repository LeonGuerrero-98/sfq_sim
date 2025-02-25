import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch 
from .pulse_functions import *
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import json
from typing import Union
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Video


class create_qutrit:

    def __init__(
        self,
        qfreq: float,
        anharm: float,
        initial_state:Qobj = basis(3,0)  
    ):
        self.qfreq = qfreq*1e9*2*np.pi
        self.anharm = anharm
        self.pulse_width = 2e-12
        self.t_delay = 0
        self.steps = 3e5
        self.progress = True
        self.int_jitter = 0

        #if initial state is 2d convert to 3d
        if initial_state.shape == (2,1):
            self.qutrit_state = Qobj(np.array([initial_state.full()[0], initial_state.full()[1], [0]]))
        elif initial_state.shape == (3,1):
            self.qutrit_state = initial_state
        else:
            raise ValueError("Initial state must be a 2D or 3D Qobj")
        

        # Ensure qubit frequency is a sensible value
        if self.qfreq/(1e9*2*np.pi) < 0.01 or self.qfreq/(1e9*2*np.pi) > 8:
            raise ValueError("Qubit frequency (in GHz) must be a sensible value >0.01 or <8")

    def set_qutrit_state(self, state: Union[Qobj, list]):
        """
        Set the state of the qutrit.

        Parameters
        ----------
        state : Union[Qobj, list]
            The initial state of the qutrit. It can be either a Qobj or a list.
            If the state is a Qobj, it must have a shape of (2, 1) or (3, 1).
            If the state is a list, it must have a length of 2 or 3.

        Raises
        ------
        ValueError
            If the initial state is not a 2D or 3D Qobj or list.
        """
        #if initial state is 2d convert to 3d
        if isinstance(state, Qobj):
            if state.shape == (2,1):
                self.qutrit_state = Qobj(np.array([state.full()[0], state.full()[1], [0]]))
            elif state.shape == (3,1):
                self.qutrit_state = state
            else:
                raise ValueError("Initial state must be a 2D or 3D Qobj")
        elif isinstance(state, list):
            if len(state) == 2:
                qutrit_state = Qobj(np.array([state[0], state[1], 0]))
            elif len(state) == 3:
                qutrit_state = Qobj(np.array(state))
            else:
                raise ValueError("Initial state must be a 2D or 3D list")
            
            if qutrit_state.norm() != 0:
                self.qutrit_state = qutrit_state.unit()
            else:
                raise ValueError("Initial state must be a non-zero state")
                              
    def apply_qutrit_sfq_Rygate(self, n:int, theta:float, pulse_width:float = 2e-12, t_delay:float = 0, steps:float = 3e5, progress:bool = True, int_jitter:float = 0):
        """
        Apply an SFQ (Single Flux Quantum) R_y gate to a qutrit.
        Parameters:
        n (int): The number of pulses.
        theta (float): The rotation angle.
        pulse_width (float, optional): The width of each pulse in seconds. Default is 2e-12.
        t_delay (float, optional): The delay time before starting the pulses in seconds. Default is 0.
        steps (float, optional): The number of simulation steps. Default is 3e5.
        progress (bool, optional): Whether to show progress during the simulation. Default is True.
        int_jitter (float, optional): The intrinsic jitter in the system. Default is 0.
        Raises:
        ValueError: If the anharmonicity is not within the sensible range (0 < anharm < 20).
        Returns:
        None
        """
 
        self.n = n
        self.theta = theta
        self.pulse_width = pulse_width
        self.t_delay = t_delay + 1/(self.qfreq*1e9) # ensures pulses start in time with qubit oscillation
        self.steps = steps
        self.progress = progress
        self.int_jitter = int_jitter

        # Ensure anharmonicity is a sensible value
        if self.anharm < 0 or self.anharm > 20:
            raise ValueError("Anharmonicity must be a sensible value >0 or <20")


        self.result = sfq_qutrit_Ry(
            self.n, self.anharm, self.qfreq,self.qutrit_state,self.theta,self.pulse_width,self.t_delay, n_steps=self.steps, progress=self.progress, int_jit = self.int_jitter)
 
        self.qutrit_state = self.result["psi"][-1]
        self.t = self.result["t"]
        self.pulse = self.result["pulse"]

    def apply_qutrit_sfq_gate_RF(self, n:int, gate:str, theta:float, pulse_width:float = 2e-12, t_delay:float = 0, steps:float = 3e5, progress:bool = True, int_jitter:float = 0):
        """
        Apply a qutrit SFQ gate using RF pulses.
        Parameters:
        n (int): The qutrit index.
        gate (str): The type of gate to apply. Must be 'x', 'y', 'X', or 'Y'.
        theta (float): The rotation angle for the gate.
        pulse_width (float, optional): The width of the pulse in seconds. Default is 2e-12.
        t_delay (float, optional): The delay time before applying the gate. Default is 0.
        steps (float, optional): The number of steps for the simulation. Default is 3e5.
        progress (bool, optional): Whether to show progress during the simulation. Default is True.
        int_jitter (float, optional): The internal jitter for the simulation. Default is 0.
        Raises:
        ValueError: If the gate is not one of 'x', 'y', 'X', or 'Y'.
        Returns:
        None
        """
        #gate must be x or y or X or Y
        if gate not in ["x","y","X","Y"]:
            raise ValueError("Gate must be x or y or X or Y")
        if gate == "x" or "X" and theta > 0:
            self.t_delay = t_delay + (0.5*np.pi)/self.qfreq
            self.gate = "X"
        elif gate == "y" or "Y" and theta > 0:
            self.t_delay = t_delay
            self.gate = "Y"
        elif gate == "x" or "X" and theta < 0:
            self.t_delay = t_delay + (1.5*np.pi)/self.qfreq
            self.gate = "X"
        elif gate == "y" or "Y" and theta < 0:
            self.t_delay = t_delay + np.pi/self.qfreq
            self.gate = "Y"

        self.n = n
        self.theta = theta
        self.pulse_width = pulse_width
        self.steps = steps
        self.progress = progress
        self.int_jitter = int_jitter

        self.result = sfq_qutrit_RF(
            self.n, self.anharm, self.qfreq,self.qutrit_state,self.theta,gate,self.pulse_width,self.t_delay, n_steps=self.steps, progress=self.progress, int_jit = self.int_jitter)
 
        self.qutrit_state = self.result["psi"][-1]
        self.t = self.result["t"]
        self.pulse = self.result["pulse"]
    
    def plot_probs(self, include_pulse=False):
        """
        Plots the state probabilities over time, with an optional pulse plot.
        Parameters:
        -----------
        include_pulse : bool, optional
            If True, includes the pulse plot in a second subplot. Default is False.
        Raises:
        -------
        ValueError
            If there is no sweep data to plot. Ensure `apply_qutrit_sfq_gate()` has been run first.
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot(s).
        """
        #verify there is sweep data to plot
        if not hasattr(self, 'result'):
            raise ValueError("No data to plot. Run apply_qutrit_sfq_gate() first.")
        if include_pulse == False:
            fig, ax = plt.subplots()

            ax.plot(self.t*1e9, self.result["P0"], label=r"$P_{|0\rangle}$")
            ax.plot(self.t*1e9, self.result["P1"], label=r"$P_{|1\rangle}$")
            ax.plot(self.t*1e9, self.result["P2"], label=r"$P_{|2\rangle}$")

            ax.grid(True)
            ax.legend(loc="best")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("State Probabilities")
            
        if include_pulse == True:
            # Create a figure with two subplots, stacked vertically
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

            # First subplot: probabilities
            ax1.plot(self.t*1e9, self.result["P0"], label=r"$P_{|0\rangle}$")
            ax1.plot(self.t*1e9, self.result["P1"], label=r"$P_{|1\rangle}$")
            ax1.plot(self.t*1e9, self.result["P2"], label=r"$P_{|2\rangle}$")

            ax1.grid(True)
            ax1.legend(loc="best")
            ax1.set_ylabel("State Probabilities")

            # Second subplot: pulse
            ax2.plot(self.t*1e9, self.pulse, label="Pulse", color="orange")
            ax2.grid(True)
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel(r"$V_{SFQ}$ (V)")

        return fig
    
    def plot_bloch(self, n_points:int = 1000):

        sx = self.result["sx"]
        sy = self.result["sy"]
        sz = self.result["sz"]

        #reduce number of points to plot

        sx = sx[::int(len(sx)/n_points)]
        sy = sy[::int(len(sy)/n_points)]
        sz = sz[::int(len(sz)/n_points)]

        b = Bloch() 
        b.add_points([sx, sy, sz],meth='l')

        b.xlabel = ['X', '']
        b.ylabel = ['Y', '']
        b.zlabel = ['Z', '']

        b.show()

    def animate_bloch(self, n_points: int = 1000, interval:int = 50, save:str = None):
        sx = self.result["sx"]
        sy = self.result["sy"]
        sz = self.result["sz"]

        # Subsample points to reduce the number of frames
        sx = sx[::int(len(sx)/n_points)]
        sy = sy[::int(len(sy)/n_points)]
        sz = sz[::int(len(sz)/n_points)]

        # Adjust the total number of frames based on the reduced points
        total_frames = len(sx)

        # Set up the figure, the axis, and the plot element
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        # Bloch sphere setup
        b = Bloch(fig=fig, axes=ax)
        
        # Initialize the points
        points = [[], [], []]

        def init():
            b.clear()
            b.render()
            return []
        progress_bar = tqdm(total=total_frames, desc="Animating", unit="frame")
        # Update function for each frame in the animation
        def update(frame):
            b.clear()

            # Add points one by one for animation
            points[0].append(sx[frame])
            points[1].append(sy[frame])
            points[2].append(sz[frame])
            
            b.add_points(points, meth='l')

            # Draw the Bloch sphere
            b.render()
            progress_bar.update(1)

            return []

        # Create the animation
        ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, interval=interval, blit=True)
        if save:
            ani.save(save,writer = 'ffmpeg', fps = 30)
            progress_bar.close()
            return Video(save)
        else:
        # Display animation in Jupyter Notebook
            progress_bar.close()
            return HTML(ani.to_jshtml())
        
    def anharm_sweep(self, anharms: tuple,n: int, theta: float,initial_state: Qobj = basis(3,0), multicore: int = 0, sweep_progress: bool = False):
    
        """
        Perform an anharmonicity sweep on a qutrit system.
        Parameters:
        -----------
        anharms : tuple
            A tuple containing the anharmonicity values to sweep over.
        n : int
            The number of steps in the sweep.
        theta : float
            The rotation angle for the Ry gate.
        initial_state : Qobj, optional
            The initial state of the qutrit, default is the ground state (basis(3,0)).
        multicore : int, optional
            The number of cores to use for parallel processing. If 0, no parallel processing is used. Default is 0.
        sweep_progress : bool, optional
            If True, display a progress bar for the sweep. Default is False.
        Returns:
        --------
        None
        """
        self.anharms = anharms
        self.n = n
        self.theta = theta
        self.set_qutrit_state(initial_state)


 
        if multicore == 0:
            results = sfq_qutrit_Ry_anharm_sweep(
                self.n, anharms, self.qfreq, self.qutrit_state, self.theta, self.pulse_width, self.t_delay, self.steps, sweep_progress, int_jit=0
            )
        else:
            with multiprocessing.Pool(processes=multicore) as pool:
                params = [(self.n, self.qutrit_state, self.theta, self.pulse_width, i, self.qfreq, self.t_delay, self.steps) for i in anharms]
                results_list = list(tqdm(pool.imap(compute_fid_Ry_anharm, params), total=len(anharms)))
            results = {key: [res[key] for res in results_list] for key in ["fids", "P2", "P1", "P0", "psi"]}

        self.fids, self.P2_f, self.P1_f, self.P0_f, self.psi_f = results["fids"], results["P2"], results["P1"], results["P0"], results["psi"]

    def jitter_sweep(self, jitter_sigmas, n:int ,theta:float, initial_state:Qobj = basis(3,0) ,multicore: int = 0, sweep_progress: bool = True, averaging: int = 1):
        
        """
        Perform a jitter sweep on the qutrit state.
        Parameters:
        -----------
        jitter_sigmas : list
            List of jitter standard deviation values to sweep over.
        n : int
            Number of steps in the sweep.
        theta : float
            Rotation angle for the Ry gate.
        initial_state : Qobj, optional
            Initial state of the qutrit, default is basis(3, 0).
        multicore : int, optional
            Number of cores to use for parallel processing. Default is 0 (no parallel processing).
        sweep_progress : bool, optional
            Whether to show progress of the sweep. Default is True.
        averaging : int, optional
            Number of times to average the results. Default is 1.
        Returns:
        --------
        None
        """
        self.jitter_sigmas = jitter_sigmas
        self.theta = theta
        self.n = n
        self.set_qutrit_state(initial_state)

        if multicore == 0:
            results = sfq_qutrit_Ry_jitter_sweep(self.n, self.anharm,self.qfreq, self.qutrit_state,self.theta,self.jitter_sigmas,averaging,
                                                                                                                                                                                    self.pulse_width, self.t_delay, self.steps, progress = sweep_progress)
            
            self.fids_mean = results["fids_mean"]
            self.P2_mean = results["P2_mean"]
            self.P1_mean = results["P1_mean"]
            self.P0_mean = results["P0_mean"]
            self.fids_err = results["fids_err"]
            self.P2_err = results["P2_err"]
            self.P1_err = results["P1_err"]
            self.P0_err = results["P0_err"]

        else:
            with multiprocessing.Pool(processes=multicore) as pool:
                    params = [(self.n,self.anharm,self.qfreq,self.qutrit_state,self.theta,self.pulse_width,self.t_delay,self.steps,sweep_progress,s,True,averaging) for s in jitter_sigmas]
                    results = list(tqdm(pool.imap(compute_fid_Ry_jitter, params), total=len(jitter_sigmas)))
            self.fids_mean, self.P2_mean, self.P1_mean, self.P0_mean = [], [], [], []
            self.fids_err, self.P2_err, self.P1_err, self.P0_err = [], [], [], []
            for res in results:
                self.fids_mean.append(res["fids_mean"])
                self.P2_mean.append(res["P2_mean"])
                self.P1_mean.append(res["P1_mean"])
                self.P0_mean.append(res["P0_mean"])
                self.fids_err.append(res["fids_err"])
                self.P2_err.append(res["P2_err"])
                self.P1_err.append(res["P1_err"])
                self.P0_err.append(res["P0_err"])
                
    def plot_anharm_sweep_results(self,log:bool = False,infidelity:bool = False):
        if not hasattr(self, 'fids'):
            raise ValueError("No sweep data to plot. Run anharm_sweep() first.")
        
        fig, ax = plt.subplots()
        if infidelity:
            ax.plot(self.anharms, 1-np.array(self.fids), label="Infidelity")
        else:
            ax.plot(self.anharms, self.fids, label="Fidelity")
        if log:
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.grid(True)
        ax.set_xlabel("Anharmonicity")
        ax.set_ylabel("Fidelity")

        return fig
    
    def plot_jitter_sweep_results(self,log:bool = False,infidelity:bool = False):
        if not hasattr(self, 'fids_mean'):
            raise ValueError("No sweep data to plot. Run jitter_sweep() first.")
        
        fig, ax = plt.subplots()
        if infidelity:
            ax.errorbar(self.jitter_sigmas, 1-np.array(self.fids_mean), yerr = self.fids_err, label="Infidelity", capsize= 2.5, marker = '.',markersize = 1.5)
        else:
            ax.errorbar(self.jitter_sigmas, self.fids_mean, yerr = self.fids_err, label="Fidelity", capsize= 2.5, marker = '.',markersize = 1.5)
        if log:
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.grid(True)
        ax.legend(loc="best")
        ax.set_xlabel("Jitter Sigma")
        ax.set_ylabel("Fidelity")

        return fig

    def save_anharm_sweep_results(self,save_folder:str = None):
        #verify there is sweep data to save
        if not hasattr(self, 'fids'):
            raise ValueError("No sweep data to save. Run anharm_sweep() first.")
        
        #Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_freq_str = f"{round(self.qfreq / (np.pi*1e9), 2)}GHz"
        if self.theta == np.pi:
            gate_str = "pi"
        elif self.theta == np.pi/2:
            gate_str = "pi_2"
        else:
            gate_str = f"{round(self.theta, 2)}"
        filename = f"Y" + gate_str + f"_anharm_sweep_{self.n}_pulses_" + q_freq_str + f"_{timestamp}.json"

        # Create a dictionary to save, converting ndarrays to lists
        data_dict = {
            "fids": self.fids.tolist() if isinstance(self.fids, np.ndarray) else self.fids,
            "P2_f": self.P2_f.tolist() if isinstance(self.P2_f, np.ndarray) else self.P2_f,
            "P1_f": self.P1_f.tolist() if isinstance(self.P1_f, np.ndarray) else self.P1_f,
            "P0_f": self.P0_f.tolist() if isinstance(self.P0_f, np.ndarray) else self.P0_f,
            "anharm": self.anharms.tolist() if isinstance(self.anharms, np.ndarray) else self.anharms
        }
        
        # Save the dictionary to a json file
        if save_folder is None:
            file_path = filename
        else:
            file_path = f"{save_folder}/{filename}"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f)

        print(f"Data saved to {file_path}")

    def save_jitter_sweep_results(self,save_folder:str = None):
        #verify there is sweep data to save
        if not hasattr(self, 'fids_mean'):
            raise ValueError("No sweep data to save. Run jitter_sweep() first.")
        
        #Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_freq_str = str(round(self.qfreq/(np.pi*1e9)),2) + "GHz"
        filename = f"{self.gate}_jitter_sweep_{self.n}_pulses_" + q_freq_str + f"_{timestamp}.json"

        #Create a dictionary to save
        data_dict = {"jitter_sigmas": self.jitter_sigmas,"fids": self.fids_mean, "fids_err": self.fids_err, "P2_mean": self.P2_mean, "P2_err": self.P2_err, "P1_mean": self.P1_mean, "P1_err": self.P1_err, "P0_mean": self.P0_mean, "P0_err": self.P0_err}
        
        #Save the dictionary to a json file
        if save_folder is None:
            with open(filename, "w") as f:
                json.dump(data_dict, f)
        else:
            with open(save_folder + filename, "w") as f:
                json.dump(data_dict, f)

        print(f"Data saved to {filename}")
        





