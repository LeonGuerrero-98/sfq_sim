import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch 
from .pulse_functions import *
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import json


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
        self.t_delay = 1e-11
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

        
    def set_qutrit_state(self, state:Qobj):
        #if initial state is 2d convert to 3d
        if state.shape == (2,1):
            self.qutrit_state = Qobj(np.array([state.full()[0], state.full()[1], [0]]))
        elif state.shape == (3,1):
            self.qutrit_state = state
        else:
            raise ValueError("Initial state must be a 2D or 3D Qobj")
                    
    def apply_qutrit_sfq_Rygate(self, n:int, theta:float, pulse_width:float = 2e-12, t_delay:float = 0, steps:float = 3e5, progress:bool = True, int_jitter:float = 0):
 
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

    

    def plot_probs(self, include_pulse=False):
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
        


    def anharm_sweep(self, anharms: tuple,n: int, theta: float,initial_state: Qobj = basis(3,0), multicore: int = 0, sweep_progress: bool = False):
        self.anharms = anharms
        self.n = n
        self.theta = theta
        self.set_qutrit_state(initial_state)


        if multicore == 0:
            # Sequential sweep without multiprocessing
            results = sfq_qutrit_Ry_anharm_sweep(
                self.n, anharms, self.qfreq,self.qutrit_state,self.theta,self.pulse_width,self.t_delay, self.steps, sweep_progress, int_jit=0
            )
        else:
            with multiprocessing.Pool(processes=multicore) as pool:
                # Wrap in tqdm to show progress
                params = [(self.n, self.qutrit_state,self.theta,self.pulse_width, i, self.qfreq, self.t_delay, self.steps) for i in anharms]
                results_list = list(tqdm(pool.imap(compute_fid_Ry, params), total=len(anharms)))
            # Combine results from multiprocessing
            results = {
                "fids": [],
                "P2": [],
                "P1": [],
                "P0": [],
                "psi_f": []
            }
            for res in results_list:
                results["fids"].append(res["fids"])
                results["P2"].append(res["P2"])
                results["P1"].append(res["P1"])
                results["P0"].append(res["P0"])
                results["psi_f"].append(res["psi"])
        # Unpack results
        self.fids = results["fids"]
        self.P2_f = results["P2"]
        self.P1_f = results["P1"]
        self.P0_f = results["P0"]
        self.psi_f = results["psi_f"]

    def jitter_sweep(self, jitter_sigmas, n:int ,theta:float, initial_state:Qobj = basis(3,0) ,multicore: int = 0, sweep_progress: bool = False, averaging: int = 1):
        self.jitter_sigmas = jitter_sigmas
        self.theta = theta
        self.n = n
        self.set_qutrit_state(initial_state)
        if multicore == 0:
            results = sfq_qutrit_Ry_jitter_sweep(self.n, self.anharm,self.qfreq, jitter_sigmas,averaging,
                                                                                                self.pulse_width, self.t_delay, self.steps, sweep_progress)
        else:
            def compute_fid(i):
                return sfq_qutrit_Ry_jitter_sweep(self.n, self.anharm, self.qfreq,self.qutrit_state,self.theta, i,averaging, self.pulse_width, self.t_delay, self.steps, sweep_progress)

            with multiprocessing.Pool(processes=multicore) as pool:
                results = list(tqdm(pool.imap(compute_fid, jitter_sigmas), total=len(jitter_sigmas)))

        self.fids_mean = results["fids_mean"]
        self.P2_mean = results["P2_mean"]
        self.P1_mean = results["P1_mean"]
        self.P0_mean = results["P0_mean"]
        self.fids_err = results["fids_err"]
        self.P2_err = results["P2_err"]
        self.P1_err = results["P1_err"]
        self.P0_err = results["P0_err"]



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
        ax.grid(True)
        ax.set_xlabel("Anharmonicity")
        ax.set_ylabel("Fidelity")

        return fig
    
    def plot_jitter_sweep_results(self,log:bool = False,infidelity:bool = False):
        if not hasattr(self, 'fids_mean'):
            raise ValueError("No sweep data to plot. Run jitter_sweep() first.")
        
        fig, ax = plt.subplots()
        if infidelity:
            ax.errorbar(self.jitter_sigmas, 1-np.array(self.fids_mean), yerr = self.fids_err, label="Infidelity")
        else:
            ax.errorbar(self.jitter_sigmas, self.fids_mean, yerr = self.fids_err, label="Fidelity")
        if log:
            ax.set_yscale('log')
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
        q_freq_str = str(round(self.qfreq/1e9),2) + "GHz"
        filename = f"{self.gate}_sweep_{self.n}_pulses_" + q_freq_str + f"{timestamp}.json"

        #Create a dictionary to save
        data_dict = {"fids": self.fids, "P2_f": self.P2_f, "P1_f": self.P1_f, "P0_f": self.P0_f, "anharm": self.anharms}
        
        #Save the dictionary to a json file
        if save_folder is None:
            with open(filename, "w") as f:
                json.dump(data_dict, f)
        else:
            with open(save_folder + filename, "w") as f:
                json.dump(data_dict, f)

        print(f"Data saved to {filename}")

    def save_jitter_sweep_results(self,save_folder:str = None):
        #verify there is sweep data to save
        if not hasattr(self, 'fids_mean'):
            raise ValueError("No sweep data to save. Run jitter_sweep() first.")
        
        #Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_freq_str = str(round(self.qfreq/1e9),2) + "GHz"
        filename = f"{self.gate}_sweep_{self.n}_pulses_" + q_freq_str + f"{timestamp}.json"

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
        





