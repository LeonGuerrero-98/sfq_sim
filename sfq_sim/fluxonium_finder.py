import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
import math
from tqdm import tqdm
from .fluxonium_finder_functions import *

scq.settings.T1_DEFAULT_WARNING=False

class fluxonium_finder:

    def __init__(
            self,
            EJ,
            qfreq):
        self.EJ = EJ
        self.qfreq = qfreq


    def param_sweep(self, n_steps = 100,EC_range = None, EL_range = None, freq_tol = 0.05, anharm_tol = 0.05, multicore = 0, qfreqs = None):
        default_EC_range = np.linspace(0.5*self.EJ,2*self.EJ,n_steps)
        default_EL_range = np.linspace(0.005*self.EJ,0.5*self.EJ,n_steps)
        if EC_range is None:
            EC_range = default_EC_range
        if EL_range is None:
            EL_range = default_EL_range

        self.target_anharms = [n+0.5 for n in range(10)]
        self.freq_tolerance = freq_tol
        self.anharm_tolerance = anharm_tol

        self.qfreqs = qfreqs
        if self.qfreqs is not None:
            inputq = self.qfreqs
        else: 
            inputq = self.qfreq

        self.results_q , self.results_a, self.potential_qubits = EC_EL_sweep(self.EJ, inputq,EC_range, EL_range, freq_tol ,anharm_tol, multicore=multicore)


    def plot_qfreq_sweep(self):
        fig, ax = plt.subplots()   
        if self.qfreqs is not None:
            colormap = plt.get_cmap('viridis', len(self.qfreqs))
            for q in self.qfreqs:
                ec_plot = []
                el_plot = []
                for i in range(len(self.results_q["qfreq"])):
                    if math.isclose(self.results_q["qfreq"][i],q,rel_tol=0.1):
                        ec_plot.append(self.results_q["EC"][i])
                        el_plot.append(self.results_q["EL"][i])
                if ec_plot != []:
                    ax.plot(ec_plot,el_plot,label='$\omega_{10}$ = '+ str(q) + " GHz",c=colormap(self.qfreqs.index(q))) 
        elif self.qfreqs is None:
            ax.plot(self.results_q['EC'],self.results_q['EL'],label=r'$\omega_{10}$ = '+str(self.qfreq))

        return fig
        


    def plot_sweep(self):
        fig, ax = plt.subplots()

        #find number of matching anharms in results
        self.target_anharms = [n+0.5 for n in range(10)]
        an = []
        test_against_anharms = self.target_anharms
        for a in self.target_anharms:
            for i in range(len(self.results_a["anharm"])):
                    if math.isclose(self.results_a["anharm"][i],a,rel_tol=0.1):
                        if a not in an:
                            an.append(a)

                            

        colormap = plt.get_cmap('viridis', len(an))
        for a in self.target_anharms:
            ec_plot = []
            el_plot = []
            for i in range(len(self.results_a["anharm"])):
                if math.isclose(self.results_a["anharm"][i],a,rel_tol=0.1):
                    ec_plot.append(self.results_a["EC"][i])
                    el_plot.append(self.results_a["EL"][i])
            if ec_plot != []:
                ax.scatter(ec_plot,el_plot,label='Anharmonicity = '+str(a),c=colormap(self.target_anharms.index(a)))     

                
        if self.qfreqs is not None:
            colormap = plt.get_cmap('viridis', len(self.qfreqs))
            for q in self.qfreqs:
                ec_plot = []
                el_plot = []
                for i in range(len(self.results_q["qfreq"])):
                    if math.isclose(self.results_q["qfreq"][i],q,rel_tol=0.1):
                        ec_plot.append(self.results_q["EC"][i])
                        el_plot.append(self.results_q["EL"][i])
                if ec_plot != []:
                    ax.plot(ec_plot,el_plot,label='$\omega_{10}$ = '+ str(q) + " GHz",c=colormap(self.qfreqs.index(q))) 
        elif self.qfreqs is None:
            ax.plot(self.results_q['EC'],self.results_q['EL'],label=r'$\omega_{10}$ = '+str(self.qfreq))
            
        if self.potential_qubits is not None:
            ax.scatter(self.potential_qubits['EC'],self.potential_qubits['EL'],marker = 'x',label='Potential qubits')
        ax.set_xlabel('EC')
        ax.set_ylabel('EL')
        #put legend outside plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return fig
    
    def contour_sweep(self, n_steps:int = 100,EC_range = None, EL_range = None, qfreqs = None, anharms = None, plot:str = "both", fill:bool = True, alpha:float = 1, save = False):
        default_EC_range = np.linspace(0.5*self.EJ,2.5*self.EJ,n_steps)
        default_EL_range = np.linspace(0.005*self.EJ,0.5*self.EJ,n_steps)
        if EC_range is None:
            EC_range = default_EC_range
        elif len(EC_range) == 2:
            EC_range = np.linspace(EC_range[0], EC_range[1], n_steps)
        else:
            raise ValueError(f"EC_range should be a list of length 2")
        
        if EL_range is None:
            EL_range = default_EL_range
        elif len(EL_range) == 2:
            EL_range = np.linspace(EL_range[0], EL_range[1], n_steps)
        else:
            raise ValueError(f"EL_range list of length 2")

        if anharms == None:
            self.target_anharms = [n+0.5 for n in range(10)]
        else:
            self.target_anharms = anharms

        self.qfreqs = qfreqs
        if self.qfreqs is not None:
            inputq = self.qfreqs
        else: 
            inputq = self.qfreq

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1") 

        valid_plot_options = {"both", "anharm", "qfreq", "none"}
        if plot not in valid_plot_options:
            raise ValueError(f"Invalid plot option: {plot}. Valid options are: {', '.join(valid_plot_options)}")

        self.potential_qubits = EC_EL_sweep_contour(self.EJ,EC_range,EL_range,inputq,self.target_anharms,plot = plot, fill = fill, alpha = alpha, save = save)

    def est_qubit_properties(self):
        if self.potential_qubits is None:
            raise AttributeError("The potential_qubits attribute is not set or initialized.")
        
        for i in range(len(self.potential_qubits)):
        # Access qubit properties for each "qubit_{i}"
            qubit_data = self.potential_qubits[f"qubit_{i}"]
            
            # Call the function and get the properties using EJ, EC, and EL from the qubit data
            qfreq, anharm, t1_eff, t2_eff, anharm_std = fluxonium_properties_from_params(
                qubit_data["EJ"],
                qubit_data["EC"],
                qubit_data["EL"]
            )
            
            # Update the dictionary for each qubit with new properties
            self.potential_qubits[f"qubit_{i}"].update({
                "qfreq": qfreq,
                "anharm": anharm,
                "t1_eff": t1_eff,
                "t2_eff": t2_eff,
                "anharm std": anharm_std
            })

    def compare_potential_qubits(self, shade_qfreq = False):
        if self.potential_qubits is None:
            raise AttributeError("The potential_qubits attribute is not set or initialized.")
        if self.potential_qubits["qubit_0"]["t1_eff"] == None:
            raise AttributeError("Make sure there are potential qubits and their properties have been estimated")
        # Create figure and axis

        t1 = []
        t2 = []
        n_s = []
        xtix = []
        qfreqs = []
        for i in range(len(self.potential_qubits)):
            t1.append(self.potential_qubits[f"qubit_{i}"]["t1_eff"]/1e6)
            t2.append(self.potential_qubits[f"qubit_{i}"]["t2_eff"]/1e6)
            n_s.append(1/self.potential_qubits[f"qubit_{i}"]["anharm std"])
            qfreqs.append(self.potential_qubits[f"qubit_{i}"]["qfreq"])
            xtix.append(f"q{i}")

        # Create figure and axis
        fig, ax1 = plt.subplots()

        # Plot the first two datasets (T1 and T2)
        ax1.plot(xtix,t1, "o--", label=r"$T_1$", color="blue")
        ax1.plot(xtix,t2, "o--", label=r"$T_2$", color="orange")
        ax1.set_ylabel("Coherence Times (ms)")
        ax1.tick_params(axis='y')

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(xtix,n_s, ".-", label=r"$\sigma_\eta^{-1}$", color="green")
        ax2.set_ylabel(r"Anharmonicity $\sigma^{-1}$")
        ax2.tick_params(axis='y')

        # Add legends for both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        # Shade regions based on qubit frequency if shade_qfreq is True
        if shade_qfreq and self.qfreqs is not None:
            for q in self.qfreqs:
                for i, qf_value in enumerate(qfreqs):  # Use 'qf_value' to avoid overwriting 'qf' list
                    if 1.05 > qf_value > q * 0.95:  # Adjust condition as needed
                        ax1.fill_between(
                            [i - 0.5, i + 0.5],
                            min(min(t1), min(t2)),
                            max(max(t1), max(t2)),
                            color="red",
                            alpha=0.2,
                        )
                        # Add a label only once to avoid duplicate legend entries
                        #ax1.text(i, max(max(t1), max(t2)), rf"$\omega_q \approx {q}$", ha="center", va="bottom", color="red")

        # Add a title for clarity
        plt.title(r"Qubits for $E_J$ = " + str(self.EJ) + " GHz")

        # Show the plot
        plt.tight_layout()  # Adjust layout to avoid overlapping labels
        return fig
                

        


        
