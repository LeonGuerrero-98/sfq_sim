import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
import math
from tqdm import tqdm
from .fluxonium_finder_functions import *
import json
from datetime import datetime

scq.settings.T1_DEFAULT_WARNING=False

class fluxonium_finder:

    def __init__(
            self,
            EJ,
            qfreq):
        self.EJ = EJ
        self.qfreq = qfreq


    def param_sweep(self, n_steps = 100,EC_range = None, EL_range = None, freq_tol = 0.05, anharm_tol = 0.05, multicore = 0, qfreqs = None):
        default_EC_range = np.linspace(0.5*self.EJ,2.5*self.EJ,n_steps)
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
    
    def contour_sweep(self, n_steps:int = 100,EC_range = None, EL_range = None, qfreqs = None, anharms = None,multicore:int = 0, verbose:bool = True):
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

        if multicore < 0:
            raise ValueError("multicore must be >= 0")
        
        self.potential_qubits, self.cont_sweep_results = EC_EL_sweep_contour(self.EJ,EC_range,EL_range,inputq,self.target_anharms,multicore,verbose)

    def est_qubit_properties(self, qfreq_tol = 0, anharm_tol = 0,averaging = 3,num_points = 250, Ic_sigma = 0.05,ec_el_tol = None, progress = True, array_var = False, array_junctions = 100):
        if self.potential_qubits is None:
            raise AttributeError("The potential_qubits attribute is not set or initialized.")
        
        if progress:
            qubit_iterator = tqdm(range(len(self.potential_qubits)), desc="Processing Qubits")
        else:
            qubit_iterator = range(len(self.potential_qubits))

        self.Ic_sigma = Ic_sigma

        for i in qubit_iterator:
            # Access qubit properties for each "qubit_{i}"
            qubit_data = self.potential_qubits[f"qubit_{i}"]
            
            # Call the function and get the properties using EJ, EC, and EL from the qubit data
            qfreq, anharm, t1_eff, t2_eff, anharm_std, anharm_std_err = fluxonium_properties_from_params(
                qubit_data["EJ"],
                qubit_data["EC"],
                qubit_data["EL"],
                Ic_sigma=Ic_sigma,
                averaging=averaging,
                num_points=num_points     
            )
            
            # Update the dictionary for each qubit with new properties
            self.potential_qubits[f"qubit_{i}"].update({
                "qfreq": qfreq,
                "anharm": anharm,
                "t1_eff": t1_eff,
                "t2_eff": t2_eff,
                "anharm std": anharm_std,
                "anharm std_err": anharm_std_err
            })
        
        filtered_pot_qubits = {}
        # Verify qubit params are close to target
        if qfreq_tol != 0 and anharm_tol == 0:
            raise ValueError("Both qfreq_tol and anharm_tol must be set for filtering")
        elif qfreq_tol == 0 and anharm_tol != 0:
            raise ValueError("Both qfreq_tol and anharm_tol must be set for filtering")
        elif qfreq_tol != 0 and anharm_tol != 0:
            j=0
            for i in range(len(self.potential_qubits)):
                qubit_props = self.potential_qubits[f"qubit_{i}"]
                # Check if qfreq is close to target
                if self.qfreqs != None:
                    is_qfreq_close = np.isclose(self.qfreqs, qubit_props["qfreq"], atol=qfreq_tol)
                else:
                    is_qfreq_close = np.isclose(self.qfreq, qubit_props["qfreq"], atol=qfreq_tol)
                if np.any(is_qfreq_close):
                    # Check if anharm is close to target
                    is_anharm_close = np.isclose(self.target_anharms, qubit_props["anharm"], atol=anharm_tol)
                    if np.any(is_anharm_close):
                        filtered_pot_qubits[f"qubit_{j}"] = {
                            "EJ": self.EJ,
                            "EC": qubit_props["EC"],
                            "EL": qubit_props["EL"],
                            "qfreq": qubit_props["qfreq"],
                            "anharm": qubit_props["anharm"],
                            "t1_eff": qubit_props["t1_eff"],
                            "t2_eff": qubit_props["t2_eff"],
                            "anharm std": qubit_props["anharm std"],
                            "anharm std_err": qubit_props["anharm std_err"]
                        }
                        j = j + 1

            #Check not getting duplicate qubits
            qubits_key_to_remove = []
            if ec_el_tol:
                for i in range(len(filtered_pot_qubits)):
                    test_against_qubit = filtered_pot_qubits[f"qubit_{i}"]
                    for j in range(i+1, len(filtered_pot_qubits)):
                        if np.isclose(test_against_qubit["EC"], filtered_pot_qubits[f"qubit_{j}"]["EC"], ec_el_tol) and np.isclose(test_against_qubit["EL"], filtered_pot_qubits[f"qubit_{j}"]["EL"], ec_el_tol):
                            qubits_key_to_remove.append(f"qubit_{j}")  # Store the actual key name

            # Remove duplicates safely by iterating over the keys in reverse order
            for key in sorted(qubits_key_to_remove, reverse=True):
                if key in filtered_pot_qubits:
                    del filtered_pot_qubits[key]
            
            # Changing the keys to new index values
            new_filtered_pot_qubits = {}

            for i, (key, value) in enumerate(filtered_pot_qubits.items()):
                new_key = f"qubit_{i}" 
                new_filtered_pot_qubits[new_key] = value

            self.potential_qubits = new_filtered_pot_qubits
        
        

    def plot_contour_sweep(self, plot:str = "both", EC_range=None, EL_range=None, fill:bool = True, alpha:float = 1, save=False, latex=False):
        
        # Validate the alpha value
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1") 

        valid_plot_options = {"both", "anharm", "qfreq", "none"}
        if plot not in valid_plot_options:
            raise ValueError(f"Invalid plot option: {plot}. Valid options are: {', '.join(valid_plot_options)}")
        
        # Extract data from self.cont_sweep_results
        EJ = self.cont_sweep_results["EJ"]
        EC = self.cont_sweep_results["EC"]
        EL = self.cont_sweep_results["EL"]
        anharms = self.cont_sweep_results["anharms"]
        qfreqs = self.cont_sweep_results["qfreqs"]
        qfreq_levels = self.cont_sweep_results["qfreq_levels"]
        anharm_levels = self.cont_sweep_results["anharm_levels"]
        plt.rcParams.update({'font.size': 16})
        # Enable or disable LaTeX for matplotlib plots
        if latex:
            plt.rcParams['text.usetex'] = True
            xlabel = r"$E_C \mathrm{ (GHz)}$"
            ylabel = r"$E_L \mathrm{ (GHz)}$"
            title = rf"$\mathrm{{Fluxonium\ Qubits;}}\ E_J = {EJ:.1f}\ \mathrm{{GHz}}$"
            qfreq_label = r"$\mathrm{Qubit\ Frequency\ (GHz)}$"
            anharm_label = r"$\mathrm{Anharmonicity}$"

        else:
            plt.rcParams['text.usetex'] = False
            xlabel = r"$E_C$ (GHz)"
            ylabel = r"$E_L$ (GHz)"
            title = f"Fluxonium Qubits; $E_J$ = {EJ:.2f} GHz"
            qfreq_label = "Qubit Frequency (GHz)"
            anharm_label = "Anharmonicity"



        # Create the contour plot for qubit frequency
        if plot == "both" or plot == "qfreq":
            plt.contour(EC, EL, qfreqs, levels=qfreq_levels, colors='black', alpha=0)  # Qubit frequency contours
            plt.clabel(plt.contour(EC, EL, qfreqs, levels=qfreq_levels, colors='black'), inline=True, fontsize=8, fmt=rf"$\omega_q = %.2f \mathrm{{ GHz}}$")
            if fill and plot in {"both", "qfreq"}:
                plt.pcolormesh(EC, EL, qfreqs, shading='auto', cmap='viridis', edgecolor='face', alpha=alpha)
                plt.colorbar(label=qfreq_label)  # Add a color bar for qubit frequencies

        # Add the anharmonicity contours
        if plot == "both" or plot == "anharm":
            plt.contour(EC, EL, anharms, levels=anharm_levels, colors='red', linestyles='dashed', alpha=0)  # Anharmonicity contours
            plt.clabel(plt.contour(EC, EL, anharms, levels=anharm_levels, colors='red', linestyles='dashed'), inline=True, fontsize=8, fmt=rf"$\eta = %.1f$")
            if fill and plot == "anharm":
                plt.pcolormesh(EC, EL, anharms, shading='auto', cmap='viridis', edgecolor='face', alpha=alpha)
                plt.colorbar(label=anharm_label)  # Add a color bar for anharmonicity

        # Plot potential qubits if present
        if plot == "both":
            if len(self.potential_qubits) != 0:
                for i in range(len(self.potential_qubits)):
                    plt.scatter(self.potential_qubits[f"qubit_{i}"]["EC"], self.potential_qubits[f"qubit_{i}"]["EL"], color='orange', s=50, zorder=1, label='Potential Qubits')

        # Set axis labels and plot title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Set EC and EL range if provided
        if EC_range:
            if not isinstance(EC_range, list) or len(EC_range) != 2:
                raise TypeError("EC_range must be a list of length 2")
            plt.xlim(EC_range[0], EC_range[1])

        if EL_range:
            if not isinstance(EL_range, list) or len(EL_range) != 2:
                raise TypeError("EL_range must be a list of length 2")
            plt.xlim(EL_range[0], EL_range[1])

        # Handle saving the figure
        fill_str = "filled" if fill else "no_fill"
        if save:
            save_filename = f"EJ_{EJ}_ECEL_sweep_{plot}_" + fill_str + ".png"
            if isinstance(save, str):
                plt.savefig(save + f"/EJ_{EJ}_ECEL_sweep_{plot}_" + fill_str + ".png", bbox_inches="tight", dpi=300)
            elif save is True:
                plt.savefig(save_filename, bbox_inches="tight", dpi=300)
            else:
                raise TypeError("save must be bool or the save path")

        plt.show()
        plt.rcParams['text.usetex'] = False #turn off latex 

    def save_potential_qubits(self,save_folder:str = None):
        if not hasattr(self,'potential_qubits'):
            raise ValueError("No Potential Qubits to save. Run Sweep first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"potential_qubits_EJ_{self.EJ}_GHz_{timestamp}.json"


        # Save the dictionary to a json file
        if save_folder is None:
            file_path = filename
        else:
            file_path = f"{save_folder}/{filename}"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.potential_qubits, f)

        print(f"Data saved to {file_path}")

                

        


        
