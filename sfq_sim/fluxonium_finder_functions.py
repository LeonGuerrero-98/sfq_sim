import numpy as np
import scqubits as scq
import math
from tqdm import tqdm
import multiprocessing
from shapely.geometry import LineString
import matplotlib.pyplot as plt

def fluxonium_H(EJ,EC,EL,flux = 0.5, cutoff = 20, truncated_dim = 6):
    fluxonium = scq.Fluxonium(
        EJ=EJ,
        EC=EC,
        EL=EL,
        flux=flux,
        cutoff=cutoff,
        truncated_dim=truncated_dim
    )
    return fluxonium

def is_close_to_target(value, target_values, tolerance=1e-5):
    for target in target_values:
        if math.isclose(value, target, abs_tol=tolerance):
            return True
    return False

def qfreq_from_params(EJ,EC,EL):
    fluxonium = scq.Fluxonium(
        EJ=EJ,
        EC=EC,
        EL=EL,
        flux=0.5,
        cutoff=20,
        truncated_dim=4
    )
    qfreq = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
    return qfreq

def anharm_from_params(EJ,EC,EL):
    fluxonium = scq.Fluxonium(
        EJ=EJ,
        EC=EC,
        EL=EL,
        flux=0.5,
        cutoff=20,
        truncated_dim=4
    )
    w10 = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
    w21 = fluxonium.eigenvals()[2] - fluxonium.eigenvals()[1]
    anharm = (w21 -w10)/w10
    return anharm




def EC_EL(EJ, qfreq,EC, EL, freq_tol, anharm_tol, target_anharms):
        results_q = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}
        results_a = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}
        potential_qubits = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}

        fluxonium = fluxonium_H(EJ,EC,EL)
        omega_10 = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
        omega_21 = fluxonium.eigenvals()[2] - fluxonium.eigenvals()[1]
        anharm = (omega_21 - omega_10) / omega_10
        # convert qfreq to list if it is not already a list
        if not isinstance(qfreq, list):
            qfreq = [qfreq]
        # if omega_10 is close to qfreq, add to list
        if is_close_to_target(omega_10, qfreq, tolerance=freq_tol) == True:
            results_q['EC'].append(EC)
            results_q['EL'].append(EL)
            results_q['qfreq'].append(omega_10)
            results_q['anharm'].append(anharm)

            if is_close_to_target(anharm, target_anharms, tolerance=anharm_tol) == True:
                potential_qubits['EC'].append(EC)
                potential_qubits['EL'].append(EL)
                potential_qubits['qfreq'].append(omega_10)
                potential_qubits['anharm'].append(anharm)
        # if anharmonicity is close to target anharmonicity, add to list
        if is_close_to_target(anharm, target_anharms, tolerance=anharm_tol) == True:
            results_a['EC'].append(EC)
            results_a['EL'].append(EL)
            results_a['qfreq'].append(omega_10)
            results_a['anharm'].append(anharm)

        return results_q, results_a, potential_qubits


def mc_sweep_f(params):
    EJ, qfreq,EC, EL, freq_tol,anharm_tol, target_anharms = params
    return EC_EL(EJ, qfreq,EC, EL, freq_tol,anharm_tol, target_anharms = target_anharms)

def EC_EL_sweep(EJ, qfreq,EC_range, EL_range, freq_tol = 0.05,anharm_tol = 0.05, target_anharms = [n+0.5 for n in range(10)], multicore = 0):
        results_q = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}
        results_a = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}
        potential_qubits = {'EC': [], 'EL': [], 'qfreq': [], 'anharm': []}

        if multicore == 0:
            for EC in tqdm(EC_range, desc="EC sweep"):
                for EL in tqdm(EL_range, desc="EL sweep", leave=False):
                    fluxonium = fluxonium_H(EJ,EC,EL)
                    omega_10 = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
                    omega_21 = fluxonium.eigenvals()[2] - fluxonium.eigenvals()[1]
                    anharm = (omega_21 - omega_10) / omega_10
                    # if omega_10 is close to qfreq, add to list
                    if is_close_to_target(omega_10, [qfreq], tolerance=freq_tol) == True:
                        results_q['EC'].append(EC)
                        results_q['EL'].append(EL)
                        results_q['qfreq'].append(omega_10)
                        results_q['anharm'].append(anharm)

                        if is_close_to_target(anharm, target_anharms, tolerance=anharm_tol) == True:
                            potential_qubits['EC'].append(EC)
                            potential_qubits['EL'].append(EL)
                            potential_qubits['qfreq'].append(omega_10)
                            potential_qubits['anharm'].append(anharm)
                    # if anharmonicity is close to target anharmonicity, add to list
                    if is_close_to_target(anharm, target_anharms, tolerance=anharm_tol) == True:
                        results_a['EC'].append(EC)
                        results_a['EL'].append(EL)
                        results_a['qfreq'].append(omega_10)
                        results_a['anharm'].append(anharm)

        else:
            pool = multiprocessing.Pool(processes=multicore)
            params = [(EJ, qfreq, EC, EL, freq_tol, anharm_tol, target_anharms) for EC in EC_range for EL in EL_range]
            results = []
            for result in tqdm(pool.imap(mc_sweep_f, params), total=len(params), desc="Multicore sweep",miniters=100):
                results.append(result)
            pool.close()
            pool.join()
            for result in results:
                results_q['EC'] += result[0]['EC']
                results_q['EL'] += result[0]['EL']
                results_q['qfreq'] += result[0]['qfreq']
                results_q['anharm'] += result[0]['anharm']
                results_a['EC'] += result[1]['EC']
                results_a['EL'] += result[1]['EL']
                results_a['qfreq'] += result[1]['qfreq']
                results_a['anharm'] += result[1]['anharm']
                potential_qubits['EC'] += result[2]['EC']
                potential_qubits['EL'] += result[2]['EL']
                potential_qubits['qfreq'] += result[2]['qfreq']
                potential_qubits['anharm'] += result[2]['anharm']
             

        return results_q, results_a, potential_qubits

def get_paths_from_contours(contours):
    paths = []
    # Loop over each contour collection
    for collection in contours.collections:
        for path in collection.get_paths():
            vertices = path.vertices  # Get the vertices of the path
            paths.append(vertices)
    return paths

def EC_EL_sweep_contour(EJ,EC,EL,qfreq,anharm,plot:str = "both",fill = True,alpha = 0.9,save = False):

    valid_plot_options = {"both", "anharm", "qfreq", "none"}
    if plot not in valid_plot_options:
        raise ValueError(f"Invalid plot option: {plot}. Valid options are: {', '.join(valid_plot_options)}")



    n_steps = len(EC)
    EC, EL = np.meshgrid(EC, EL)

    qfreqs = np.zeros_like(EC)
    anharms = np.zeros_like(EC)

    print("Performing sweep...")
    for i in tqdm(range(n_steps), desc="Sweep Progress", leave=True):
        for j in range(n_steps):
            qfreqs[i, j] = qfreq_from_params(EJ, EC[i, j], EL[i, j])
            anharms[i, j] = anharm_from_params(EJ, EC[i, j], EL[i, j])
    print("Sweep Complete")

    if type(qfreq) == list:
        qfreq_levels = qfreq
    elif type(qfreq) == float:
        qfreq_levels = [qfreq]
    else:
        raise TypeError("qfreq must be a list of a float")

    anharm_levels = anharm

    # Create the contour plot for qubit frequency and anharmonicity
    qfreq_contours = plt.contour(EC, EL, qfreqs, levels=qfreq_levels, colors='black')
    anharm_contours = plt.contour(EC, EL, anharms, levels=anharm_levels, colors='red', linestyles='dashed')

    qfreq_paths = get_paths_from_contours(qfreq_contours)
    anharm_paths = get_paths_from_contours(anharm_contours)

    # Find intersections between contour paths
    intersections = []
    intersection_points = []  # To store EC and EL coordinates of intersections

    for qfreq_path in qfreq_paths:
        line1 = LineString(qfreq_path)
        for anharm_path in anharm_paths:
            line2 = LineString(anharm_path)
            intersection = line1.intersection(line2)
            if not intersection.is_empty:
                intersections.append(intersection)

                # Extract the EC and EL coordinates of the intersection
                if intersection.geom_type == 'Point':
                    intersection_points.append((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        intersection_points.append((point.x, point.y))

    plt.close()

    if plot != "none":

        # Enable LaTeX in all matplotlib plots
        #plt.rcParams['text.usetex'] = True
        # Create the contour plot for qubit frequency
        if plot == "both" or plot == "qfreq":
            plt.contour(EC, EL, qfreqs, levels=qfreq_levels, colors='black', alpha =0)  # Qubit frequency contours
            plt.clabel(plt.contour(EC, EL, qfreqs, levels=qfreq_levels, colors='black'), inline=True, fontsize=8, fmt=rf"$\omega_q$ = %.2f GHz")
            if fill == True and plot in {"both", "qfreq"}:
                #plt.contourf(EC, EL, qfreqs, levels=100, cmap='viridis', alpha=alpha)  # Filled qubit frequency contours
                plt.pcolormesh(EC, EL, qfreqs, shading='auto', cmap='viridis', edgecolor='face',alpha=alpha)
                # Add a color bar and labels
                plt.colorbar(label="Qubit Frequency (GHz)")  # Add a color bar for qubit frequencies
        # Add the anharmonicity contours
        if plot == "both" or plot == "anharm":
            plt.contour(EC, EL, anharms, levels=anharm_levels, colors='red', linestyles='dashed',alpha = 0)  # Anharmonicity contours
            plt.clabel(plt.contour(EC, EL, anharms, levels=anharm_levels, colors='red', linestyles='dashed'), inline=True, fontsize=8, fmt=rf"$\eta$ = %.1f")
            if fill == True and plot == "anharm":
                #plt.contourf(EC, EL, anharms, levels = 100, cmap='viridis', linestyles = "dashed")  # Filled qubit frequency contours
                # Add a color bar and labels
                plt.pcolormesh(EC, EL, anharms, shading='auto', cmap='viridis', edgecolor='face',alpha=alpha)  # Match edge color to face color
                plt.colorbar(label="Anharmonicity")  # Add a color bar for qubit frequencies
        
        # Plot the intersection points as scatter points
        if plot == "both":
            for point in intersection_points:
                plt.scatter(point[0], point[1], color='orange', s=50, zorder=1, label='Intersection')

        plt.xlabel(r"$E_C$ (GHz)")
        plt.ylabel(r"$E_L$ (GHz)")
        plt.title(rf"Fluxonium Qubits; $E_J$ = {EJ} GHz")

        if fill == True:
            fill_str = "filled"
        elif fill == False:
            fill_str = "no_fill"
        if save != False:
            if save == True:
                plt.savefig(f"EJ_{EJ}_ECEL_sweep_{plot}_" + fill_str+ ".png",bbox_inches = "tight",dpi = 300)
            if type(save) == str:
                plt.savefig(save + f"\EJ_{EJ}_ECEL_sweep_{plot}_" + fill_str+ ".png",bbox_inches = "tight",dpi = 300)
            else:
                raise TypeError("save must be bool or the save path")

        plt.show()
        # Initialize potential_qubits as an empty dictionary
        potential_qubits = {}

        # Loop through the intersection points and store EJ, EC, and EL values in a dictionary
        for i, point in enumerate(intersection_points):
            if plot:
                print(f"Intersection {i+1}: EC = {point[0]}, EL = {point[1]}")
            
            # Assign EJ, EC, and EL values directly to the potential_qubits dictionary with a unique key for each qubit
            potential_qubits[f"qubit_{i}"] = {
                "EJ": EJ,
                "EC": point[0],
                "EL": point[1]
            }

    print(str(len(potential_qubits)) + " potential qubits found.")  
    return potential_qubits

def anharm_variability(EJ,EC,EL,Ic_sigma = 0.05, num_points = 1000):
    rand_EJ = np.random.normal(loc = EJ,scale = Ic_sigma*EJ,size = num_points)
    rand_anharms = []
    for ej in rand_EJ:
        fluxonium = scq.Fluxonium(
        EJ=ej,
        EC=EC,
        EL=EL,
        flux=0.5,
        cutoff=20,
        truncated_dim=4
        )
        qfreq = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
        w21 = fluxonium.eigenvals()[2] - fluxonium.eigenvals()[1]
        rand_anharms.append((w21 -qfreq)/qfreq)
    rand_anharms = np.array(rand_anharms)
    anharm_std = np.std(rand_anharms,ddof=1) #ddof = 1 gives sample stdev instead of population stdev
    return anharm_std

def fluxonium_properties_from_params(EJ,EC,EL,Ic_sigma = 0.05):
    
    fluxonium = scq.Fluxonium(
        EJ=EJ,
        EC=EC,
        EL=EL,
        flux=0.5,
        cutoff=20,
        truncated_dim=4
    )
    qfreq = fluxonium.eigenvals()[1] - fluxonium.eigenvals()[0]
    w21 = fluxonium.eigenvals()[2] - fluxonium.eigenvals()[1]
    anharm = (w21 -qfreq)/qfreq
    t1_eff = fluxonium.t1_effective()
    t2_eff = fluxonium.t2_effective()
    anharm_std = anharm_variability(EJ,EC,EL,Ic_sigma,1000)#how much the anharmonicity changes depering on the variability of JJs
    return qfreq, anharm,t1_eff, t2_eff, anharm_std


    

        

    




