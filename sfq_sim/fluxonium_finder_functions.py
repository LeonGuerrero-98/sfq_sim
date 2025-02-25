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

def mc_cont_sweep_f(task):
    EJ, EC_val, EL_val = task
    return (qfreq_from_params(EJ, EC_val, EL_val), anharm_from_params(EJ, EC_val, EL_val))

def EC_EL_sweep_contour(EJ,EC,EL,qfreq,anharm, multicore:int = 0, verbose:bool = True):
    
    #Verification
    if type(qfreq) == list:
        qfreq_levels = qfreq
    elif type(qfreq) == float:
        qfreq_levels = [qfreq]
    else:
        raise TypeError("qfreq must be a list of a float")
    if type(anharm) == list:
        anharm_levels = anharm
    elif type(qfreq) == float:
        anharm_levels = [anharm]
    else:
        raise TypeError("anharm must be a list of a float")
    


    n_steps = len(EC)
    EC, EL = np.meshgrid(EC, EL)

    qfreqs = np.zeros_like(EC)
    anharms = np.zeros_like(EC)

    if multicore == 0:
        if verbose:
            print("Performing sweep...")
        for i in tqdm(range(n_steps), desc="Sweep Progress", leave=True,disable = not verbose):
            for j in range(n_steps):
                qfreqs[i, j] = qfreq_from_params(EJ, EC[i, j], EL[i, j])
                anharms[i, j] = anharm_from_params(EJ, EC[i, j], EL[i, j])
        if verbose:
            print("Sweep Complete")

    else:
        if verbose:
            print(f"Performing sweep using {multicore} cores...")

        # Create a list of indices to compute over
        tasks = [(EJ, EC[i, j], EL[i, j]) for i in range(n_steps) for j in range(n_steps)]

        # Initialize progress bar
        with tqdm(total=len(tasks), desc="Sweep Progress", disable= not verbose) as pbar:
            with multiprocessing.Pool(processes=multicore) as pool:  
                # Use imap_unordered to get results as they complete
                for idx, result in enumerate(pool.imap(mc_cont_sweep_f, tasks)):
                    qf, ah = result
                    i, j = divmod(idx, n_steps)
                    qfreqs[i, j] = qf
                    anharms[i, j] = ah
                    if idx % 100 == 0:
                        pbar.update(100)
        if verbose:
            print("Sweep Complete")



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

    potential_qubits = {}

    # Loop through the intersection points and store EJ, EC, and EL values in a dictionary
    for i, point in enumerate(intersection_points):
        if verbose:
            print(f"Intersection {i+1}: EC = {point[0]}, EL = {point[1]}")
        
        # Assign EJ, EC, and EL values directly to the potential_qubits dictionary with a unique key for each qubit
        potential_qubits[f"qubit_{i}"] = {
            "EJ": EJ,
            "EC": point[0],
            "EL": point[1]
        }

    sweep_results = {"EJ":EJ,"EC":EC,"EL":EL,"anharms":anharms,"qfreqs":qfreqs,"anharm_levels":anharm_levels,"qfreq_levels":qfreq_levels}

    return potential_qubits, sweep_results


    # Initialize potential_qubits as an empty dictionary


def anharm_variability(EJ,EC,EL,Ic_sigma = 0.05, num_points = 1000,array_var = False,array_junctions = 100):
    rand_EJ = np.random.normal(loc = EJ,scale = Ic_sigma*EJ,size = num_points)
    if array_var == True:
        rand_EL = np.random.normal(loc = EJ,scale = (Ic_sigma*EJ)/np.sqrt(array_junctions),size = num_points)
    rand_anharms = []
    for i in range(len(rand_EJ)):
        if array_var == True:
            EL = rand_EL[i]
        fluxonium = scq.Fluxonium(
        EJ=rand_EJ[i],
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

def fluxonium_properties_from_params(EJ,EC,EL,Ic_sigma = 0.05,averaging = 3,num_points = 250):
    
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
    anharm_std_list = []
    for i in range(averaging):
        anharm_std_list.append(anharm_variability(EJ,EC,EL,Ic_sigma,num_points))
    anharm_std = np.mean(anharm_std_list)
    anharm_std_err = np.std(anharm_std_list)/np.sqrt(averaging)
    return qfreq, anharm,t1_eff, t2_eff, anharm_std, anharm_std_err


    

        

    




