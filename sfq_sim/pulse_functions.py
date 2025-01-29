#SFQ pulse functions
import numpy as np
from qutip import *
from scipy import interpolate as sp
from .constants import *
from tqdm import tqdm



def normal_dist(x,x0,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*sigma**2))

def normal_sfq(n,omega_10,pulse_width = 2e-12,t_delay = 0,n_steps = 3e5):
    '''
    normal_sfq function.

    Generates sfq pulses with normal distribution shape.

    Parameters:
        n (int): number of pulses.
        omega_10 (float): qubit frequency
        pulse_width (float): width of each pulse.

        n_steps (int, float): number of total steps in time array. Default = 3e5
        t_delay (float): delay time before first pulse (required so that the full first pulse is captured, not just the second half). Default = 1e-11.

    Returns:
        t (array): time array.
        pulse (array): signal pulse array.
    '''
    T_sep = 2*np.pi/omega_10
    n_steps = int(n_steps)
    t = np.linspace(-pulse_width*5,t_delay + n*T_sep,n_steps)
    pulse = Phi_0*normal_dist(t,t_delay,pulse_width)
    for i in range(n-1):
        pulse = np.add(pulse,Phi_0*normal_dist(t,-5*pulse_width + t_delay + (i+1)*T_sep,pulse_width))
    return t,pulse

def jitter_sfq(n,omega_10,noise_sigma,pulse_width = 2e-12,t_delay = 1e-11,n_steps = 3e5):
    '''
    jitter_sfq function.

    Generates sfq pulses with normal distribution shape and timing jitter due to precision limits on equipment.

    Parameters:
        n (int): number of pulses.
        omega_10 (float): qubit frequency.
        t_delay (float): delay time before first pulse (required so that the full first pulse is captured, not just the second half.).
        pulse_width (float): width of each pulse.
        n_steps (int, float): number of steps in time array.
        noise_sigma (float): standard deviation of noise.

    Returns:
        t (array): time array.
        pulse (array): signal pulse array.

    '''
    T_sep = 2*np.pi/omega_10
    n_steps = int(n_steps)
    t = np.linspace(0,t_delay + pulse_width/2 + n*T_sep,n_steps)
    noise = np.random.normal(0,noise_sigma,1)
    pulse = Phi_0*normal_dist(t,t_delay + pulse_width/2,pulse_width)
    for i in range(n-1):
        noise = np.random.normal(0,noise_sigma,1)
        pulse = np.add(pulse,Phi_0*normal_dist(t,t_delay + (pulse_width/2) + (i+1)*T_sep + noise,pulse_width))
    return t,pulse

def jitter_sfq_int(n,omega_10,noise_sigma,pulse_width = 2e-12,t_delay = 1e-11,n_steps = 3e5):
    '''
    jitter_sfq_int function.

    Generates sfq pulses with normal distribution shape and timing jitter due to precision limits on equipment.
    SFQ pulses generated internally (from a clock ring), which leads to incoherent accumulation of timing errors, leading to a root k degradation of timing jitter for the kth pulse.

    Parameters:
        n (int): number of pulses.
        t_delay (float): delay time before first pulse (required so that the full first pulse is captured, not just the second half.).
        pulse_width (float): width of each pulse.
        T_sep (float): time between pulses.
        n_steps (int, float): number of steps in time array.
        noise_sigma (float): standard deviation of noise.

    Returns:
        t (array): time array.
        pulse (array): signal pulse array.

    '''
    T_sep = 2*np.pi/omega_10
    n_steps = int(n_steps)
    t = np.linspace(0,t_delay + pulse_width/2 + n*T_sep,n_steps)
    noise = np.random.normal(0,noise_sigma,1)
    pulse = Phi_0*normal_dist(t,t_delay + pulse_width/2,pulse_width)
    for i in range(n-1):
        noise = np.random.normal(0,np.sqrt(i+1)*noise_sigma,1)
        pulse = np.add(pulse,Phi_0*normal_dist(t,t_delay + (pulse_width/2) + (i+1)*T_sep + noise,pulse_width))
    return t,pulse

def Ry_3d(theta):
    return Qobj(np.array([[np.cos(theta/2), -np.sin(theta/2), 0],
                          [np.sin(theta/2), np.cos(theta/2), 0],
                          [0, 0, 1]]))

def Rx_3d(theta):
    return Qobj(np.array([[np.cos(theta/2), -1j*np.sin(theta/2), 0],
                          [-1j*np.sin(theta/2), np.cos(theta/2), 0],
                          [0, 0, 1]]))


def sfq_qutrit_Ry(n, anharm, omega_10,initial_state,theta,pulse_width = 2e-12,t_delay = 1e-11, n_steps = 3e5, progress = True, int_jit = 0, store_final_only = False):
    #calculate omega20 and delta theta based on input values
    omega_20 = anharm*omega_10 + (2 * omega_10)
    delta_theta = theta / n 
    if int_jit != 0:
        t,pulse = jitter_sfq_int(n,omega_10,int_jit,pulse_width,t_delay,n_steps)
    else:
        t,pulse = normal_sfq(n,omega_10,pulse_width,t_delay,n_steps) # Generating pulse signal
    pulse_func = sp.interp1d(t,pulse,fill_value = "extrapolate") # Interpolating pulse function

    #if initial state is 2d convert to 3d
    if initial_state.shape == (2,1):
        psi0 = Qobj(np.array([psi0.full()[0], psi0.full()[1], [0]]))
    elif initial_state.shape == (3,1):
        psi0 = initial_state
    else:
        raise ValueError("Initial state must be a 2D or 3D Qobj")
    
    # target final state is Ry(pi/2) |psi0>
    target_state = Ry_3d(theta)*psi0
    target_state_op = target_state*target_state.dag()

    state2 = (basis(3,2)) # Define |2> state
    state2_op = state2*state2.dag() # Define |2><2| operator

    state1 = (basis(3,1)) # Define |1> state
    state1_op = state1*state1.dag() # Define |1><1| operator

    state0 = (basis(3,0)) # Define |0> state
    state0_op = state0*state0.dag() # Define |0><0| operator

    sigmax3d = Qobj(np.array([np.append(sigmax().full()[0], [0]),
                            np.append(sigmax().full()[1], [0]),
                            [0, 0, 1]]))
    sigmay3d = Qobj(np.array([np.append(sigmay().full()[0], [0]),
                            np.append(sigmay().full()[1], [0]),
                            [0, 0, 1]]))
    sigmaz3d = Qobj(np.array([np.append(sigmaz().full()[0], [0]),
                            np.append(sigmaz().full()[1], [0]),
                            [0, 0, 1]]))

    b = delta_theta/(2*Phi_0) # Matrix for free Hamiltonian
    H_sfq = 1j*b*(create(3) - destroy(3)) # SFQ Hamiltonian
    free_matrix = [0,0,0],[0,omega_10,0],[0,0,omega_20] # Free Hamiltonian Matrix
    H_free = Qobj(free_matrix) # Free Hamiltonian converted to Qobj

    def oper(t):
        return H_free + pulse_func(t)*H_sfq # Full time-dependent Hamiltoian. SFQ Element multiplied by pulse function

    H_t = QobjEvo(oper) # Convert Hamiltonian to QobjEvo object for time-dependent evolution

     # Solve for coefficients of each level. Max step must be < 1/2 pulse width, otherwise will lead to incorrect solution
    if store_final_only == False:
        result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op, sigmax3d,sigmay3d,sigmaz3d], options={"max_step": pulse_width/3, "progress_bar": progress, "store_states": True})
        fids = result.expect[0] 
        P2 = result.expect[1]
        P1 = result.expect[2]
        P0 = result.expect[3]
        sx = result.expect[4]
        sy = result.expect[5]
        sz = result.expect[6]
        psi = result.states
    else:
        result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op, sigmax3d,sigmay3d,sigmaz3d], options={"max_step": pulse_width/3, "progress_bar": progress, "store_final_state": True})
        fids = result.expect[0][-1] 
        P2 = result.expect[1][-1]
        P1 = result.expect[2][-1]
        P0 = result.expect[3][-1]
        sx = result.expect[4][-1]
        sy = result.expect[5][-1]
        sz = result.expect[6][-1]
        psi = result.states
        t = None
        pulse = None
    #create a dictionary to store the results
    results = {"fids": fids, "P2": P2, "P1": P1, "P0": P0, "sx": sx, "sy": sy, "sz": sz, "psi": psi, "t": t, "pulse": pulse}
    #in this case fidelities is P1, for consistancy output 4 arrays with first one being fids
    return results


def sfq_qutrit_Ry_anharm_sweep(n, anharms, omega_10,initial_state,theta,pulse_width = 2e-12 ,t_delay = 1e-11, n_steps = 3e5, progress = False, int_jit = 0):

    #calculate delta theta based on input values
    delta_theta = theta / n 

    fids=[] # Store probabilities of |+>
    P2 = [] # and probabilities of |2>
    P1 = []
    P0 = []
    psi_f = []

    if int_jit != 0:
        t,pulse = jitter_sfq_int(n,omega_10,int_jit,pulse_width,t_delay,n_steps)
    else:
        t,pulse = normal_sfq(n,omega_10,pulse_width,t_delay,n_steps) # Generating pulse signal
    pulse_func = sp.interp1d(t,pulse,fill_value = "extrapolate") # Interpolating pulse function

    #if initial state is 2d convert to 3d
    if initial_state.shape == (2,1):
        psi0 = Qobj(np.array([psi0.full()[0], psi0.full()[1], [0]]))
    elif initial_state.shape == (3,1):
        psi0 = initial_state
    else:
        raise ValueError("Initial state must be a 2D or 3D Qobj")
    
    # target final state is Ry(pi/2) |psi0>
    target_state = Ry_3d(theta)*psi0
    target_state_op = target_state*target_state.dag()

    state2 = (basis(3,2)) # Define |2> state
    state2_op = state2*state2.dag() # Define |2><2| operator

    state1 = (basis(3,1)) # Define |1> state
    state1_op = state1*state1.dag() # Define |1><1| operator

    state0 = (basis(3,0)) # Define |0> state
    state0_op = state0*state0.dag() # Define |0><0| operator
    if progress == True:
        for anharm in tqdm(anharms, desc="Anharmonicity Sweep Progress"):
            omega_20 = anharm*omega_10 + (2 * omega_10)
            b = delta_theta/(2*Phi_0) # Matrix for free Hamiltonian
            H_sfq = 1j*b*(create(3) - destroy(3)) # SFQ Hamiltonian
            free_matrix = [0,0,0],[0,omega_10,0],[0,0,omega_20] # Free Hamiltonian Matrix
            H_free = Qobj(free_matrix) # Free Hamiltonian converted to Qobj

            def oper(t):
                return H_free + pulse_func(t)*H_sfq # Full time-dependent Hamiltoian. SFQ Element multiplied by pulse function

            H_t = QobjEvo(oper) # Convert Hamiltonian to QobjEvo object for time-dependent evolution

            result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op], options={"max_step": pulse_width/3, "progress_bar": False, "store_final_state": True}) # Solve for coefficients of each level. Max step must be < 1/2 pulse width, otherwise will lead to incorrect solution
            #only store final result of evolution
            fids.append(result.expect[0][-1]) 
            P2.append(result.expect[1][-1]) 
            P1.append(result.expect[2][-1])
            P0.append(result.expect[3][-1])
            psi_f.append(result.states)
    else:
        for anharm in anharms:
            omega_20 = anharm*omega_10 + (2 * omega_10)
            b = delta_theta/(2*Phi_0)
            H_sfq = 1j*b*(create(3) - destroy(3)) # SFQ Hamiltonian
            free_matrix = [0,0,0],[0,omega_10,0],[0,0,omega_20] # Free Hamiltonian Matrix
            H_free = Qobj(free_matrix) # Free Hamiltonian converted to Qobj

            def oper(t):
                return H_free + pulse_func(t)*H_sfq # Full time-dependent Hamiltoian. SFQ Element multiplied by pulse function

            H_t = QobjEvo(oper) # Convert Hamiltonian to QobjEvo object for time-dependent evolution

            result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op], options={"max_step": pulse_width/3, "progress_bar": False, "store_final_state": True}) # Solve for coefficients of each level. Max step must be < 1/2 pulse width, otherwise will lead to incorrect solution
            #only store final result of evolution
            fids.append(result.expect[0][-1]) 
            P2.append(result.expect[1][-1]) 
            P1.append(result.expect[2][-1])
            P0.append(result.expect[3][-1])
            psi_f.append(result.states)

    #create dictionary to store results
    results = {"anharm": anharms,"fids": fids, "P2": P2, "P1": P1, "P0": P0, "psi_f": psi_f}

    return results

def sfq_qutrit_Ry_jitter_sweep(n, anharm, omega_10,initial_state,theta,jitter_sigmas,averaging = 1,pulse_width = 2e-12,t_delay = 1e-11, n_steps = 3e5, progress = True):
    
    fids_mean,P2_mean,P1_mean,P0_mean = [],[],[],[]
    fids_err,P2_err,P1_err,P0_err = [],[],[],[]  


    #if initial state is 2d convert to 3d
    if initial_state.shape == (2,1):
        psi0 = Qobj(np.array([psi0.full()[0], psi0.full()[1], [0]]))
    elif initial_state.shape == (3,1):
        psi0 = initial_state
    else:
        raise ValueError("Initial state must be a 2D or 3D Qobj")

    for sigma in tqdm(jitter_sigmas, desc="Jitter Sigma Sweep Progress",leave = progress):
        fids,P2,P1,P0 = [],[],[],[]
        for i in range(averaging):
            results = sfq_qutrit_Ry(n,anharm,omega_10,initial_state,theta,pulse_width,t_delay,n_steps,progress = False, int_jit=sigma, store_final_only=True)
            fids.append(results["fids"])
            P2.append(results["P2"])
            P1.append(results["P1"])
            P0.append(results["P0"])

        fids_mean.append(np.mean(fids))
        P2_mean.append(np.mean(P2))
        P1_mean.append(np.mean(P1))
        P0_mean.append(np.mean(P0))
        fids_err.append(np.std(P1)/np.sqrt(averaging))
        P2_err.append(np.std(P2)/np.sqrt(averaging))
        P1_err.append(np.std(P1)/np.sqrt(averaging))
        P0_err.append(np.std(P0)/np.sqrt(averaging))

        result = {"jitter_sigmas": jitter_sigmas, "fids_mean": fids_mean, "P2_mean": P2_mean, "P1_mean": P1_mean, "P0_mean": P0_mean, "fids_err": fids_err, "P2_err": P2_err, "P1_err": P1_err, "P0_err": P0_err}
    return result

def P2_j(Theta, n, delta, j ,eta):
    # Define complex exponential parts
    exp1_num = np.exp(1j * n * (2 * np.pi * eta) + delta / 2)
    exp1_den = np.exp(1j * (2 * np.pi* eta) + delta / 2)
    
    exp2_num = np.exp(1j * n * (2 * np.pi* eta) - delta / 2)
    exp2_den = np.exp(1j * (2 * np.pi* eta) - delta / 2)
    
    # Compute the first fraction
    term1 = (1 - exp1_num) / (1 - exp1_den)
    
    # Compute the second fraction with (-1)^j factor
    term2 = (-1)**j * (1 - exp2_num) / (1 - exp2_den)
    
    # Combine both terms and compute absolute value squared
    result = np.abs(term1 - term2)**2
    
    # Final formula for P2_j
    P2_j_value = (Theta**2 / (8 * n**2)) * result
    
    return P2_j_value

def compute_fid_Ry_anharm(params):
    n, initial_state, theta ,pulse_width, i, qfreq, t_delay, steps = params
    return sfq_qutrit_Ry(n, i, qfreq,initial_state,theta, pulse_width, t_delay, steps, False, int_jit=0, store_final_only=True)

def compute_fid_Ry_jitter(params):
    fids_mean,P2_mean,P1_mean,P0_mean = [],[],[],[]
    fids_err,P2_err,P1_err,P0_err = [],[],[],[]
    n,anharm,omega_10,initial_state,theta,pulse_width,t_delay,n_steps,progress, sigma, store_final_only, averaging = params
    fids,P2,P1,P0 = [],[],[],[]
    for i in range(averaging):
        
        results = sfq_qutrit_Ry(n,anharm,omega_10,initial_state,theta,pulse_width,t_delay,n_steps,progress, int_jit=sigma, store_final_only=store_final_only)
        fids.append(results["fids"])
        P2.append(results["P2"])
        P1.append(results["P1"])
        P0.append(results["P0"])


    fids_mean.append(np.mean(fids))
    P2_mean.append(np.mean(P2))
    P1_mean.append(np.mean(P1))
    P0_mean.append(np.mean(P0))


    fids_err.append(np.std(fids)/np.sqrt(averaging))
    P2_err.append(np.std(P2)/np.sqrt(averaging))
    P1_err.append(np.std(P1)/np.sqrt(averaging))
    P0_err.append(np.std(P0)/np.sqrt(averaging))

    #create dict with mean and error values
    result = {"fids_mean": fids_mean, "P2_mean": P2_mean, "P1_mean": P1_mean, "P0_mean": P0_mean, "fids_err": fids_err, "P2_err": P2_err, "P1_err": P1_err, "P0_err": P0_err}
    return result

def sfq_qutrit_RF(n, anharm, omega_10,initial_state,theta,gate,pulse_width = 2e-12,t_delay = 0, n_steps = 3e5, progress = True, int_jit = 0, store_final_only = False):
    #calculate omega20 and delta theta based on input values
    delta_theta = theta / n 
    if int_jit != 0:
        t,pulse = jitter_sfq_int(n,omega_10,int_jit,pulse_width,t_delay,n_steps)
    else:
        t,pulse = normal_sfq(n,omega_10,pulse_width,t_delay,n_steps) # Generating pulse signal
    pulse_func = sp.interp1d(t,pulse,fill_value = "extrapolate") # Interpolating pulse function

    def H_sfq_rot_frame(t):
        Sy = 1j*(create(3) - destroy(3))
        Sx = -1*(create(3) + destroy(3))

        H = (np.cos(omega_10*t)*Sy + np.sin(omega_10*t)*Sx)*pulse_func(t)
        return H

    #if initial state is 2d convert to 3d
    if initial_state.shape == (2,1):
        psi0 = Qobj(np.array([psi0.full()[0], psi0.full()[1], [0]]))
    elif initial_state.shape == (3,1):
        psi0 = initial_state
    else:
        raise ValueError("Initial state must be a 2D or 3D Qobj")
    
    # target final state is Ry(pi/2) |psi0>
    if gate == "X":
        target_state = Rx_3d(theta)*psi0
    if gate == "Y":
        target_state = Ry_3d(theta)*psi0
    target_state_op = target_state*target_state.dag()

    state2 = (basis(3,2)) # Define |2> state
    state2_op = state2*state2.dag() # Define |2><2| operator

    state1 = (basis(3,1)) # Define |1> state
    state1_op = state1*state1.dag() # Define |1><1| operator

    state0 = (basis(3,0)) # Define |0> state
    state0_op = state0*state0.dag() # Define |0><0| operator

    sigmax3d = Qobj(np.array([np.append(sigmax().full()[0], [0]),
                            np.append(sigmax().full()[1], [0]),
                            [0, 0, 1]]))
    sigmay3d = Qobj(np.array([np.append(sigmay().full()[0], [0]),
                            np.append(sigmay().full()[1], [0]),
                            [0, 0, 1]]))
    sigmaz3d = Qobj(np.array([np.append(sigmaz().full()[0], [0]),
                            np.append(sigmaz().full()[1], [0]),
                            [0, 0, 1]]))

    b = delta_theta/(2*Phi_0) # Matrix for free Hamiltonian
    H0 = Qobj(np.array([[0,0,0],[0,0,0],[0,0,anharm*omega_10]]))

    def oper(t):
        return H0 + b*H_sfq_rot_frame(t) # Full time-dependent Hamiltoian. SFQ Element multiplied by pulse function

    H_t = QobjEvo(oper) # Convert Hamiltonian to QobjEvo object for time-dependent evolution

     # Solve for coefficients of each level. Max step must be < 1/2 pulse width, otherwise will lead to incorrect solution
    if store_final_only == False:
        result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op, sigmax3d,sigmay3d,sigmaz3d], options={"max_step": pulse_width/3, "progress_bar": progress, "store_states": True})
        fids = result.expect[0] 
        P2 = result.expect[1]
        P1 = result.expect[2]
        P0 = result.expect[3]
        sx = result.expect[4]
        sy = result.expect[5]
        sz = result.expect[6]
        psi = result.states
    else:
        result = sesolve(H_t, psi0, t, e_ops=[target_state_op, state2_op, state1_op, state0_op, sigmax3d,sigmay3d,sigmaz3d], options={"max_step": pulse_width/3, "progress_bar": progress, "store_final_state": True})
        fids = result.expect[0][-1] 
        P2 = result.expect[1][-1]
        P1 = result.expect[2][-1]
        P0 = result.expect[3][-1]
        sx = result.expect[4][-1]
        sy = result.expect[5][-1]
        sz = result.expect[6][-1]
        psi = result.states
        t = None
        pulse = None
    #create a dictionary to store the results
    results = {"fids": fids, "P2": P2, "P1": P1, "P0": P0, "sx": sx, "sy": sy, "sz": sz, "psi": psi, "t": t, "pulse": pulse}
    #in this case fidelities is P1, for consistancy output 4 arrays with first one being fids
    return results