import pytest
import numpy as np

from sfq_sim.pulse_functions import *

import pytest
import numpy as np
from qutip import Qobj,basis

def test_sfq_qutrit_Ry_basic_execution():
    # Basic test to check if the function runs without errors
    
    n = 1
    anharm = 0.95
    omega_10 = 1e9
    initial_state = basis(3,0)  # Example 2D input
    theta = np.pi / 2
    pulse_width = 2e-12
    t_delay = 1e-11
    n_steps = 100

    result = sfq_qutrit_Ry(n, anharm, omega_10, initial_state, theta, pulse_width, t_delay, n_steps)
    
    # Assert that results is a dictionary and has expected keys
    assert isinstance(result, dict), "Result should be a dictionary."
    expected_keys = ['fids', 'P2', 'P1', 'P0', 'sx', 'sy', 'sz', 'psi', 't', 'pulse']
    assert all(key in result for key in expected_keys), "Missing keys in the result dictionary."

def test_initial_state_validation():
    # Check that incorrect initial state shapes raise the correct error
    
    n = 1
    anharm = 0.95
    omega_10 = 1e9
    theta = np.pi / 2
    
    # Case: Invalid shape (not 2D or 3D)
    initial_state_invalid = Qobj(np.array([[1, 0], [0, 1]]))  # 2x2 matrix, invalid for this function
    with pytest.raises(ValueError, match="Initial state must be a 2D or 3D Qobj"):
        sfq_qutrit_Ry(n, anharm, omega_10, initial_state_invalid, theta)

def test_fidelity_output():
    # Check that fidelity output is as expected
    #from module_containing_function import sfq_qutrit_Ry
    
    n = 1
    anharm = 0.95
    omega_10 = 1e9
    initial_state = basis(3,0)
    theta = np.pi / 2

    result = sfq_qutrit_Ry(n, anharm, omega_10, initial_state, theta, n_steps = 100)
    
    # Assert that fidelity (fids) is within expected bounds (0 <= fids <= 1)
    assert 0 <= result['fids'].all() <= 1, "Fidelity should be between 0 and 1."
