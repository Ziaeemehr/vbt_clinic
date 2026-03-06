"""
Test script to verify the updated run_sweep function.
"""
import os
import numpy as np
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp

from src.utils import *
from src.bold_delay_cpu import run_sweep, check_input_parameters
from src import gast_model as gm

jax.config.update("jax_enable_x64", True)

print("=" * 80)
print("Testing Updated bold_delay_cpu.py Functions")
print("=" * 80)

# Setup
seed = 42
np.random.seed(seed)

PID_DIR = "/home/ziaee/Desktop/workstation/vbt_clinical/package/dataset"

# Load connectivity data
data_conn = load_connectivity_data(PID_DIR)
Ceids, Leids, Seids, Rd1, Rd2, Rsero, region_names = data_conn

nn = len(region_names)
n_svar = 10

dt = 0.1
T = 10 * 1000  # 10 seconds
tr = 500.0
num_skip = 10
t_cut = 0.0
n_devices = 4

v_c = 3.9
idelays = (Leids[Ceids != 0.0] / v_c / dt).astype(np.uint32)
num_time = int(T / dt)
Ja = np.load(join(PID_DIR, "Ja.npy"))

# Base parameters
base_theta = gm.sigm_d1d2sero_default_theta._replace(
    I=46.5,
    Ja=Ja,
    Jsa=Ja,
    Jsg=13,
    Jg=0,
    Jdopa=100000 * 0.01,
    Rd1=Rd1,
    Rd2=Rd2,
    Rs=Rsero,
    Sd1=-10.0,
    Sd2=-10.0,
    Ss=-40.0,
    Zd1=0.5,
    Zd2=1.0,
    Zs=0.25,
    we=0.27,
    wi=1.0,
    wd=1,
    ws=0.01,
    sigma_V=0.15,
    sigma_u=0.015,
)

init_state = jnp.array([0.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

setup = {
    "Seids": Seids,
    "idelays": idelays,
    "horizon": 650,
    "num_svar": n_svar,
    "dt": dt,
    "num_skip": num_skip,
    "num_time": num_time,
    "bold_decimate": int(tr / (dt * num_skip)),
    "t_cut": t_cut,
    "init_state": init_state,
}

print("\n1. Testing check_input_parameters")
print("-" * 80)
try:
    check_input_parameters(base_theta, nn)
    print("✓ check_input_parameters passed for base_theta")
except Exception as e:
    print(f"✗ check_input_parameters failed: {e}")

print("\n2. Testing run_sweep with parameter sweep (3 simulations)")
print("-" * 80)
try:
    # Create 3 parameter sets with different we values
    theta_list = [
        base_theta._replace(we=we_val)
        for we_val in [0.2, 0.2]
    ]
    
    bolds, ts = run_sweep(
        theta_list=theta_list,
        setup=setup,
        seed=seed,
        n_devices=4,
        verbose=True,
        same_noise=False
    )
    print(f"✓ run_sweep successful")
    print(f"  BOLD shape: {bolds.shape}")
    print(f"  Time shape: {ts.shape}")
    print(f"  Results differ: {not jnp.allclose(bolds[0], bolds[1])}")
except Exception as e:
    print(f"✗ run_sweep failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing run_sweep with same parameters and same_noise=True")
print("-" * 80)
try:
    # Create 10 parameter sets
    theta_list = [
        base_theta._replace(we=we_val)
        for we_val in [0.2, 0.2]
    ]
    
    bolds, ts = run_sweep(
        theta_list=theta_list,
        setup=setup,
        seed=seed,
        n_devices=4,
        verbose=True,
        same_noise=True
    )
    print(f"✓ run_sweep with batching successful")
    print(f"  BOLD shape: {bolds.shape}")
    print(f"  Expected: (10, {ts.shape[0]}, {nn})")
    print(f"  Results differ: {not jnp.allclose(bolds[0], bolds[1])}")
except Exception as e:
    print(f"✗ run_sweep with batching failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
