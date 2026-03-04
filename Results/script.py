import os
import sys
import scipy
import warnings
from copy import deepcopy

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from src.utils import *
from src.bold_delay_utils import run_sweep

jax.config.update("jax_enable_x64", True)

seed = 42
np.random.seed(seed)
# torch.manual_seed(seed)

path = "outputs/explore"
os.makedirs(path, exist_ok=True)

PID_DIR = "/home/ziaee/Desktop/workstation/vbt_clinical/package/dataset"

data_conn = load_connectivity_data(PID_DIR)
Ceids, Leids, Seids, Rd1, Rd2, Rsero, region_names = data_conn


nn = len(region_names)
n_svar = 10

num_sim = 1
dt = 0.1
T = 0.5 * 60 * 1000  # total time in ms
tr = 500.0  # in ms (bold_decimate * num_skip * dt)
num_skip = 10  # skip steps in activity recording
t_cut = 10.0 * 1e3  # in ms
same_noise = True

v_c = 3.9  # conduction velocity in m/s
idelays = (Leids[Ceids != 0.0] / v_c / dt).astype(np.uint32)
num_time = int(T / dt)
Ja = np.load(join(PID_DIR, "Ja.npy"))


setup = {
    "Seids": Seids,
    "idelays": idelays,
    "horizon": 650,  # int(1.1 * idelays.max()),  # a bit larget than max delay
    "num_item": num_sim,  # jp_p1.shape[0],
    "dt": dt,
    "num_skip": num_skip,
    "num_time": num_time,
    "t_cut": t_cut,
    "init_state": jnp.array(
        [0.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(n_svar, 1),
    "bold_decimate": int(tr / (dt * num_skip)),
    "region_names": region_names,
    "noise": 0.15,
}

node_theta = gm.sigm_d1d2sero_default_theta._replace(
    I=46.5,
    Ja=Ja,
    Jsa=Ja,  # check shape in prepare
    Jsg=13,
    Jg=0,
    Jdopa=...,
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
    ws=0.0,
    sigma_V=setup["noise"],
    sigma_u=0.1 * setup["noise"],
)

JJdopa = 0.01
ws = 0.01
par_dict = {
    "Jdopa": 100000 * JJdopa,
    "ws": ws,
}
node_theta = node_theta._replace(**par_dict)


for i in range(2):
    tic = time.time()

    bolds, ts = run_sweep(
        node_theta,
        setup,
        verbose=True,
    )

    toc = time.time()
    print(f"Simulation completed in {toc - tic:.2f} seconds.")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ts / 1000, bolds[:, :2, 0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BOLD signal")
    ax.margins(x=0)
    ax.set_title("Simulated BOLD signal (first item)")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"bold_signal_{i+2}.png"))
    plt.close()
