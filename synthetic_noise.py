import os
head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)

from utils.common_utils import *

set_seed(1)

data_path = os.path.join(head_path,"Dataset")


# Load the noise-free sensor data

os.chdir(data_path)

data = np.load("FEM_precomputed.npz", allow_pickle=True)
v = data["v"]
f = data["f"]

data = np.load("sensor_data_noise_free.npz", allow_pickle=True)
sensor_node_idx = data["sensor_node_idx"].copy()
sensor_ur = data["sensor_ur"].copy()
sensor_uz = data["sensor_uz"].copy()

os.chdir(head_path)

print("v.shape =", v.shape)
print("f.shape =", f.shape)
print("sensor_node_idx.shape =", sensor_node_idx.shape)
print("sensor_ur.shape =", sensor_ur.shape)
print("sensor_uz.shape =", sensor_uz.shape)


# Add Gaussian noise to the sensor data

std_list = [1,2,3,4,5]

for std in std_list:


    set_seed(1)

    noise_std = std * 1e-6  # micrometers to meters

    n_sensor = len(sensor_uz)
    noise_ur = np.random.normal(loc=0.0, scale=noise_std, size=(n_sensor,))
    noise_uz = np.random.normal(loc=0.0, scale=noise_std, size=(n_sensor,))


    sensor_ur_noisy = sensor_ur + noise_ur
    sensor_uz_noisy = sensor_uz + noise_uz


    os.chdir(data_path)

    np.savez_compressed(
        "sensor_data_noise_{0}μm.npz".format(std),
        sensor_node_idx=sensor_node_idx,
        sensor_ur = sensor_ur_noisy.copy(),
        sensor_uz = sensor_uz_noisy.copy(),
    )

    os.chdir(head_path)

    # =============================================================================
    # Plot
    # =============================================================================
    x, y = v.T.copy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300, sharex=True)

    # Left: u_r
    axs[0].scatter(
        x[sensor_node_idx], sensor_ur,
        s=40, c="red", edgecolors="black", linewidths=1.0,
        label="Noise free"
    )
    axs[0].scatter(
        x[sensor_node_idx], sensor_ur_noisy,
        s=40, c="none", edgecolors="tab:blue", linewidths=1.0,
        label=r"Noise std ({0}$\mu$m)".format(std)
    )
    axs[0].set_xlabel("r [m]")
    axs[0].set_ylabel(r"$u_r$ [m]")
    axs[0].legend(loc="lower right", fontsize=8)
    axs[0].grid(alpha=0.3)

    # Right: u_z
    axs[1].scatter(
        x[sensor_node_idx], sensor_uz,
        s=40, c="red", edgecolors="black", linewidths=1.0,
        label="Noise free"
    )
    axs[1].scatter(
        x[sensor_node_idx], sensor_uz_noisy,
        s=40, c="none", edgecolors="tab:blue", linewidths=1.0,
        label=r"Noise std ({0}$\mu$m)".format(std)
    )
    axs[1].set_xlabel("r [m]")
    axs[1].set_ylabel(r"$u_z$ [m]")
    axs[1].legend(loc="lower right", fontsize=8)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


