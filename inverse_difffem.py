import os
head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)

from utils.common_utils import *
from utils.common_utils import plot_loss_history
from utils.common_utils import plot_param_history

from utils.loss_function import evaluate_loss_diffFEM


set_seed(1)

data_path = os.path.join(head_path,"Dataset")

noise_std_list = [0,1,2,3,4,5]

for noise_std in noise_std_list:


    # Load data

    os.chdir(data_path)

    # =============================================================================
    # Sensor data
    # =============================================================================
    if noise_std==0:
        data = np.load("sensor_data_noise_free.npz", allow_pickle=True)
    else:
        data = np.load("sensor_data_noise_{0}μm.npz".format(noise_std), allow_pickle=True)


    sensor_node_idx = data["sensor_node_idx"].copy()
    sensor_ur = data["sensor_ur"].copy()
    sensor_uz = data["sensor_uz"].copy()


    # =============================================================================
    # FEM matrix
    # =============================================================================
    data = np.load("FEM_precomputed.npz", allow_pickle=True)

    v = data["v"].copy()
    f = data["f"].copy()

    dirichlet = data["dirichlet"].copy()
    free      = data["free"].copy()

    Bot_line_idx    = data["Bot_line_idx"].copy()
    Left_line_idx   = data["Left_line_idx"].copy()
    Right_line_idx  = data["Right_line_idx"].copy()
    Top_line_idx    = data["Top_line_idx"].copy()
    Force_line_idx  = data["Force_line_idx"].copy()

    K_surface_unit  = data["K_surface_unit"].copy()
    K_base_unit     = data["K_base_unit"].copy()
    K_subgrade_unit = data["K_subgrade_unit"].copy()

    F = data["F"].copy()

    nnode = len(v)
    nel   = len(f)

    os.chdir(head_path)

    # =============================================================================
    #  Scaling (Normalization)
    # =============================================================================
    F_ref = 1e+3
    u_ref = 1e-3
    K_ref = F_ref / u_ref
    E_ref = F_ref / u_ref**2

    K_surface_unit  = K_surface_unit  / (K_ref / E_ref)
    K_base_unit     = K_base_unit     / (K_ref / E_ref)
    K_subgrade_unit = K_subgrade_unit / (K_ref / E_ref)

    F = F / F_ref

    sensor_ur = sensor_ur / u_ref
    sensor_uz = sensor_uz / u_ref

    # =============================================================================
    # numpy to torch
    # =============================================================================

    sensor_ur_torch = np_to_torch(sensor_ur)
    sensor_uz_torch = np_to_torch(sensor_uz)

    v_torch = np_to_torch(v)
    f_torch = np_to_torch(f)

    F_torch = np_to_torch(F)

    try:
        K_surface_unit_torch  = np_to_torch(K_surface_unit.item().toarray())
        K_base_unit_torch     = np_to_torch(K_base_unit.item().toarray())
        K_subgrade_unit_torch = np_to_torch(K_subgrade_unit.item().toarray())
    except:
        K_surface_unit_torch  = np_to_torch(K_surface_unit.toarray())
        K_base_unit_torch     = np_to_torch(K_base_unit.toarray())
        K_subgrade_unit_torch = np_to_torch(K_subgrade_unit.toarray())

    dirichlet_torch = np_to_torch(dirichlet).to(int)
    free_torch = np_to_torch(free).to(int)

    sensor_node_idx_torch = np_to_torch(sensor_node_idx).to(int)

    Bot_line_idx_torch   = np_to_torch(Bot_line_idx).to(int)
    Left_line_idx_torch  = np_to_torch(Left_line_idx).to(int)
    Right_line_idx_torch = np_to_torch(Right_line_idx).to(int)
    Top_line_idx_torch   = np_to_torch(Top_line_idx).to(int)
    Force_line_idx_torch = np_to_torch(Force_line_idx).to(int)


    # DiffFEM inversion (L-BFGS)


    def closure():
        optimizer.zero_grad()

        E_surface  = E_list[0]
        E_base     = E_list[1]
        E_subgrade = E_list[2]

        loss, ur_pred_torch, uz_pred_torch = evaluate_loss_diffFEM(E_surface, E_base, E_subgrade,
                                                                   K_surface_unit_torch, K_base_unit_torch, K_subgrade_unit_torch,
                                                                   F_torch, free_torch, dirichlet_torch, nnode, nel,
                                                                   sensor_ur_torch, sensor_uz_torch, sensor_node_idx_torch)
        loss.backward(retain_graph=False)

        # Record
        loss_history['loss_data'].append(loss.item())
        E_history["E1"].append( E_list[0].item() *E_ref )
        E_history["E2"].append( E_list[1].item() *E_ref )
        E_history["E3"].append( E_list[2].item() *E_ref )

        # Plot
        global iters; iters +=1
        if iters % 10 == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"L-BFGS solver : Epoch {iters} \t Elapsed Time = {elapsed_time_str}")
            plot_loss_history(loss_history)
            plot_param_history(E_history,
                                E_surface_true = 1378 * 1e6,
                                E_base_true    = 206 * 1e6,
                                E_subgrade_true= 69 * 1e6, )


        return loss



    E_history = {"E1": [],  "E2": [], "E3": [], }
    loss_history = {"loss_data": [], }
    E_list = nn.Parameter(np_to_torch(np.array([1e6, 1e6, 1e6]) / E_ref) )

    optimizer = torch.optim.LBFGS(
        [E_list],
        lr=1.0,
        max_iter=1000,
        max_eval=1000,
        tolerance_grad=1e-20,
        tolerance_change=1e-20,
        history_size=200,
        line_search_fn="strong_wolfe",
    )


    iters = 0
    start_time = time.time()

    loss = optimizer.step(closure)


    plot_loss_history(loss_history)
    plot_param_history(E_history,
                        E_surface_true = 1378 * 1e6,
                        E_base_true    = 206 * 1e6,
                        E_subgrade_true= 69 * 1e6, )




    save_path = os.path.join(head_path,"Results")
    os.makedirs(save_path, exist_ok=True)
    save_name = "inverse_DiffFEM_history_noise_{0}.npz".format(noise_std)

    np.savez(
        os.path.join(save_path, save_name),
        loss_data = np.array(loss_history["loss_data"]),
        E1 = np.array(E_history["E1"]),
        E2 = np.array(E_history["E2"]),
        E3 = np.array(E_history["E3"]),
        Elapsed_time = time.time() - start_time
    )

    os.chdir(head_path)

    torch.cuda.empty_cache()
