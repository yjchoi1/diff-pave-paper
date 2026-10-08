import numpy as np
import torch

def evaluate_loss_diffFEM(E_surface, E_base, E_subgrade,
                          K_surface, K_base, K_subgrade,
                          F, free, diriclet, nnode, nel,
                          sensor_ur, sensor_uz, sensor_idx):

    K_global = E_surface*K_surface + E_base*K_base + E_subgrade*K_subgrade
    u_f = torch.linalg.solve(K_global[free][:, free], F[free, None]).squeeze(1)
    u_pred_torch = torch.zeros_like(F).index_copy(0, free, u_f)

    ur_pred_torch = u_pred_torch[:nnode]
    uz_pred_torch = u_pred_torch[nnode:]

    loss  = torch.mean( (sensor_ur - ur_pred_torch[sensor_idx])**2)
    loss += torch.mean( (sensor_uz - uz_pred_torch[sensor_idx])**2)

    return loss, ur_pred_torch.detach(), uz_pred_torch.detach()


def evaluate_loss_PINN(E_surface, E_base, E_subgrade,
                       K_surface, K_base, K_subgrade,
                       F, free, diriclet, nnode, nel,
                       sensor_ur, sensor_uz, sensor_idx,
                       NN_surface, NN_base, NN_subgrade,
                       v_input,f,
                       ur_init=None, uz_init=None):

    K_global = E_surface*K_surface + E_base*K_base + E_subgrade*K_subgrade

    # Prediction
    u_1_pred = NN_surface( v_input )  * 1e-1
    u_2_pred = NN_base( v_input )     * 1e-1
    u_3_pred = NN_subgrade( v_input ) * 1e-1

    eps = 1e-3
    mask1 = ((v_input[:, 1] >  1.40 - eps ) & (v_input[:, 1] < 1.50 + eps )).unsqueeze(1)
    mask2 = ((v_input[:, 1] >= 1.15 - eps ) & (v_input[:, 1] < 1.40 - eps )).unsqueeze(1)
    mask3 = ((v_input[:, 1] >= 0.00 - eps ) & (v_input[:, 1] < 1.15 - eps )).unsqueeze(1)

    u_pred_torch = u_1_pred * mask1 + u_2_pred * mask2 + u_3_pred * mask3
    ur_pred_torch = u_pred_torch[:,0]
    uz_pred_torch = u_pred_torch[:,1]

    # Boundary conditions
    ur_pred_torch *= v_input[:,0]*(1.5-v_input[:,0])*v_input[:,1]
    uz_pred_torch *= v_input[:,1]

    if ur_init is not None:
        ur_pred_torch = ur_pred_torch * 1 + ur_init
    if uz_init is not None:
        uz_pred_torch = uz_pred_torch * 1 + uz_init

    # Physics loss
    u_pred_ = torch.cat((ur_pred_torch, uz_pred_torch), dim=0)
    residual = K_global @ u_pred_ - F

    residual[diriclet] = 0
    residual_x, residual_y = residual[:nnode], residual[nnode:]

    indices = (torch.abs(v_input[:,1] - 1.5) < 0.2 + eps)  & (v_input[:,0] < 0.2 + eps)
    residual_x[indices] *= 5
    residual_y[indices] *= 5

    residual = torch.cat((residual_x, residual_y))
    loss_physics = torch.mean( residual[free]**2 )



    # Data loss
    sensor_ur_pred_torch = ur_pred_torch[sensor_idx]
    sensor_uz_pred_torch = uz_pred_torch[sensor_idx]
    loss_data  = torch.mean( (sensor_ur - sensor_ur_pred_torch)**2)
    loss_data += torch.mean( (sensor_uz - sensor_uz_pred_torch)**2)

    # Interface continuity loss, weighted together with the data loss
    m115 = (torch.abs(v_input[:,1] - 1.15) < eps)
    loss_if_115 = ((NN_base(v_input[m115]) - NN_subgrade(v_input[m115]))**2).mean() if m115.any() else 0.0

    m140 = (torch.abs(v_input[:,1] - 1.40) < eps)
    loss_if_140 = ((NN_surface(v_input[m140]) - NN_base(v_input[m140]))**2).mean() if m140.any() else 0.0

    loss_data += loss_if_115 + loss_if_140

    return loss_physics, loss_data, ur_pred_torch.detach(), uz_pred_torch.detach(), residual_x.detach(), residual_y.detach()


def evaluate_loss_vanilla_PINN(E_surface, E_base, E_subgrade,
                               K_surface, K_base, K_subgrade,
                               F, free, diriclet, nnode, nel,
                               sensor_ur, sensor_uz, sensor_idx,
                               NN, v_input, f):

    K_global = E_surface * K_surface + E_base * K_base + E_subgrade * K_subgrade

    u_pred = NN(v_input) * 1e-1
    ur_pred_torch = u_pred[:, 0]
    uz_pred_torch = u_pred[:, 1]

    # Boundary conditions
    ur_pred_torch = ur_pred_torch * v_input[:, 0] * (1.5 - v_input[:, 0]) * v_input[:, 1]
    uz_pred_torch = uz_pred_torch * v_input[:, 1]

    # Physics loss
    u_pred_ = torch.cat((ur_pred_torch, uz_pred_torch), dim=0)
    residual = K_global @ u_pred_ - F

    residual[diriclet] = 0
    residual_x, residual_y = residual[:nnode], residual[nnode:]

    eps = 1e-3
    indices = (torch.abs(v_input[:, 1] - 1.5) < 0.2 + eps) & (v_input[:, 0] < 0.2 + eps)
    residual_x[indices] *= 5
    residual_y[indices] *= 5

    residual = torch.cat((residual_x, residual_y))
    loss_physics = torch.mean(residual[free] ** 2)

    # Data loss
    sensor_ur_pred_torch = ur_pred_torch[sensor_idx]
    sensor_uz_pred_torch = uz_pred_torch[sensor_idx]
    loss_data  = torch.mean((sensor_ur - sensor_ur_pred_torch) ** 2)
    loss_data += torch.mean((sensor_uz - sensor_uz_pred_torch) ** 2)

    return loss_physics, loss_data, ur_pred_torch.detach(), uz_pred_torch.detach(), residual_x.detach(), residual_y.detach()


def solve_forward(E_surface, E_base, E_subgrade,
                  K_surface, K_base, K_subgrade,
                  F, free, dirichlet, nnode, nel=None):
    """
    - If K_* are torch tensors => solve entirely in torch (keeps autograd graph)
    - If K_* are numpy/scipy => solve with scipy.sparse + spsolve

    Inputs
      - K_surface/base/subgrade: (2*nnode, 2*nnode) stiffness unit matrices
      - E_surface/base/subgrade: scalars (float or torch scalar)
      - F: (2*nnode,) load vector
      - dirichlet: indices of fixed dofs (length ndir)
      - free: ignored (computed from dirichlet for safety/consistency)
      - nnode: number of nodes

    Returns
      - ur, uz : displacement components, same backend type as K (torch or numpy)
    """

    # -----------------------------
    # Detect torch backend
    # -----------------------------
    is_torch = (
        hasattr(K_surface, "is_sparse") or
        (type(K_surface).__module__.startswith("torch"))
    )

    if is_torch:
        import torch

        # Ensure tensors
        def _to_torch(x, *, dtype=None, device=None):
            if torch.is_tensor(x):
                return x
            return torch.tensor(x, dtype=dtype, device=device)

        # Infer dtype/device from K_surface
        device = K_surface.device
        dtype  = K_surface.dtype

        E_surface  = _to_torch(E_surface,  dtype=dtype, device=device)
        E_base     = _to_torch(E_base,     dtype=dtype, device=device)
        E_subgrade = _to_torch(E_subgrade, dtype=dtype, device=device)

        F = _to_torch(F, dtype=dtype, device=device).reshape(-1)

        # dirichlet indices -> torch long on same device
        if not torch.is_tensor(dirichlet):
            dirichlet_t = torch.tensor(np.array(dirichlet, dtype=np.int64), device=device, dtype=torch.long)
        else:
            dirichlet_t = dirichlet.to(device=device, dtype=torch.long).reshape(-1)

        ndof = 2 * nnode
        total = torch.arange(ndof, device=device, dtype=torch.long)

        # free = complement of dirichlet (torch way)
        # Using boolean mask (stable + fast)
        mask = torch.ones(ndof, device=device, dtype=torch.bool)
        mask[dirichlet_t] = False
        free_t = total[mask]

        # Initialize solution (dirichlet values assumed 0 here)
        sol = torch.zeros(ndof, device=device, dtype=dtype)

        # Assemble global K (keep as torch)
        # Works for dense or sparse K_* (but note later slicing)
        K_global = E_surface * K_surface + E_base * K_base + E_subgrade * K_subgrade

        # Helper: index_select rows & cols for dense/sparse
        # For sparse: easiest is to go dense for the submatrices we solve with.
        if getattr(K_global, "is_sparse", False) or getattr(K_global, "is_sparse_csr", False) or getattr(K_global, "is_sparse_coo", False):
            K_dense = K_global.to_dense()
        else:
            K_dense = K_global

        # Extract sub-blocks
        K_ff = K_dense.index_select(0, free_t).index_select(1, free_t)
        K_fd = K_dense.index_select(0, free_t).index_select(1, dirichlet_t)

        # Right-hand side
        # sol[dirichlet] = 0 so term is typically zero, but keep generic
        F_f = F.index_select(0, free_t) - K_fd @ sol.index_select(0, dirichlet_t)

        # Solve (autograd-friendly)
        # K_ff: (nfree, nfree), F_f: (nfree,)
        sol_f = torch.linalg.solve(K_ff, F_f.unsqueeze(-1)).squeeze(-1)

        sol = sol.clone()
        sol[free_t] = sol_f

        u_sol = sol
        ur = u_sol[:nnode].reshape(-1)
        uz = u_sol[nnode:].reshape(-1)
        return ur, uz

    else:
        # -----------------------------
        # Numpy / SciPy backend
        # -----------------------------
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve

        K_surface_unit  = csc_matrix(K_surface)
        K_base_unit     = csc_matrix(K_base)
        K_subgrade_unit = csc_matrix(K_subgrade)

        K_global = (E_surface * K_surface_unit
                    + E_base * K_base_unit
                    + E_subgrade * K_subgrade_unit)

        ndof = 2 * nnode
        total = np.arange(ndof)
        dirichlet = np.asarray(dirichlet, dtype=np.int64).reshape(-1)
        free = np.setdiff1d(total, dirichlet)

        sol = np.zeros(ndof, dtype=np.asarray(F).dtype)

        K = csc_matrix(K_global)
        K_ff = K[free][:, free]
        K_fd = K[free][:, dirichlet]

        F = np.asarray(F).reshape(-1)
        F_f = F[free] - K_fd @ sol[dirichlet]

        sol_f = spsolve(K_ff, F_f)
        sol[free] = sol_f.copy()

        ur = sol[:nnode].copy().reshape(-1)
        uz = sol[nnode:].copy().reshape(-1)
        return ur, uz
