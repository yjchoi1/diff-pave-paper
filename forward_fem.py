import os
head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)

from utils.common_utils import *

set_seed(1)

data_path = os.path.join(head_path,"Dataset")

os.chdir(data_path)

data = np.load("mesh_information.npz")

v                = data["v"]
f                = data["f"]
surface_el_idx   = data["surface_el_idx"]
base_el_idx      = data["base_el_idx"]
subgrade_el_idx  = data["subgrade_el_idx"]
sensor_node_idx  = data["sensor_node_idx"]

print("v.shape               =", v.shape)
print("f.shape               =", f.shape)
print("surface_el_idx.shape  =", surface_el_idx.shape)
print("base_el_idx.shape     =", base_el_idx.shape)
print("subgrade_el_idx.shape =", subgrade_el_idx.shape)
print("sensor_node_idx.shape =", sensor_node_idx.shape)


W = 1.5
H = 1.5
load_W = 0.15


r, z = v[:,0].copy(), v[:,1].copy()
elem_centroid = v[f].mean(axis=1)

eps = 1e-6

# --- boundary node indices
Bot_line_idx   = np.where(np.abs(z) < eps)[0]
Left_line_idx  = np.where(np.abs(r) < eps)[0]
Right_line_idx = np.where(np.abs(W - r) < eps)[0]
Top_line_idx   = np.where(np.abs(H - z) < eps)[0]

Force_line_idx = np.where(
    (0 <= r) & (r <= load_W) &
    (H - eps <= z) & (z <= H + eps)
)[0]

def sort_idx(idx):
    return idx[np.lexsort((v[idx][:, 0], v[idx][:, 1]))]

Bot_line_idx   = sort_idx(Bot_line_idx)
Left_line_idx  = sort_idx(Left_line_idx)
Right_line_idx = sort_idx(Right_line_idx)
Top_line_idx   = sort_idx(Top_line_idx)
Force_line_idx = sort_idx(Force_line_idx)
sensor_node_idx = sort_idx(sensor_node_idx)

print("-"*50)
print("Bot_line_idx.shape   =", Bot_line_idx.shape)
print("Left_line_idx.shape  =", Left_line_idx.shape)
print("Right_line_idx.shape  =", Right_line_idx.shape)
print("Force_line_idx.shape =", Force_line_idx.shape)

nnode = len(v)
nel = len(f)


os.chdir(head_path)

# Plot mesh

from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

# Left: element groups
ax = axs[0]

el_lists = [surface_el_idx, base_el_idx, subgrade_el_idx]
colors   = ['red', 'blue', 'black']
labels   = ['Surface', 'Base', 'Subgrade']

n_groups = len(el_lists)
group_id = np.full(len(f), n_groups, dtype=int)  # default: Others

for gid, idx in enumerate(el_lists):
    group_id[idx] = gid

segments = [[] for _ in range(n_groups + 1)]
for e_idx, elem in enumerate(f):
    pts  = v[elem[[0, 1, 2, 3, 0]]]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    segments[group_id[e_idx]].append(segs)

for gid in range(n_groups + 1):
    if not segments[gid]:
        continue
    seg_arr = np.concatenate(segments[gid], axis=0)
    if gid < n_groups:
        ax.add_collection(LineCollection(seg_arr, colors=colors[gid], linewidths=1.0, alpha=0.1))
    else:
        ax.add_collection(LineCollection(seg_arr, colors='k', linewidths=0.4, alpha=0.1))

proxies = [mlines.Line2D([], [], color=c, linewidth=1.5, label=lb)
           for c, lb in zip(colors, labels)]
proxies.append(mlines.Line2D([], [], color='k', linewidth=0.8, label='Others'))

ax.set_aspect('equal')
ax.set_title('Mesh')
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.autoscale()

# Right: node sets
ax2 = axs[1]
ax2.scatter(v[:, 0], v[:, 1], s=1,  color='gray',   label='All nodes', alpha=0.1)
ax2.scatter(v[Bot_line_idx, 0],    v[Bot_line_idx, 1],    s=7,  color='blue',   label='Bottom')
ax2.scatter(v[Left_line_idx, 0],   v[Left_line_idx, 1],   s=7,  color='orange', label='Left')
ax2.scatter(v[Right_line_idx, 0],  v[Right_line_idx, 1],  s=7,  color='green',  label='Right')
ax2.scatter(v[Force_line_idx, 0],  v[Force_line_idx, 1],  s=12, color='black',  label='Force')
ax2.scatter(v[sensor_node_idx, 0], v[sensor_node_idx, 1], s=30, color='red',
            marker='x', linewidths=2, label='Sensor')

ax2.set_aspect('equal')
ax2.set_title('Node sets')
ax2.set_xlabel('r')
ax2.set_ylabel('z')
ax2.legend(loc='best')

plt.tight_layout()
plt.show()
plt.close()


from matplotlib.collections import LineCollection

fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

el_lists = [surface_el_idx, base_el_idx, subgrade_el_idx]
colors   = ['red', 'blue', 'black']

n_groups = len(el_lists)
group_id = np.full(len(f), n_groups, dtype=int)  # default: Others

for gid, idx in enumerate(el_lists):
    group_id[idx] = gid

segments = [[] for _ in range(n_groups + 1)]
for e_idx, elem in enumerate(f):
    pts  = v[np.array(elem)[[0, 1, 2, 3, 0]]]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    segments[group_id[e_idx]].append(segs)

for gid in range(n_groups + 1):
    if not segments[gid]:
        continue
    seg_arr = np.concatenate(segments[gid], axis=0)
    if gid < n_groups:
        lc = LineCollection(seg_arr, colors=colors[gid], linewidths=1.0, alpha=0.1)
    else:
        lc = LineCollection(seg_arr, colors='k', linewidths=0.4, alpha=0.1)
    ax.add_collection(lc)

# Update the data limits after adding the collections
ax.autoscale(enable=True)
ax.autoscale_view()

ax.set_aspect('equal', adjustable='box')

# Hide the axes
ax.set_axis_off()

plt.tight_layout(pad=0)
plt.show()
plt.close()


import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

handles = [
    mlines.Line2D([], [], color='red',   lw=2, label='Surface'),
    mlines.Line2D([], [], color='blue',  lw=2, label='Base'),
    mlines.Line2D([], [], color='black', lw=2, label='Subgrade'),
]

ax.legend(
    handles=handles,
    loc='center',
    frameon=False,
    fontsize=10
)

ax.set_axis_off()
plt.tight_layout(pad=0)
plt.show()
plt.close()



# FEM pre-computation

def shape_Q4(xi, eta):
    """
    xi, eta: scalar (isoparametric coords in [-1, 1])
    return:
        N  : (4,)   shape functions
        dN : (4, 2) [dN/dxi, dN/deta]
    """
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)

    N = np.array([N1, N2, N3, N4], dtype=float)

    # dN/dxi
    dN1_dxi = -0.25 * (1 - eta)
    dN2_dxi =  0.25 * (1 - eta)
    dN3_dxi =  0.25 * (1 + eta)
    dN4_dxi = -0.25 * (1 + eta)

    # dN/deta
    dN1_deta = -0.25 * (1 - xi)
    dN2_deta = -0.25 * (1 + xi)
    dN3_deta =  0.25 * (1 + xi)
    dN4_deta =  0.25 * (1 - xi)

    dN = np.array([
        [dN1_dxi, dN1_deta],
        [dN2_dxi, dN2_deta],
        [dN3_dxi, dN3_deta],
        [dN4_dxi, dN4_deta],
    ], dtype=float)

    return N, dN

def C_matrix(E, v):
    lam = E * v / ((1 + v) * (1 - 2*v))
    mu  = E / (2 * (1 + v))
    C = np.array([
        [lam + 2*mu, lam,         lam,         0.0],
        [lam,        lam + 2*mu,  lam,         0.0],
        [lam,        lam,         lam + 2*mu,  0.0],
        [0.0,        0.0,         0.0,         mu ],
    ], dtype=float)
    return C

def assemble_K_sparse(v, f, K_el, fmt="csc"):
    N = len(v)
    dm = np.concatenate([f, f + N], axis=1)          # (nel, 8)

    I = np.repeat(dm, 8, axis=1)                     # (nel, 64)
    J = np.tile(dm, (1, 8))                          # (nel, 64)
    V = K_el.reshape(-1, 64)                         # (nel, 64)

    K = coo_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(2*N, 2*N))
    return K.tocsc() if fmt == "csc" else K.tocsr()

# =============================================================================
# matrix N & B
# =============================================================================
xi_gp = 1.0 / np.sqrt(3.0)
gauss_pts = [(-xi_gp, -xi_gp),
             ( xi_gp, -xi_gp),
             ( xi_gp,  xi_gp),
             (-xi_gp,  xi_gp)]
gauss_weight = np.array([1, 1, 1, 1])

N_all  = []
dN_all = []
for (xi, eta) in gauss_pts:
    N, dN = shape_Q4(xi, eta)
    N_all.append(N)
    dN_all.append(dN)

N_all  = np.stack(N_all,  axis=0)  # (4_gp, 4)
dN_all = np.stack(dN_all, axis=0)  # (4_gp, 4, 2)

dx_dxi = np.einsum("eni,wnj->weij",v[f],dN_all)
dxi_dx = np.linalg.inv(dx_dxi)
Jacob = np.linalg.det(dx_dxi)
dNdx = np.einsum("wni,weij->wenj", dN_all, dxi_dx)

radius_gp = np.einsum("wn,en->we", N_all, v[f][:, :, 0] )  # (w, e)
intg_weight = np.einsum("wi,w->wi", Jacob, gauss_weight)  * (2.0 * np.pi * radius_gp)

B = np.zeros((4, len(f), 4, 8))
B[:,:,0,:4] += dNdx[:,:,:,0].copy()        # dN/dr * u_r
B[:,:,1,4:] += dNdx[:,:,:,1].copy()        # dN/dz * u_z
B[:,:,2,:4] += (N_all[:, None, :] / radius_gp[:, :, None])
B[:,:,3,:4] += dNdx[:,:,:,1].copy()        # dN/dz * u_r
B[:,:,3,4:] += dNdx[:,:,:,0].copy()        # dN/dr * u_z


# =============================================================================
# Unit K matrix
# =============================================================================
E_unit = 1.0
xnu_surface  = 0.35
xnu_base     = 0.40
xnu_subgrade = 0.45

# --- Surface
C_surface = C_matrix(E_unit, xnu_surface)  # (4,4)
K_el_surface = np.zeros((len(f), 8, 8), dtype=B.dtype)
K_el_surface[surface_el_idx] = np.einsum(
    "wemi,mn,wenj,we->eij",
    B[:, surface_el_idx], C_surface, B[:, surface_el_idx], intg_weight[:, surface_el_idx]
)

# --- Base
C_base = C_matrix(E_unit, xnu_base)        # (4,4)
K_el_base = np.zeros((len(f), 8, 8), dtype=B.dtype)
K_el_base[base_el_idx] = np.einsum(
    "wemi,mn,wenj,we->eij",
    B[:, base_el_idx], C_base, B[:, base_el_idx], intg_weight[:, base_el_idx]
)

# --- Subgrade
C_subgrade = C_matrix(E_unit, xnu_subgrade)  # (4,4)
K_el_subgrade = np.zeros((len(f), 8, 8), dtype=B.dtype)
K_el_subgrade[subgrade_el_idx] = np.einsum(
    "wemi,mn,wenj,we->eij",
    B[:, subgrade_el_idx], C_subgrade, B[:, subgrade_el_idx], intg_weight[:, subgrade_el_idx]
)


K_surface_unit  = assemble_K_sparse(v, f, K_el_surface,  fmt="csc")
K_base_unit     = assemble_K_sparse(v, f, K_el_base,     fmt="csc")
K_subgrade_unit = assemble_K_sparse(v, f, K_el_subgrade, fmt="csc")


# =============================================================================
# Force vector F
# =============================================================================
pressure = 700*1e3  # Pa

F = np.zeros(2 * len(v))

idx0 = Force_line_idx[:-1]
idx1 = Force_line_idx[1:]

r0 = v[idx0, 0]
r1 = v[idx1, 0]
z0 = v[idx0, 1]
z1 = v[idx1, 1]

L = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)

F_i = (-pressure) * np.pi * L * (2*r0 + r1) / 3.0
F_j = (-pressure) * np.pi * L * (r0 + 2*r1) / 3.0

F[idx0 + len(v)] += F_i
F[idx1 + len(v)] += F_j


# Solve


from scipy.sparse import csc_matrix

K_surface_unit  = csc_matrix(K_surface_unit)
K_base_unit     = csc_matrix(K_base_unit)
K_subgrade_unit = csc_matrix(K_subgrade_unit)

E_surface  = 1378 * 1e6
E_base     =  206 * 1e6
E_subgrade =   69 * 1e6

K_global = E_surface*K_surface_unit + E_base*K_base_unit + E_subgrade*K_subgrade_unit


from scipy.sparse.linalg import spsolve
dirichlet = np.unique(np.concatenate((
    Bot_line_idx,          # u_r at bottom
    Bot_line_idx + len(v), # u_z at bottom
    Left_line_idx,         # u_r at left
    Right_line_idx,        # u_r at right

)))

total = np.arange(2*len(v))
free  = np.setdiff1d(total, dirichlet)

sol = np.zeros(2*len(v))


K = csc_matrix(K_global)

K_ff = K[free][:, free]
K_fd = K[free][:, dirichlet]

F_f = F[free] - K_fd @ sol[dirichlet]

sol_f = spsolve(K_ff, F_f)

sol[free] = sol_f.copy()
u_sol = sol.copy()


ur = u_sol[:len(v)].copy()
uz = u_sol[len(v):].copy()


# Plot displacement

import matplotlib.tri as mtri

triangles = np.vstack([
    f[:, [0, 1, 2]],
    f[:, [0, 2, 3]]
])
tri = mtri.Triangulation(v[:,0], v[:,1], triangles)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300,
                        sharex=True, sharey=False)

# --- u_r ---
tc1 = axs[0].tripcolor(
    tri, ur,
    shading="gouraud",
    cmap="RdBu_r"
)
axs[0].set_title(r"$u_r$")
axs[0].set_xlabel("r")
axs[0].set_ylabel("z")
axs[0].set_aspect("equal")

cbar1 = fig.colorbar(tc1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.set_label(r"$u_r$ [m]")

# --- u_z ---
tc2 = axs[1].tripcolor(
    tri, uz,
    shading="gouraud",
    cmap="RdBu_r"
)
axs[1].set_title(r" $u_z$")
axs[1].set_xlabel("r ")
axs[1].set_aspect("equal")

cbar2 = fig.colorbar(tc2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.set_label(r"$u_z$ [m]")

plt.tight_layout()
plt.show()
plt.close()


# Plot strain

u_sol_el = np.concatenate( (ur[f], uz[f]), axis=1)
strain = np.einsum("weij, ej -> wei", B, u_sol_el)
e_rr = strain[:,:,0].mean(0).copy()   # (nel,)
e_zz = strain[:,:,1].mean(0).copy()   # (nel,)
e_tt = strain[:,:,2].mean(0).copy()   # (nel,)
e_rz = strain[:,:,3].mean(0).copy() / 2   # (nel,)

el_center_coord = v[f].mean(1) # (nel,2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

polys = v[f]  # v: (nnode,2), f: (nel,4)

fields = [
    (e_rr, r"$\varepsilon_{rr}$"),
    (e_zz, r"$\varepsilon_{zz}$"),
    (e_tt, r"$\varepsilon_{\theta\theta}$"),
    (e_rz, r"$\varepsilon_{rz}$"),
]

fig, axs = plt.subplots(2, 2, figsize=(7, 6), dpi=300, sharex=True, sharey=True)
axs = axs.ravel()

for ax, (val, title) in zip(axs, fields):
    val = np.asarray(val).reshape(-1)

    pc = PolyCollection(
        polys,
        array=val,          # (nel,) element values
        cmap="RdBu_r",
        edgecolors="k",
        linewidths=0.0
    )
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("r")
    ax.set_ylabel("z")

    cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

plt.tight_layout()
plt.show()
plt.close()



# Plot top-surface displacement and sensors

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300, sharex=True)

# Left: u_r on the top surface
axs[0].plot(
    v[Top_line_idx, 0],
    ur[Top_line_idx],
    color='k',
    lw=1.5,
    label=r'$u_r$ (top)'
)

axs[0].scatter(
    v[sensor_node_idx, 0],
    ur[sensor_node_idx],
    s=40,
    facecolors='red',
    edgecolors='black',
    linewidths=1.0,
    label='Sensor'
)

axs[0].set_xlabel('r [m]')
axs[0].set_ylabel(r'$u_r$ [m]')
axs[0].set_title(r'Top surface $u_r$')
axs[0].legend(loc='lower right')
axs[0].grid(alpha=0.3)

# Right: u_z on the top surface
axs[1].plot(
    v[Top_line_idx, 0],
    uz[Top_line_idx],
    color='k',
    lw=1.5,
    label=r'$u_z$ (top)'
)

axs[1].scatter(
    v[sensor_node_idx, 0],
    uz[sensor_node_idx],
    s=40,
    facecolors='red',
    edgecolors='black',
    linewidths=1.0,
    label='Sensor'
)

axs[1].set_xlabel('r [m]')
axs[1].set_ylabel(r'$u_z$ [m]')
axs[1].set_title(r'Top surface $u_z$')
axs[1].legend(loc='lower right')
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()


# Save data

os.chdir(data_path)

np.savez_compressed(
    "FEM_precomputed.npz",
    v=v,
    f=f,
    dirichlet=dirichlet,
    free=free,

    Bot_line_idx=Bot_line_idx,
    Left_line_idx=Left_line_idx,
    Right_line_idx=Right_line_idx,
    Top_line_idx=Top_line_idx,
    Force_line_idx=Force_line_idx,

    K_surface_unit=K_surface_unit,
    K_base_unit=K_base_unit,
    K_subgrade_unit=K_subgrade_unit,
    F=F

)


np.savez_compressed(
    "sensor_data_noise_free.npz",
    sensor_node_idx=sensor_node_idx,
    sensor_ur = ur[sensor_node_idx].copy(),
    sensor_uz = uz[sensor_node_idx].copy(),
)


os.chdir(head_path)
