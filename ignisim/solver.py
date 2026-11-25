from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import math
import numpy as np

SIGMA_SB = 5.670374419e-8

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrap(fn): return fn
        return wrap


@dataclass
class SimulationParams:
    heating_mode: str = "Laser"
    source_profile: str = "Top-hat"
    laser_input: str = "Power"
    laser_power_W: float = 0.167
    incident_flux_W_m2: float = 20000.0
    spot_width_m: float = 0.001
    spot_center_m: float = 0.025
    out_of_plane_m: float = 0.01

    layer1_thickness_m: float = 3.0e-5
    layer2_thickness_m: float = 5.0e-3
    sample_width_m: float = 0.05

    kx1: float = 2.0
    ky1: float = 390.0
    rho1: float = 2000.0
    cp1: float = 800.0
    kx2: float = 0.55
    ky2: float = 0.55
    rho2: float = 1300.0
    cp2: float = 1700.0

    T_initial_K: float = 298.0
    T_ambient_K: float = 298.0
    T_surroundings_K: float = 298.0
    absorptivity: float = 1.0
    h_front_W_m2K: float = 15.0
    emissivity_model: str = "Constant"
    emissivity: float = 0.9
    eps0: float = 0.9
    eps1: float = 0.9
    eps_Tc_K: float = 623.15
    eps_dT_K: float = 25.0
    back_bc: str = "Adiabatic"

    ignition_C: float = 381.0
    duration_s: float = 300.0
    Nx: int = 90
    Ny: int = 60
    dt_s: float = 0.0  # <=0 means automatic stable dt
    stability_factor: float = 0.88
    max_steps: int = 2_000_000
    snapshots: int = 6
    history_points: int = 400


def quality_preset(name: str) -> Dict[str, int]:
    name = (name or "Balanced").lower()
    if name.startswith("preview"):
        return {"Nx": 55, "Ny": 35, "max_steps": 250_000}
    if name.startswith("high"):
        return {"Nx": 140, "Ny": 90, "max_steps": 4_000_000}
    return {"Nx": 90, "Ny": 60, "max_steps": 2_000_000}


def _source_profile(y: np.ndarray, p: SimulationParams) -> Tuple[np.ndarray, float, float]:
    Ly = p.sample_width_m
    if p.heating_mode.lower().startswith("uniform"):
        q_in = float(p.incident_flux_W_m2)
        q = np.full_like(y, q_in, dtype=np.float64)
        P_incident = q_in * Ly * p.out_of_plane_m
        return q, q_in, P_incident

    width = max(float(p.spot_width_m), 1e-12)
    oop = max(float(p.out_of_plane_m), 1e-12)
    if p.laser_input.lower().startswith("power"):
        P_incident = float(p.laser_power_W)
        if p.source_profile.lower().startswith("gauss"):
            sigma = width / 2.355  # spot width interpreted as FWHM
            shape = np.exp(-0.5 * ((y - p.spot_center_m) / max(sigma, 1e-12)) ** 2)
            area_factor = np.trapezoid(shape, y) * oop
            q = shape * (P_incident / max(area_factor, 1e-30))
            q_in = float(np.max(q))
        else:
            q_in = P_incident / (width * oop)
            mask = np.abs(y - p.spot_center_m) <= width / 2.0
            if not np.any(mask):
                mask[int(np.argmin(np.abs(y - p.spot_center_m)))] = True
            q = np.zeros_like(y, dtype=np.float64)
            q[mask] = q_in
        return q, q_in, P_incident

    q_in = float(p.incident_flux_W_m2)
    if p.source_profile.lower().startswith("gauss"):
        sigma = width / 2.355
        q = q_in * np.exp(-0.5 * ((y - p.spot_center_m) / max(sigma, 1e-12)) ** 2)
        P_incident = float(np.trapezoid(q, y) * oop)
    else:
        mask = np.abs(y - p.spot_center_m) <= width / 2.0
        if not np.any(mask):
            mask[int(np.argmin(np.abs(y - p.spot_center_m)))] = True
        q = np.zeros_like(y, dtype=np.float64)
        q[mask] = q_in
        P_incident = q_in * width * oop
    return q, q_in, P_incident


def _material_columns(x: np.ndarray, p: SimulationParams):
    is_l1 = x <= p.layer1_thickness_m
    kx = np.where(is_l1, p.kx1, p.kx2).astype(np.float64)
    ky = np.where(is_l1, p.ky1, p.ky2).astype(np.float64)
    rho_cp = np.where(is_l1, p.rho1 * p.cp1, p.rho2 * p.cp2).astype(np.float64)
    return kx, ky, rho_cp


@njit(cache=True)
def _run_explicit(
    T, kx_col, ky_col, rho_cp_col, q_y, dx, dy, dt, nsteps,
    absorptivity, h_front, T_inf, T_sur, eps_model_flag, emissivity,
    eps0, eps1, eps_Tc_K, eps_dT_K, back_fixed, ignition_K,
    snap_steps, hist_stride
):
    Nx, Ny = T.shape
    Tn = T.copy()
    ns = len(snap_steps)
    snaps = np.empty((ns, Nx, Ny), dtype=np.float64)
    snap_times_step = np.empty(ns, dtype=np.int64)
    snap_counter = 0
    max_hist = nsteps // hist_stride + 2
    hist_times_step = np.empty(max_hist, dtype=np.int64)
    hist_surface_max = np.empty(max_hist, dtype=np.float64)
    hist_depth_center = np.empty((max_hist, Nx), dtype=np.float64)
    hist_counter = 0
    center_j = Ny // 2
    ignition_step = -1

    for n in range(nsteps):
        # interior
        for i in range(1, Nx - 1):
            kxi = kx_col[i]
            kx_l = 2.0 * kxi * kx_col[i-1] / max(kxi + kx_col[i-1], 1e-30)
            kx_r = 2.0 * kxi * kx_col[i+1] / max(kxi + kx_col[i+1], 1e-30)
            kyi = ky_col[i]
            for j in range(1, Ny - 1):
                divx = (kx_r * (T[i+1,j]-T[i,j]) - kx_l * (T[i,j]-T[i-1,j])) / (dx*dx)
                divy = kyi * (T[i,j+1] - 2.0*T[i,j] + T[i,j-1]) / (dy*dy)
                Tn[i,j] = T[i,j] + dt * (divx + divy) / rho_cp_col[i]

        # lateral adiabatic
        for i in range(1, Nx - 1):
            Tn[i,0] = Tn[i,1]
            Tn[i,Ny-1] = Tn[i,Ny-2]

        # back boundary
        if back_fixed == 1:
            for j in range(Ny): Tn[Nx-1,j] = T_inf
        else:
            for j in range(Ny): Tn[Nx-1,j] = Tn[Nx-2,j]

        # front boundary with convection + radiation
        for j in range(Ny):
            Ts = T[0,j]
            eps = emissivity
            if eps_model_flag == 1:
                z = (Ts - eps_Tc_K) / max(eps_dT_K, 1e-12)
                if z > 60.0: logistic = 1.0
                elif z < -60.0: logistic = 0.0
                else: logistic = 1.0 / (1.0 + math.exp(-z))
                eps = eps0 + (eps1 - eps0) * logistic
                eps = min(1.0, max(0.0, eps))
            qnet = absorptivity*q_y[j] - h_front*(Ts-T_inf) - eps*SIGMA_SB*(Ts**4-T_sur**4)
            Tn[0,j] = Tn[1,j] + dx*qnet/max(kx_col[0], 1e-30)

        T, Tn = Tn, T
        step = n + 1

        smax = T[0,0]
        for j in range(1, Ny):
            if T[0,j] > smax: smax = T[0,j]
        if ignition_step < 0 and smax >= ignition_K:
            ignition_step = step

        if step % hist_stride == 0 or step == nsteps:
            hist_times_step[hist_counter] = step
            hist_surface_max[hist_counter] = smax
            for i in range(Nx): hist_depth_center[hist_counter,i] = T[i,center_j]
            hist_counter += 1

        while snap_counter < ns and step >= snap_steps[snap_counter]:
            snaps[snap_counter,:,:] = T
            snap_times_step[snap_counter] = step
            snap_counter += 1

    return (T, snaps[:snap_counter], snap_times_step[:snap_counter], ignition_step,
            hist_times_step[:hist_counter], hist_surface_max[:hist_counter], hist_depth_center[:hist_counter])


def simulate(p: SimulationParams) -> Dict[str, Any]:
    if p.Nx < 5 or p.Ny < 5:
        raise ValueError("Nx and Ny must both be at least 5.")
    if p.layer1_thickness_m <= 0 or p.layer2_thickness_m <= 0 or p.sample_width_m <= 0:
        raise ValueError("Geometry dimensions must be positive.")
    if p.duration_s <= 0:
        raise ValueError("Simulation duration must be positive.")

    Lx = p.layer1_thickness_m + p.layer2_thickness_m
    x = np.linspace(0.0, Lx, int(p.Nx))
    y = np.linspace(0.0, p.sample_width_m, int(p.Ny))
    dx = float(x[1]-x[0]); dy = float(y[1]-y[0])
    kx_col, ky_col, rho_cp_col = _material_columns(x, p)

    dt_limits = rho_cp_col / (2.0*kx_col/(dx*dx) + 2.0*ky_col/(dy*dy))
    dt_max = float(np.min(dt_limits))
    dt_requested = float(p.dt_s)
    dt = p.stability_factor * dt_max if dt_requested <= 0 else dt_requested
    if dt > dt_max:
        raise ValueError(f"Unstable timestep: dt={dt:.3e} s exceeds dt_max={dt_max:.3e} s. Use Auto or reduce dt.")

    nsteps = int(math.ceil(p.duration_s / dt))
    if nsteps > int(p.max_steps):
        raise ValueError(
            f"This configuration requires {nsteps:,} explicit steps, above the selected limit of {p.max_steps:,}. "
            "Choose a lower resolution, shorter duration, or a larger allowable step budget."
        )
    dt = p.duration_s / nsteps

    q_y, q_peak, P_incident = _source_profile(y, p)
    T0 = np.full((p.Nx, p.Ny), float(p.T_initial_K), dtype=np.float64)
    nsnap = max(2, int(p.snapshots))
    snap_steps = np.unique(np.clip(np.rint(np.linspace(1, nsteps, nsnap)).astype(np.int64), 1, nsteps))
    hist_stride = max(1, nsteps // max(10, int(p.history_points)))

    outputs = _run_explicit(
        T0, kx_col, ky_col, rho_cp_col, q_y, dx, dy, dt, nsteps,
        float(p.absorptivity), float(p.h_front_W_m2K), float(p.T_ambient_K), float(p.T_surroundings_K),
        1 if p.emissivity_model.lower().startswith("temp") else 0, float(p.emissivity),
        float(p.eps0), float(p.eps1), float(p.eps_Tc_K), float(p.eps_dT_K),
        1 if p.back_bc.lower().startswith("fixed") else 0, float(p.ignition_C + 273.15),
        snap_steps, hist_stride
    )
    Tfinal, snaps, snap_steps_done, ignition_step, hist_steps, hist_surface_max, hist_depth = outputs
    snap_times = snap_steps_done.astype(float) * dt
    hist_times = hist_steps.astype(float) * dt
    ignition_time = None if ignition_step < 0 else float(ignition_step*dt)

    return {
        "params": asdict(p), "x_m": x, "y_m": y, "q_y_W_m2": q_y,
        "q_peak_W_m2": float(q_peak), "P_incident_W": float(P_incident),
        "P_absorbed_W": float(P_incident*p.absorptivity),
        "dt_s": float(dt), "dt_max_s": dt_max, "nsteps": nsteps,
        "T_final_K": Tfinal, "snapshot_T_K": snaps, "snapshot_times_s": snap_times,
        "ignition_time_s": ignition_time, "ignition_K": float(p.ignition_C+273.15),
        "history_times_s": hist_times, "history_surface_max_K": hist_surface_max,
        "history_depth_center_K": hist_depth,
        "numba": NUMBA_AVAILABLE,
    }
