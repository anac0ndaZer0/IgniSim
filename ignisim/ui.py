from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from .solver import SimulationParams, simulate, quality_preset

BRAND_CSS = r"""
:root { --igni-orange:#ff6b2c; --igni-amber:#ffb347; --panel:#171a21; }
.gradio-container { max-width: 1480px !important; margin: auto !important; }
.igni-hero { padding: 18px 22px; border: 1px solid rgba(255,255,255,.08); border-radius:18px;
  background: linear-gradient(135deg, rgba(255,107,44,.18), rgba(255,179,71,.04) 42%, rgba(255,255,255,.02)); }
.igni-brand { font-size: 34px; font-weight: 800; letter-spacing:-1.2px; margin:0; }
.igni-brand span { color: var(--igni-orange); }
.igni-sub { opacity:.76; margin-top:4px; }
.igni-card { border-radius:16px !important; }
#run-button { background: linear-gradient(90deg,#ff6b2c,#ff934d) !important; color:white !important; border:none !important; }
footer { display:none !important; }
"""


def _heatmap(res):
    T = res["T_final_K"] - 273.15
    x = res["y_m"] * 1e3
    zdepth = res["x_m"] * 1e3
    fig = go.Figure(go.Heatmap(x=x, y=zdepth, z=T, colorscale="Inferno", colorbar=dict(title="°C")))
    fig.update_yaxes(autorange="reversed", title="Depth (mm)")
    fig.update_xaxes(title="Surface position (mm)")
    fig.update_layout(title="Final 2D temperature field", margin=dict(l=55,r=25,t=55,b=50), height=470)
    return fig


def _surface_plot(res):
    fig = go.Figure()
    y = res["y_m"] * 1e3
    for T, t in zip(res["snapshot_T_K"], res["snapshot_times_s"]):
        fig.add_trace(go.Scatter(x=y, y=T[0]-273.15, mode="lines", name=f"{t:.1f} s"))
    fig.add_hline(y=res["ignition_K"]-273.15, line_dash="dash", annotation_text="Ignition threshold")
    fig.update_layout(title="Surface temperature evolution", xaxis_title="Surface position (mm)", yaxis_title="Temperature (°C)", height=380, margin=dict(l=55,r=25,t=55,b=50))
    return fig


def _depth_plot(res):
    times = res["history_times_s"]
    x = res["x_m"] * 1e3
    hist = res["history_depth_center_K"] - 273.15
    fig = go.Figure()
    if len(times):
        picks = np.unique(np.linspace(0, len(times)-1, min(6,len(times))).astype(int))
        for k in picks:
            fig.add_trace(go.Scatter(x=x, y=hist[k], mode="lines", name=f"{times[k]:.1f} s"))
    fig.update_layout(title="Through-thickness profile at centerline", xaxis_title="Depth (mm)", yaxis_title="Temperature (°C)", height=380, margin=dict(l=55,r=25,t=55,b=50))
    return fig


def _history_plot(res):
    fig = go.Figure(go.Scatter(x=res["history_times_s"], y=res["history_surface_max_K"]-273.15, mode="lines", name="Max surface T"))
    fig.add_hline(y=res["ignition_K"]-273.15, line_dash="dash", annotation_text="Ignition threshold")
    fig.update_layout(title="Maximum surface temperature vs time", xaxis_title="Time (s)", yaxis_title="Temperature (°C)", height=380, margin=dict(l=55,r=25,t=55,b=50))
    return fig


def _status_cards(res):
    ign = res["ignition_time_s"]
    ign_text = "Not reached" if ign is None else f"{ign:.2f} s"
    maxC = float(np.max(res["T_final_K"])-273.15)
    return (
        f"### Simulation complete\n"
        f"**Peak temperature:** {maxC:.1f} °C  \n"
        f"**Ignition threshold:** {ign_text}  \n"
        f"**Peak incident flux:** {res['q_peak_W_m2']:,.0f} W/m²  \n"
        f"**Incident / absorbed power:** {res['P_incident_W']:.4g} / {res['P_absorbed_W']:.4g} W  \n"
        f"**Solver:** {res['nsteps']:,} steps · Δt {res['dt_s']:.3e} s"
    )


def _export(res):
    outdir = Path(tempfile.mkdtemp(prefix="ignisim_"))
    final = pd.DataFrame(res["T_final_K"]-273.15, index=res["x_m"], columns=res["y_m"])
    final.index.name = "depth_m"
    final.to_csv(outdir/"final_temperature_C.csv")
    surface = pd.DataFrame({"y_m":res["y_m"], "incident_flux_W_m2":res["q_y_W_m2"]})
    for T,t in zip(res["snapshot_T_K"],res["snapshot_times_s"]): surface[f"surface_T_C_{t:.3f}s"] = T[0]-273.15
    surface.to_csv(outdir/"surface_profiles.csv", index=False)
    hist = pd.DataFrame({"time_s":res["history_times_s"], "max_surface_T_C":res["history_surface_max_K"]-273.15})
    hist.to_csv(outdir/"surface_history.csv", index=False)
    clean_params = dict(res["params"])
    summary = {"parameters":clean_params,"derived":{"dt_s":res["dt_s"],"dt_max_s":res["dt_max_s"],"nsteps":res["nsteps"],"q_peak_W_m2":res["q_peak_W_m2"],"P_incident_W":res["P_incident_W"],"P_absorbed_W":res["P_absorbed_W"],"ignition_time_s":res["ignition_time_s"]}}
    (outdir/"run_summary.json").write_text(json.dumps(summary,indent=2))
    import zipfile
    zpath = outdir/"IgniSim_results.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for f in outdir.iterdir():
            if f != zpath: z.write(f,arcname=f.name)
    return str(zpath)


def run_sim(
    heating_mode, quality, source_profile, laser_input, laser_power, incident_flux, spot_width_mm, spot_center_mm, oop_mm,
    l1_mm,l2_mm,width_mm,
    kx1,ky1,rho1,cp1,kx2,ky2,rho2,cp2,
    Ti_C,Tinf_C,Tsur_C,absorptivity,hfront,emissivity_model,emissivity,eps0,eps1,epsTc_C,epsdT,
    back_bc,ignition_C,duration_s,Nx_adv,Ny_adv,dt_adv,max_steps_adv
):
    preset=quality_preset(quality)
    if quality == "Custom":
        Nx=int(Nx_adv); Ny=int(Ny_adv); max_steps=int(max_steps_adv)
    else:
        Nx=preset["Nx"]; Ny=preset["Ny"]; max_steps=preset["max_steps"]
    p=SimulationParams(
        heating_mode=heating_mode, source_profile=source_profile, laser_input=laser_input,
        laser_power_W=float(laser_power), incident_flux_W_m2=float(incident_flux), spot_width_m=float(spot_width_mm)/1000,
        spot_center_m=float(spot_center_mm)/1000, out_of_plane_m=float(oop_mm)/1000,
        layer1_thickness_m=float(l1_mm)/1000, layer2_thickness_m=float(l2_mm)/1000, sample_width_m=float(width_mm)/1000,
        kx1=float(kx1),ky1=float(ky1),rho1=float(rho1),cp1=float(cp1),kx2=float(kx2),ky2=float(ky2),rho2=float(rho2),cp2=float(cp2),
        T_initial_K=float(Ti_C)+273.15,T_ambient_K=float(Tinf_C)+273.15,T_surroundings_K=float(Tsur_C)+273.15,
        absorptivity=float(absorptivity),h_front_W_m2K=float(hfront),emissivity_model=emissivity_model,emissivity=float(emissivity),
        eps0=float(eps0),eps1=float(eps1),eps_Tc_K=float(epsTc_C)+273.15,eps_dT_K=float(epsdT),back_bc=back_bc,
        ignition_C=float(ignition_C),duration_s=float(duration_s),Nx=Nx,Ny=Ny,dt_s=float(dt_adv),max_steps=max_steps
    )
    try:
        res=simulate(p)
        return _status_cards(res), _heatmap(res), _surface_plot(res), _depth_plot(res), _history_plot(res), _export(res)
    except Exception as e:
        blank=go.Figure(); blank.update_layout(title="Simulation not run")
        return f"### Could not run simulation\n`{e}`", blank, blank, blank, blank, None


def toggle_mode(mode):
    laser = mode == "Laser"
    return gr.update(visible=laser), gr.update(visible=not laser)


def toggle_quality(q):
    return gr.update(visible=q=="Custom")


def build_app():
    with gr.Blocks(title="IgniSim — 2D Heat Transfer Simulator") as demo:
        gr.HTML("""<div class='igni-hero'><div class='igni-brand'><span>Igni</span>Sim</div><div class='igni-sub'>2D transient heat-transfer simulation for localized laser and uniform surface heating</div></div>""")
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes="igni-card"):
                    heating_mode=gr.Radio(["Laser","Uniform"], value="Laser", label="Heating mode")
                    quality=gr.Dropdown(["Preview","Balanced","High fidelity","Custom"], value="Balanced", label="Simulation quality")
                with gr.Tabs():
                    with gr.Tab("Heat source"):
                        with gr.Group(visible=True) as laser_group:
                            source_profile=gr.Radio(["Top-hat","Gaussian"], value="Top-hat", label="Laser spatial profile")
                            laser_input=gr.Radio(["Power","Flux"], value="Power", label="Specify laser by")
                            with gr.Row():
                                laser_power=gr.Number(0.167,label="Laser power (W)")
                                incident_flux=gr.Number(20000,label="Incident flux (W/m²)")
                            with gr.Row():
                                spot_width=gr.Number(1.0,label="Spot width / FWHM (mm)")
                                spot_center=gr.Number(25.0,label="Spot center (mm)")
                                oop=gr.Number(10.0,label="Out-of-plane width (mm)")
                        with gr.Group(visible=False) as uniform_group:
                            gr.Markdown("Uniform mode applies one incident heat flux across the full exposed width.")
                            uniform_flux=gr.Number(20000,label="Uniform incident flux (W/m²)")
                    with gr.Tab("Geometry"):
                        with gr.Row():
                            l1=gr.Number(0.03,label="Layer 1 thickness (mm)")
                            l2=gr.Number(5.0,label="Layer 2 thickness (mm)")
                            width=gr.Number(50.0,label="Sample width (mm)")
                    with gr.Tab("Materials"):
                        gr.Markdown("#### Layer 1")
                        with gr.Row():
                            kx1=gr.Number(2.0,label="kₓ (W/m·K)"); ky1=gr.Number(390.0,label="kᵧ (W/m·K)")
                            rho1=gr.Number(2000,label="Density (kg/m³)"); cp1=gr.Number(800,label="Heat capacity (J/kg·K)")
                        gr.Markdown("#### Layer 2")
                        with gr.Row():
                            kx2=gr.Number(0.55,label="kₓ (W/m·K)"); ky2=gr.Number(0.55,label="kᵧ (W/m·K)")
                            rho2=gr.Number(1300,label="Density (kg/m³)"); cp2=gr.Number(1700,label="Heat capacity (J/kg·K)")
                    with gr.Tab("Boundary & losses"):
                        with gr.Row():
                            Ti=gr.Number(24.85,label="Initial T (°C)"); Tinf=gr.Number(24.85,label="Ambient T (°C)"); Tsur=gr.Number(24.85,label="Radiative surroundings (°C)")
                        with gr.Row():
                            absorptivity=gr.Slider(0,1,value=1.0,step=.01,label="Absorptivity")
                            hfront=gr.Number(15,label="Front convection h (W/m²·K)")
                            backbc=gr.Radio(["Adiabatic","Fixed"],value="Adiabatic",label="Back boundary")
                        emissivity_model=gr.Radio(["Constant","Temperature-dependent"],value="Constant",label="Emissivity model")
                        with gr.Row():
                            emissivity=gr.Slider(0,1,value=.9,step=.01,label="Constant ε")
                            eps0=gr.Slider(0,1,value=.9,step=.01,label="ε low-T")
                            eps1=gr.Slider(0,1,value=.9,step=.01,label="ε high-T")
                        with gr.Row():
                            epsTc=gr.Number(350,label="Transition center (°C)")
                            epsdT=gr.Number(25,label="Transition width (K)")
                    with gr.Tab("Simulation"):
                        with gr.Row():
                            duration=gr.Number(300,label="Duration (s)")
                            ignition=gr.Number(381,label="Ignition threshold (°C)")
                        with gr.Group(visible=False) as custom_group:
                            gr.Markdown("Custom resolution uses the original explicit solver stability criterion. Set Δt = 0 for automatic stable timestep.")
                            with gr.Row():
                                Nx=gr.Number(90,label="Nx",precision=0); Ny=gr.Number(60,label="Ny",precision=0)
                                dt=gr.Number(0,label="Δt (s, 0 = Auto)")
                                maxsteps=gr.Number(2000000,label="Max step budget",precision=0)
                run=gr.Button("Run simulation",variant="primary",elem_id="run-button",size="lg")
            with gr.Column(scale=7):
                status=gr.Markdown("### Ready\nConfigure the model and run a simulation.")
                heatmap=gr.Plot(label="Temperature field")
                with gr.Tabs():
                    with gr.Tab("Surface"): surface_plot=gr.Plot()
                    with gr.Tab("Depth"): depth_plot=gr.Plot()
                    with gr.Tab("Time history"): history_plot=gr.Plot()
                download=gr.File(label="Download run data (.zip)")

        heating_mode.change(toggle_mode, inputs=heating_mode, outputs=[laser_group,uniform_group])
        quality.change(toggle_quality, inputs=quality, outputs=custom_group)

        def dispatch(mode, quality_, source_profile_, laser_input_, laser_power_, laser_flux_, uniform_flux_, *rest):
            flux = uniform_flux_ if mode == "Uniform" else laser_flux_
            return run_sim(mode,quality_,source_profile_,laser_input_,laser_power_,flux,*rest)

        inputs=[heating_mode,quality,source_profile,laser_input,laser_power,incident_flux,uniform_flux,spot_width,spot_center,oop,l1,l2,width,
                kx1,ky1,rho1,cp1,kx2,ky2,rho2,cp2,Ti,Tinf,Tsur,absorptivity,hfront,emissivity_model,emissivity,eps0,eps1,epsTc,epsdT,backbc,ignition,duration,Nx,Ny,dt,maxsteps]
        run.click(dispatch,inputs=inputs,outputs=[status,heatmap,surface_plot,depth_plot,history_plot,download])
        gr.Markdown("*IgniSim is an engineering simulation interface, not a certified fire-safety or product-compliance tool. Validate material properties, boundary conditions, and convergence for consequential use.*")
    return demo
