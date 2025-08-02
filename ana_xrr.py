import copy
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import refnx  # https://refnx.readthedocs.io/en/latest/installation.html
import seaborn as sns
from refnx.analysis import CurveFitter, Objective, Transform  # type: ignore
from refnx.dataset import ReflectDataset  # type: ignore
from refnx.reflect import SLD, ReflectModel, Structure  # type: ignore
from scipy import signal


cfg_model = {"bkg": 3e-8, "dq": 1, "scale": 1, "q_offset": 0}
cfg_sld = {
    "al": 22.425,
    "nb": 65.035,
    "ta": 105.455,
    "alox": 28.419,
    "nbox": 36.204,
    "hiox": 50.2044,
    'taox': 55.2044,
    'tasiox':80,
    "al2o3": 33.38,
    "siox": 22.723,
    "si": 20.071,
    "ta2o5": 59.37,
    "buff": 18,
    "buff_low": 4.5,
    "buff_med": 10,
}
cfg_isld = {
    "al": -0.411,
    "nb": -3.928,
    "ta": -8.5,
    "ta2o5": -3.542,
    "si": -0.458,
    "siox": -0.294,
    "alox": -0.37,
    "nbox": -0.3,
    "al2o3": -0.388,
    "taox": -3.5,
    "tasiox": -4,
    "buff": -0.2,
    "air": 0,
    "buff_low": -0.2,
    "buff_med": -0.2,
    "hiox": -0.3,
}
# cfg_isld = {'al':0, 'nb':0, 'alox':0, 'nbox':0,'al2o3':0,'buff':0,'air':0}
cfg_rough = {
    "al": 1,
    "nb": 1,
    "ta": 1,
    "ta2o5": 1,
    "si": 1,
    "siox": 1,
    "alox": 2,
    "nbox": 1,
    "taox": 2,
    "tasiox":8,
    "al2o3": 5,
    "buff": 4,
    "buff_low": 4,
    "buff_med": 4,
    "hiox": 1,
}
cfg_rough_max = {
    "al": 10,
    "nb": 10,
    "ta": 10,
    "alox": 10,
    "nbox": 10,
    "taox": 20,
    "tasiox": 25,
    'siox':30,
    "al2o3": 10,
    "buff": 10,
    "buff_low": 10,
    "buff_med": 10,
}
cfg_thk = {
    "al": 200,
    "nb": 500,
    "ta": 500,
    "tasiox":20,
    'taox':50,
    "alox": 7,
    "nbox": 10,
    'siox':10,
    "buff": 10,
    "al2o3": np.inf,
    'si': np.inf,
    "buff_low": 10,
    "buff_med": 10,
    "hiox": 10,
}
cfg_max_thk = {
    "al": 500,
    "nb": 500,
    "ta": 500,
    "tasiox": 100,
    "alox": 50,
    "nbox": 50,
    "buff": 50,
    "al2o3": np.inf,
    "buff_low": 50,
    "buff_med": 50,
    "hiox": 50,
}
cfg_vary = {"bkg": True, "dq": False, "scale": True, "q_offset": True}


def save_fig(fig, path, name=""):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + "/" + name + "_" + timestr + ".png"
    fig.savefig(path_out)
    return str(path_out)


def save_text(string, path, name=""):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + name + "_" + timestr + ".txt"
    text_file = open(path_out, "w")
    text_file.write(string)
    text_file.close()
    return str(path_out)

# Load data
def loadRefData(path, x_start=None, x_end=None, qrng=None, coef=1.9, x_err=None):
    data = pd.read_csv(path)  # load text
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = len(data)
    x = data['tth'][x_start:x_end]
    if x_err is not None: 
        x_err = x_err[x_start:x_end]
    y = data['Normalized'][x_start:x_end].map(lambda x : x / data['Normalized'].max())
    # conv t-th to q using 0.72769 A wavelength
    x_new = x.map(lambda x: refnx.util.q(x / 2, 0.72769))
    #4 * np.pi * np.sin(np.radians(x/2)) / wavelength
    compiled = (x_new, y)

    if qrng is not None:
        y = y[(x_new > qrng[0]) & (x_new < qrng[1])]
        if x_err is not None:
            x_err = x_err[(x_new > qrng[0]) & (x_new < qrng[1])]
        x_new = x_new[(x_new > qrng[0]) & (x_new < qrng[1])]

    if x_err is not None:
        y_err = y
        x_err = x_err / 100 * x_new * coef
        compiled = (x_new, y, y_err, x_err)
    else:
        compiled = (x_new, y)

    # compiled = (x_new,y, y_err)
    # returns the ReflectDataset object
    # print (f"Model will fit:\nd_min: {min(x_new)} A\n"
    #   f"d_max: {max(x_new)} A\n")

    return ReflectDataset(compiled)

def standalone_xerr(x, factor=80):
    # Calculate theta in radians
    wavelength = 0.72769
    theta_radians = 2 * np.arcsin((x * wavelength) / (4 * np.pi))
    # Convert to degrees
    cot = 1 / np.tan(theta_radians)
    x_err = cot/factor
    print ("Error at" ,(theta_radians[400] *180/np.pi) ," is" , x_err[400] , "%")

    print ("Error at" ,(theta_radians[-1] *180/np.pi) ," is" , x_err[-1] , "%")
    
    return x_err

# Print out data about the model
def generate_table(data):
    # Create a DataFrame from the provided data
    df = pd.DataFrame(
        {
            "Layer": data["layers"],
            "Thickness (Å)": data["thk"],
            "Roughness (Å)": data["rough"],
            "SLD (10^-6 cm^-2)": data["sld"],
        }
    )
    # Add additional parameters as a separate DataFrame
    additional_params = pd.DataFrame(
        {
            "Parameter": [
                "Chisq",
                "Log10 Background",
                "DQ - Resolution",
                "Q Offset",
                "Scale",
            ],
            "Value": [
                data["chisq"],
                np.log10(data["bkg"]),
                data["dq - resolution"],
                data["q_offset"],
                data["scale"],
            ],
        }
    )
    # Print the main table
    print("Layer Properties:")
    print(df.to_string(index=False))
    # Print additional parameters
    print("\nAdditional Parameters:")
    print(additional_params.to_string(index=False))

# 
def make_model_par(level, par, use_err=False):
    if "rough" not in par:
        par["rough"] = None
    if "thk" not in par:
        par["thk"] = None
    if "sld" not in par:
        par["sld"] = None
    structure = make_layer_stack(
        level, thick=par["thk"], rough=par["rough"], sld=par["sld"]
    )
    cfg_mod = {
        "scale": par["scale"],
        "bkg": par["bkg"],
        "q_offset": par["q_offset"],
        "dq": par["dq - resolution"],
    }
    model = make_model(structure, cfg_mod, use_err)
    return structure, model

# Takes in list of layers and info about properties, creates structure
def make_layer_stack(level, thick=None, rough=None, sld=None):
    layer_list = level.split("/")

    layer_list_names = [f"{layer}_{i}" for i, layer in enumerate(layer_list)]
    layers = []
    if thick is None:
        thick = [None] * len(layer_list)
    if rough is None:
        rough = [None] * len(layer_list)
    if sld is None:
        sld = [None] * len(layer_list)

    layers.append(make_layer("air", thick=np.inf, rough=0, sld=0))
    for i, (layer, name, t, r, s) in enumerate(
        zip(layer_list, layer_list_names, thick, rough, sld)
    ):
        layers.append(make_layer(name, thick=t, rough=r, sld=s, material=layer))

    structure = Structure(layers)
    return structure

# Fills in structural info
def make_layer(name, sld=None, thick=None, rough=None, material=None):
    # Name is unique name for this layer. If not unique, SLD will be linked to ohter layers with same name.
    if sld is None:
        sld = cfg_sld[material]
    if thick is None:
        thick = cfg_thk[material]
    if rough is None:
        rough = cfg_rough[material]
    if material is None:
        isld = 0
    else:
        isld = cfg_isld[material]
    l = SLD(sld + 1j * isld, name=name)
    return l(thick, rough)

# This provides model specfic components
def make_model(structure, cfg=cfg_model, use_err=False):
    model = ReflectModel(
        structure,
        bkg=cfg["bkg"],
        dq=cfg["dq"],
        scale=cfg["scale"],
        q_offset=cfg["q_offset"],
    )
    model.bkg.setp(bounds=(5e-10, 5e-8), vary=cfg_vary["bkg"])  # background
    if use_err: 
        cfg_vary['dq']=False
    else:
        cfg_vary['dq']=True
    model.dq.setp(bounds=(0.001, 2.2), vary=cfg_vary["dq"])  # due to error?
    model.scale.setp(bounds=(0.4, 1.2), vary=cfg_vary["scale"])  # maximum value
    model.q_offset.setp(
        bounds=(-0.02, 0.02), vary=cfg_vary["q_offset"]
    )  # due to error?
    return model

# Needs to be updated, meant to repeat scan n times; probably easier to just use for loops
def repeat_scans(
    file,
    level="alox/al",
    qrng=[0.025, 0.3],
    save_name="test",
    cfg_in={},
    cfg_in2={},
    cfg_sld_in={},
    run_mc=False,
    SLD_rng=[1, 0.3],
    plot=False,
    no_fit=False,
    careful=False,
    n=5,
    start_new=False,
):
    chi = []
    chi_start = []
    chi_min = 1e3
    for i in range(n):
        if start_new or i == 0:
            obj, par = sim_al(
                file,
                level,
                qrng,
                save_name,
                cfg_in,
                cfg_in2,
                cfg_sld_in,
                run_mc,
                SLD_rng,
                plot,
                no_fit,
                careful,
            )
        else:
            obj, par = refit(copy.deepcopy(obj), file, qrng, plot)
            chi_start.append(par["start_chi"])
        chi.append(par["chisq"])
        if par["chisq"] < chi_min:
            chi_min = par["chisq"]
            obj_min = obj
            par_min = par
    return obj_min, par_min, chi, chi_start

# Check if layer is metal, oxide or buffer
def check_cat(level, thk):
    sld = []
    full_thk = []

    layer_list = level.split("/")
    ox_indices = [i + 1 for i, layer in enumerate(layer_list) if "ox" in layer]
    metal_indices = [
        i + 1 for i, layer in enumerate(layer_list) if "al" == layer or "nb" == layer or "ta" == layer
    ]
    buff_indices = [i + 1 for i, layer in enumerate(layer_list) if "buff" in layer]
    cat = {"ox": ox_indices, "metal": metal_indices, "buff": buff_indices}
    for m in metal_indices:
        sld.append(cfg_sld[layer_list[m - 1]])
    if len(thk) == len(layer_list):
        full_thk = thk
    else:
        mm = [m - 1 for m in metal_indices]
        j = 0
        for i, layer in enumerate(layer_list):
            if i in mm:
                full_thk.append(thk[j])
                j += 1
            else:
                full_thk.append(cfg_thk[layer])

    return cat, sld, full_thk

# Main function for setting up model and running simulation
def sim(
    file,
    level,
    params,
    qrng=[0.025, 0.3],
    plot=True,
    SLD_rng=[0.6, 0.15],
    careful=False,
    fit=True,
    multi=None,
    plot_init=False,
    img_path=None,
    run_mc=False,
    use_err=False,
):
    # possible layer names are al, nb, nbox, alox, buff
    fname = level.replace("/", "-")
    fsimp = os.path.splitext(os.path.basename(file))[0]
    qm = qrng[1] * 100
    fname = fsimp + "_" + fname + f"_{qm:.0f}"
    cats, slds, thk = check_cat(level, params["thk"])
    params["thk"] = thk
    structure, model = make_model_par(level, params, use_err=use_err)

    if use_err: 
        data = loadRefData(file)
        x_err = standalone_xerr(data.x)
        data =loadRefData(file, qrng=qrng, coef=params['x_err'], x_err=x_err)
    else:
        data = loadRefData(file, qrng=qrng)
        
    objective = Objective(model, data, transform=Transform("logY"))
    if fit is False and multi is None:
        multi = False
    else:
        multi = True
    if img_path is None:
        save = False
    else:
        save = True
    if plot and not multi:
        fig, a = plt.subplots()
        a.set_xlabel("q (1/Å)")
        a.set_ylabel("Normalized Counts")
        if plot_init:
            a.loglog(data.x, objective.model(data.x), "k")
        a.loglog(data.x, data.y, ".-", color="#b51d14", linewidth=1, markersize=4)
        if save:
            save_fig(fig, img_path, fname)
        # print(objective)

    # Defining the bounds, this can definitely be improved.
    j = 0
    for i, s in enumerate(structure):
        if i in cats["ox"]:
            s.sld.real.setp(
                bounds=(
                    (1 - SLD_rng[0] / 1.5) * s.sld.real.value,
                    (1 + SLD_rng[0]) * s.sld.real.value,
                ),
                vary=True,
            )
            if careful:
                s.thick.setp(bounds=(s.thick.value / 2, 1.5 * s.thick.value), vary=True)
                s.rough.setp(bounds=(s.rough.value / 2, 1.5 * s.rough.value), vary=True)
            else:
                s.thick.setp(bounds=(0, 100), vary=True)
                s.rough.setp(bounds=(0, 25), vary=True)
        elif i in cats["metal"]:
            if careful:
                s.thick.setp(
                    bounds=(0.9 * s.thick.value, 1.1 * s.thick.value), vary=True
                )
                s.rough.setp(bounds=(s.rough.value / 2, 1.5 * s.rough.value), vary=True)
                s.sld.real.setp(
                    bounds=((1 - SLD_rng[1]) * slds[j], (1 + SLD_rng[1]) * slds[j]),
                    vary=True,
                )
                j += 1
            else:
                # s.thick.setp(bounds = (s.thick.value/2,s.thick.value*2), vary = True)
                s.thick.setp(
                    bounds=(0.83 * s.thick.value, 1.17 * s.thick.value), vary=True
                )
                s.sld.real.setp(
                    bounds=(
                        (1 - SLD_rng[1]) * s.sld.real.value,
                        (1 + SLD_rng[1]) * s.sld.real.value,
                    ),
                    vary=True,
                )
                s.rough.setp(bounds=(0, 12), vary=True)
        elif i in cats["buff"]:
            s.sld.real.setp(
                bounds=(
                    (1 - SLD_rng[0]) * s.sld.real.value,
                    (1 + SLD_rng[0]) * s.sld.real.value,
                ),
                vary=True,
            )
            if careful:
                s.thick.setp(
                    bounds=(0.7 * s.thick.value, 1.3 * s.thick.value), vary=True
                )
                s.rough.setp(bounds=(s.rough.value / 2, 1.5 * s.rough.value), vary=True)
            else:
                s.thick.setp(bounds=(0, 55), vary=True)
                s.rough.setp(bounds=(0, 15), vary=True)

        rough = params["sub_rough_max"]
        structure[-1].rough.setp(bounds=(0, rough), vary=True)
    # print(objective.chisqr())
    if fit:
        fitter = CurveFitter(objective)
        fitter.fit(
            "differential_evolution", maxiter=1500
        )  # , tol=0.005, mutation=(0.5,1.2))

        if plot:
            plot_obj([objective], [""], path=img_path, fname=fname, save=save)
            if save:
                save_text(str(objective), img_path, fname)
            # plt.semilogy(data.x, objective.model(data.x))

            # fig2 = plt.figure()
            # plt.plot(data.x, objective.residuals())

            # plt.figure()
            # plt.plot(*structure.sld_profile())
            # plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
            # plt.xlabel("distance / $\\AA$")
        if run_mc:
            mc_ana(objective, fitter)

        params = process_objective(objective)
        params["sub_rough_max"] = rough

    return objective, params

# Perform monte carlo analysis
def mc_ana(objective, fitter):

    fitter.sample(400, pool=1)
    fitter.reset()
    res = fitter.sample(30, nthin=20, pool=1)
    objective.corner()
    print(objective)
    fig, ax = objective.model.structure.plot(samples=100)

# Take in objective and refit; not sure it's that useful with this analysis type
def refit(
    objective,
    file,
    qrng=[0.025, 0.3],
    plot=True,
    reset=False,
    SLD_rng=[0.2, 0.2],
    fit=True,
    save=True,
    img_path=None,
    fname="",
    use_err=False,
):
    data = loadRefData(
        file, qrng=qrng, use_err=use_err
    )  # this function loads to the data and converts the x-data from 2-theta to q
    objective.data = data
    if plot and not fit:
        fig1 = plt.figure()
        plt.xlabel("q (1/A)")
        plt.ylabel("logR")
        plt.loglog(data.x, data.y, ".-", color="#b51d14", linewidth=1, markersize=4)
        plt.semilogy(data.x, objective.model(data.x))
    # have it reset the bounds
    if reset:
        for s in objective.model.structure:
            s.sld.real.bounds.lb = (1 - SLD_rng) * s.sld.real.value
            s.sld.real.bounds.ub = (1 + SLD_rng) * s.sld.real.value
            s.thick.bounds.lb = (1 - SLD_rng) * s.thick.value
            s.thick.bounds.ub = (1 + SLD_rng) * s.thick.value
            s.rough.bounds.lb = (1 - SLD_rng) * s.rough.value
            s.rough.bounds.ub = (1 + SLD_rng) * s.rough.value
    if not fit:
        return objective, qrng
    fitter = CurveFitter(objective)
    start_chi = objective.chisqr()
    fitter.fit("differential_evolution")
    params = process_objective(objective)
    params["start_chi"] = start_chi

    if plot:
        plot_obj([objective], [""], path=img_path, fname=fname, save=save)

    return objective, params

# Save model parameters to a readable dict
def process_objective(obj):
    thk = []
    rough = []
    sld = []
    lays = []
    for i in range(1, len(obj.parameters[1])):
        check_rng(obj.parameters[1][i][0])
        check_rng(obj.parameters[1][i][2])
        check_rng(obj.parameters[1][i][1][0])
        lays.append(obj.parameters[1][i].name)
        rough.append(obj.parameters[1][i][2].value)
        thk.append(obj.parameters[1][i][0].value)
        sld.append(obj.parameters[1][i][1][0].value)

    params = {"thk": thk, "rough": rough, "sld": sld, "layers": lays}

    for item in params:
        params[item] = np.array(params[item])
    # Add the model params to the dict.
    for i in range(len(obj.parameters[0])):
        params[obj.parameters[0][i].name] = obj.parameters[0][i].value
    params["chisq"] = obj.chisqr()
    return params

# Compare model outputs by comparing names
def compare_models(pars):
    pp = {
        "sld": {},
        "thick": {},
        "rough": {},
        "chisq": [],
        "dq - resolution": [],
        "q_offset": [],
        "scale": [],
        "bkg": [],
    }
    layer_list = pars[-1]["layers"]

    for l in layer_list:
        indices = [i for i, layer in enumerate(layer_list) if layer == l]
        for i, index in enumerate(indices):
            if i > 0:
                l = l + str(i)
            pp["rough"][l] = []
            pp["thick"][l] = []
            pp["sld"][l] = []

    for p in pars:
        pp["chisq"].append(p["chisq"])
        pp["dq - resolution"].append(p["dq - resolution"])
        pp["q_offset"].append(p["q_offset"])
        pp["scale"].append(p["scale"])
        pp["bkg"].append(p["bkg"])

        for l in set(layer_list):
            if l in p["layers"]:
                # index = p['layers'].index(l)
                indices = [i for i, layer in enumerate(p["layers"]) if layer == l]
                for i, index in enumerate(indices):
                    if i > 0:
                        l = l + str(i)
                    pp["rough"][l].append(p["rough"][index])
                    pp["thick"][l].append(p["thk"][index])
                    pp["sld"][l].append(p["sld"][index])
    return pp

# Print warning if we're at the edge of the bounds
def check_rng(par):
    if par.value > 0.95 * par.bounds.ub:
        print(f"Warning: {par.name} is at the upper edge of its bounds")
    if par.value < 1.05 * par.bounds.lb:
        print(f"Warning: {par.name} is at the lower edge of its bounds")

# If you have list with structure [[obj1, pars1], [obj2, pars2], ...], returns list of just pars
def collect_pars(out):
    full_pars = [None] * len(out)
    for i, o in enumerate(out):
        full_pars[i] = [oo[1] for oo in o]
    return full_pars


def collect_obj(out):
    full_obj = [None] * len(out)
    for i, o in enumerate(out):
        full_obj[i] = [oo[0] for oo in o]
    return full_obj

def rearrange_results(results, dims):
    
    results_3d = [[[None for _ in range(dims[2])] for _ in range(dims[1])] for _ in range(dims[0])]

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                results_3d[i][j][k] = results[i * dims[1]*dims[2] + j * dims[2] + k]

    results_3d_swapped = [[[results_3d[i][k][j] for k in range(len(results_3d[i]))] for j in range(len(results_3d[i][0]))] for i in range(len(results_3d))]

    full_obj = [[[None for _ in range(len(results_3d_swapped[0][0]))] for _ in range(len(results_3d_swapped[0]))] for _ in range(len(results_3d_swapped))]
    full_par = [[[None for _ in range(len(results_3d_swapped[0][0]))] for _ in range(len(results_3d_swapped[0]))] for _ in range(len(results_3d_swapped))]

    for i in range(len(results_3d_swapped)):
        for j in range(len(results_3d_swapped[i])):
            for k in range(len(results_3d_swapped[i][j])):
                full_obj[i][j][k] = results_3d_swapped[i][j][k][0]
                full_par[i][j][k] = results_3d_swapped[i][j][k][1]
    
    return results_3d_swapped, full_obj, full_par

# Plot model for one or more objectives
def plot_obj(
    obj, labs=None, multi=True, save=True, path=None, fname="test", ind=0, psd=True
):

    if multi:
        # fig, ax = plt.subplots(2,2, figsize=(12,6))
        # ax=ax.flatten()
        fig, ax = plt.subplot_mosaic(
            [["a", "a"], ["b", "c"]], figsize=(8, 7), constrained_layout=True
        )

        ax["a"].set_xlabel("q (1/Å)")
        ax["a"].set_ylabel("Normalized Counts")

        ax["c"].set_ylabel("SLD /$10^{-6} \\AA^{-2}$")
        ax["c"].set_xlabel("Depth ($\\AA$)")
        if len(obj) == 1:
            ax["a"].loglog(
                obj[0].data.x,
                obj[0].data.y,
                ".-",
                markersize=4,
                label="Data",
                color="#b51d14",
            )
        else:
            ax["a"].loglog(
                obj[ind].data.x,
                obj[ind].data.y,
                ".-",
                markersize=4,
                label="Data",
                color="k",
            )
        for l, objective in zip(labs, obj):
            ax["a"].semilogy(
                objective.data.x, objective.model(objective.data.x), label=l
            )

            ax["c"].plot(*objective.model.structure.sld_profile(), label=l)
            if psd:
                fs = 1 / (
                    (max(objective.data.x) - min(objective.data.x))
                    / len(objective.data.x)
                )
                #f, Pxx_den = signal.periodogram(objective.residuals(), fs)
                f, Pxx_den = signal.periodogram(np.log(objective.data.y), fs)
                f = f[1:]
                Pxx_den = Pxx_den[1:]
                x_max = 250
                inds = 5.3 * f < x_max

                ax["b"].semilogy(5.3 * f / 10, Pxx_den, ".-")
            else:
                ax["b"].plot(objective.data.x, objective.residuals(), label=l)

        ax["a"].legend(frameon=False)
        if not psd:
            ax["b"].set_xlabel("q (1/Å)")
            ax["b"].set_ylabel("Residuals")
            ax["b"].axhline(y=0, color="black", linestyle="--")
        else:
            ax["b"].set_xlabel("Thickness (nm)")
            ax["b"].set_ylabel("PSD")

        # if len(labs[0])>0:
        #    ax['b'].legend()
        #    ax['c'].legend()
        fig.suptitle(fname)
    else:
        fig = plt.figure()
        plt.xlabel("q (1/Å)")
        plt.ylabel("Normalized Counts")
        # plt.loglog(data.x, model(data.x), linewidth=0.5, label='guess')
        plt.loglog(
            obj[ind].data.x,
            obj[ind].data.y,
            ".-",
            markersize=4,
            label="Data",
            color="#b51d14",
        )
        for l, objective in zip(labs, obj):
            plt.semilogy(objective.data.x, objective.model(objective.data.x), label=l)

        plt.legend()
        fig2 = plt.figure()
        for l, objective in zip(labs, obj):
            plt.plot(objective.data.x, objective.residuals(), label=l)
        plt.axhline(y=0, color="black", linestyle="--")
        plt.legend()
        plt.figure()
        for l, objective in zip(labs, obj):
            plt.plot(*objective.model.structure.sld_profile(), label=l)
        plt.legend()
        plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
        plt.xlabel("distance / $\\AA$")

    fig.tight_layout()
    if save:
        save_fig(fig, path, fname)
    plt.show()

    # return fig


def plot_obj_pane(obj, labs=None, save=True, path=None, fname="test", ind=0):

    nmod = len(obj)
    nrows = int(np.ceil(nmod / 2))
    fig, ax = plt.subplots(nrows, 2, figsize=(12, nrows * 3), sharex=True)
    ax = ax.flatten()
    # ax['a'].set_xlabel('q (1/Å)')
    # ax['a'].set_ylabel('Normalized Counts')

    for a, l, objective in zip(ax, labs, obj):
        a.loglog(
            obj[ind].data.x,
            obj[ind].data.y,
            ".-",
            markersize=3,
            label=l,
            color="#b51d14",
        )
        a.semilogy(
            objective.data.x,
            objective.model(objective.data.x),
            label=f"$\chi^2=${objective.chisqr():.1f}",
        )
        a.legend(frameon=False)

    fig.suptitle(fname)

    fig.tight_layout()
    if save:
        save_fig(fig, path, fname)
    plt.show()


def plot_vary(var, vals, name, level=1, cfg={}, q=None):
    sns.set_palette("coolwarm", len(vals))
    if q is None:
        q = np.linspace(4e-2, 1, 5000)
    if len(vals) > 4:
        fig, ax = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(10, 6))
    else:
        fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(7, 6))

    ax[1, 0].set_xlabel("q (1/Å)")
    ax[1, 1].set_xlabel("q (1/Å)")
    ax[0, 0].set_ylabel("Normalized Counts")
    ax[1, 0].set_ylabel("Normalized Counts")

    ax = ax.flatten()
    fig2, ax2 = plt.subplots(1, 1)

    for i in range(len(vals)):
        cfg[var] = vals[i]
        model = make_model_nb(level, cfg)
        ax[i].loglog(q, model(q), linewidth=1)
        ax[i].set_title(f"{vals[i]:.1f}")
        ax2.semilogy(q, model(q), linewidth=1, label=f"{vals[i]:.1f}")

    fig.suptitle(name)
    fig2.suptitle(name)
    fig.tight_layout()
    ax2.legend()

    fig.savefig(name + ".png")

    ax2.set_xlabel("q (1/Å)")
    ax2.set_ylabel("Normalized Counts")

    fig2.tight_layout()
    fig2.savefig(name + "2.png")


def plot_model(level, params, q=None, img_path=None, ax=None): 
    fname = level.replace("/", "-")
    cats, slds, thk = check_cat(level, params["thk"])
    params["thk"] = thk
    structure, model = make_model_par(level, params)
    if img_path is None:
        save = False  
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if q is None:
        q = np.linspace(2e-2, 1.4, 5000)
    ax.semilogy(q, model(q), label=level, linewidth=0.5)
    ax.legend()

    # if data is not None:
    #     ax.semilogy(
    #         data.x,
    #         data.y,
    #         ".-",
    #         color="#b51d14",
    #         linewidth=0.5,
    #         markersize=4,
    #         label=name,
    #     )
    return ax


def psd(obj):

    plt.figure()
    fs = 1 / ((max(obj.data.x) - min(obj.data.x)) / len(obj.data.x))
    f, Pxx_den = signal.periodogram(obj.residuals(), fs)
    plt.semilogy(f[1:] * 2 * np.pi, Pxx_den[1:], ".-")
    # plt.ylim([1e-9, 1e0])
    plt.xlabel("length scale [A]")
    plt.ylabel("PSD [$V^2$/Hz]")
    plt.show()


def compare_pars(pars, save=True, img_path=None, fname="test"):

    pp = compare_models(pars)
    nlays = len(pp["sld"].keys())

    fig, ax = plt.subplots(3, nlays, figsize=(nlays * 3, 8), sharex=True)

    ax = ax.flatten()
    for i, layer in enumerate(pp["sld"].keys()):
        ax[i].plot(pp["sld"][layer], ".-", label=layer + " " + "sld")
        ax[i + nlays].plot(pp["rough"][layer], ".-", label=layer + " " + "rough")
        if i < nlays - 1:
            ax[i + 2 * nlays].plot(
                pp["thick"][layer], ".-", label=layer + " " + "thick"
            )
        else:
            ax[i + 2 * nlays].plot(
                np.sum(pp["thick"][layer][0:-2]), ".-", label="total thick"
            )

    for a in ax:
        a.legend()

    fig.tight_layout()

    fig2, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].plot(pp["chisq"], ".-", label="chisq")
    ax[1].semilogy(pp["bkg"], ".-", label="background")
    ax[2].plot(pp["scale"], ".-", label="scale")
    ax[3].plot(pp["q_offset"], ".-", label="q offset")
    for a in ax:
        a.legend()
    fig2.tight_layout()

    if save:
        save_fig(fig, img_path, fname + "_pars")
        save_fig(fig2, img_path, fname)
