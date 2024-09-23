import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy
import refnx #https://refnx.readthedocs.io/en/latest/installation.html
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel, Structure
import time
import mpld3 #you may need to pip install mpld3 to get this working - it allows zoom action on inline plots
import seaborn as sns
import copy
import pandas as pd 

cfg_model = {'bkg': 3e-8, 'dq': 1, 'scale': 1, 'q_offset': 0}
cfg_sld = {'al':22.425, 'nb':64.035, 'alox':28.419, 'nbox':17.819,'al2o3':33.38,'buff':14}
cfg_isld = {'al':-0.411, 'nb':-3.928, 'alox':-0.37, 'nbox':-.3,'al2o3':-0.388,'buff':-0.2,'air':0}
#cfg_isld = {'al':0, 'nb':0, 'alox':0, 'nbox':0,'al2o3':0,'buff':0,'air':0}
cfg_rough = {'al':1, 'nb':1, 'alox':2, 'nbox':1,'al2o3':5,'buff':4}
cfg_thk = {'al':200, 'nb':500, 'alox':10, 'nbox':15,'buff':20}

cfg_vary = {'bkg':True, 'dq':False, 'scale':True, 'q_offset':True, 
                'al_thk':True, 'nb_thk':True, 'top_oxide_thk':True, 'bottom_oxide_thk':True, 
                'al_rough':True,'al_sld':True,'nb_rough':True, 'nb_sld':True, 'top_oxide_rough':True, 'top_oxide_sld':True, 
                'bottom_oxide_rough':True, 'bottom_oxide_sld':True, 'substrate_sld':False, 'substrate_rough':True, 'middle_oxide_thk':True, 'middle_oxide_rough':True, 'middle_oxide_sld':True}

def save_fig(fig, path, name='' ):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + "/" +name+"_" +timestr+ ".png"
    fig.savefig(path_out)
    return str(path_out)

def save_text(string, path, name=''):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + name + "_" + timestr + ".txt"
    text_file = open(path_out, "w")
    text_file.write(string)
    text_file.close()
    return str(path_out)

def loadRefData(path,x_start=None, x_end=None, qrng=None, use_err=False):
    data = np.loadtxt(path) #load text
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = len(data)
    x = data[x_start:x_end,0].T
    y = data[x_start:x_end,1].T/max(data[:,1])
    #conv t-th to q using 0.72769 A wavelength
    x_new = refnx.util.q(x/2, 0.72769)
    x_err = data[x_start:x_end,2].T
    
    if qrng is not None:
        y = y[(x_new>qrng[0]) & (x_new<qrng[1])]
        x_err = x_err[(x_new>qrng[0]) & (x_new<qrng[1])]
        x_new = x_new[(x_new>qrng[0]) & (x_new<qrng[1])]                
    compiled = (x_new,y)
    y_err = y
    x_err=x_err/100*x_new*2
    
    compiled = (x_new,y, y_err, x_err)
    #compiled = (x_new,y, y_err)
    #returns the ReflectDataset object
    #print (f"Model will fit:\nd_min: {min(x_new)} A\n"
    #   f"d_max: {max(x_new)} A\n")

    return ReflectDataset(compiled)

def generate_table(data):
    # Create a DataFrame from the provided data
    df = pd.DataFrame({
        'Layer': data['layers'],
        'Thickness (Å)': data['thk'],
        'Roughness (Å)': data['rough'],
        'SLD (10^-6 cm^-2)': data['sld']
    })
    # Add additional parameters as a separate DataFrame
    additional_params = pd.DataFrame({
        'Parameter': ['Chisq', 'Log10 Background', 'DQ - Resolution', 'Q Offset','Scale'],
        'Value': [data['chisq'], np.log10(data['bkg']), data['dq - resolution'], data['q_offset'], data['scale']]
    })
    # Print the main table
    print("Layer Properties:")
    print(df.to_string(index=False))
    # Print additional parameters
    print("\nAdditional Parameters:")
    print(additional_params.to_string(index=False))

def make_model_par(level, par):
    if 'rough' not in par:
        par['rough'] = None
    if 'thk' not in par:
        par['thk'] = None
    if 'sld' not in par:
        par['sld'] = None
    structure = make_layer_stack(level, thick=par['thk'], rough=par['rough'], sld=par['sld'])
    cfg_mod = {'scale':par['scale'], 'bkg':par['bkg'], 'q_offset':par['q_offset'], 'dq':par['dq - resolution']}
    model = make_model(structure, cfg_mod)
    return structure, model

def make_layer_stack(level, thick=None, rough=None, sld=None): 
    layer_list = level.split('/')

    layer_list_names = [f"{layer}_{i}" for i, layer in enumerate(layer_list)]
    layers =[]
    if thick is None:
        thick = [None]*len(layer_list)
    if rough is None:
        rough = [None]*len(layer_list)
    if sld is None:
        sld = [None]*len(layer_list)

    layers.append(make_layer('air', thick=np.inf, rough=0, sld=0)) 
    for i, (layer, name, t, r,s) in enumerate(zip(layer_list, layer_list_names, thick, rough, sld)):
        layers.append(make_layer(name, thick=t, rough=r,sld=s, material=layer))
    
    structure = Structure(layers)
    return structure

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
    l = SLD(sld+1j*isld, name = name)
    return l(thick, rough)

def make_model(structure, cfg=cfg_model): 
    model = ReflectModel(structure, bkg=cfg['bkg'], dq=cfg['dq'], scale = cfg['scale'], q_offset=cfg['q_offset'])
    model.bkg.setp(bounds = (1e-9, 5e-7), vary = cfg_vary['bkg']) #background
    model.dq.setp(bounds = (0.001, 2.2), vary = cfg_vary['dq']) # due to error?
    model.scale.setp(bounds = (0.8, 1.4), vary = cfg_vary['scale']) # maximum value
    model.q_offset.setp(bounds = (-.025, 0.025), vary = cfg_vary['q_offset']) # due to error? 
    return model 

def repeat_scans(file, level='alox/al', qrng=[0.025, 0.3], save_name='test', cfg_in={}, cfg_in2={}, cfg_sld_in={}, run_mc=False, SLD_rng=[1,0.3], plot=False, no_fit=False, careful=False,n=5, start_new=False):
    chi=[]
    chi_start=[]
    chi_min = 1e3
    for i in range(n):
        if start_new or i==0: 
            obj, par = sim_al(file, level, qrng, save_name, cfg_in, cfg_in2, cfg_sld_in, run_mc, SLD_rng, plot, no_fit, careful)
        else:             
            obj, par = refit(copy.deepcopy(obj), file, qrng, plot)
            chi_start.append(par['start_chi'])
        chi.append(par['chisq'])
        if par['chisq'] < chi_min: 
            chi_min = par['chisq']
            obj_min = obj
            par_min = par
    return obj_min, par_min, chi, chi_start

def check_cat(level):
    sld=[]
    layer_list = level.split('/')
    ox_indices = [i+1 for i, layer in enumerate(layer_list) if 'ox' in layer]
    metal_indices = [i+1 for i, layer in enumerate(layer_list) if 'al' == layer or 'nb' == layer]
    buff_indices = [i+1 for i, layer in enumerate(layer_list) if 'buff' in layer]
    cat = {'ox':ox_indices, 'metal':metal_indices, 'buff':buff_indices}
    for m in metal_indices:
        sld.append(cfg_sld[layer_list[m-1]])
    return cat, sld 

def sim(file, level, params, qrng=[0.025, 0.3], plot=True, SLD_rng=[0.6, 0.15], careful=False, fit=True, multi=None, plot_init=False, img_path=None, run_mc=False, use_err=False): 
    # possible layer names are al, nb, nbox, alox, buff 
    fname = level.replace('/', '-')
    structure, model = make_model_par(level, params)
    cats, slds = check_cat(level)

    data = loadRefData(file, qrng=qrng, use_err=use_err) #this function loads to the data and converts the x-data from 2-theta to q
    objective = Objective(model, data, transform=Transform("logY"))   
    if fit is False and multi is None: 
        multi=False
    else:
        multi=True
    if img_path is None: 
        save=False
    else:
        save=True
    if plot and not multi:
        fig, a = plt.subplots()
        a.set_xlabel("q (1/Å)")
        a.set_ylabel("Normalized Counts")
        if plot_init: 
            a.loglog(data.x, objective.model(data.x),'k')
        a.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)
        if save:
            save_fig(fig, img_path, fname)
        #print(objective)
    
    # Defining the bounds, this can definitely be improved.
    j=0
    for i, s in enumerate(structure):
        if i in cats['ox']:
            s.sld.real.setp(bounds=((1-SLD_rng[0]/1.5)*s.sld.real.value,(1+SLD_rng[0])*s.sld.real.value), vary=True)
            if careful: 
                s.thick.setp(bounds = (s.thick.value/2,1.5*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
            else:
                s.thick.setp(bounds = (0,45), vary = True)
                s.rough.setp(bounds = (0,13), vary = True)
        elif i in cats['metal']:
            if careful: 
                s.thick.setp(bounds = (0.9*s.thick.value,1.1*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
                s.sld.real.setp(bounds=((1-SLD_rng[1])*slds[j],(1+SLD_rng[1])*slds[j]), vary=True)
                j+=1
            else:
                #s.thick.setp(bounds = (s.thick.value/2,s.thick.value*2), vary = True)
                s.thick.setp(bounds = (0.9*s.thick.value,1.1*s.thick.value), vary = True)
                s.sld.real.setp(bounds=((1-SLD_rng[1])*s.sld.real.value,(1+SLD_rng[1])*s.sld.real.value), vary=True)
                s.rough.setp(bounds = (0,10), vary = True)
        elif i in cats['buff']:
            s.sld.real.setp(bounds=((1-SLD_rng[0])*s.sld.real.value,(1+SLD_rng[0])*s.sld.real.value), vary=True)
            if careful: 
                s.thick.setp(bounds = (0.7*s.thick.value,1.3*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
            else:
                s.thick.setp(bounds = (0,40), vary = True)
                s.rough.setp(bounds = (0,30), vary = True)

        rough = params['sub_rough_max']
        structure[-1].rough.setp(bounds = (0,rough), vary = True)
    print(objective.chisqr())
    if fit:
        fitter = CurveFitter(objective)
        fitter.fit("differential_evolution", maxiter=2000, tol=0.005, mutation=(0.5,1.2))

        if plot:
            plot_obj([objective], [''], path=img_path, fname=fname, save=save)
            if save:
                save_text(str(objective), img_path, fname)
            #plt.semilogy(data.x, objective.model(data.x))

            # fig2 = plt.figure()
            # plt.plot(data.x, objective.residuals())

            # plt.figure()
            # plt.plot(*structure.sld_profile())
            # plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
            # plt.xlabel("distance / $\\AA$")
        if run_mc:
            mc_ana(objective, fitter)

        params = process_objective(objective)
        params['sub_rough_max'] = rough
        
    
    return objective, params

def mc_ana(objective, fitter):
    
    fitter.sample(400, pool=1)
    fitter.reset()
    res = fitter.sample(30, nthin=20, pool=1)
    objective.corner()
    print(objective)
    fig,ax = objective.model.structure.plot(samples = 100)
    

def refit(objective, file, qrng=[0.025, 0.3], plot=True, reset=False, SLD_rng = [0.2, 0.2], fit=True, save=True, img_path=None, fname='', use_err=False): 
    data = loadRefData(file, qrng=qrng, use_err=use_err) #this function loads to the data and converts the x-data from 2-theta to q
    objective.data = data
    if plot and not fit:
        fig1 = plt.figure()             
        plt.xlabel("q (1/A)")
        plt.ylabel("logR")
        plt.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)     
        plt.semilogy(data.x, objective.model(data.x))
    # have it reset the bounds 
    if reset: 
        for s in objective.model.structure:
            s.sld.real.bounds.lb = (1-SLD_rng)*s.sld.real.value
            s.sld.real.bounds.ub = (1+SLD_rng)*s.sld.real.value
            s.thick.bounds.lb = (1-SLD_rng)*s.thick.value
            s.thick.bounds.ub = (1+SLD_rng)*s.thick.value
            s.rough.bounds.lb = (1-SLD_rng)*s.rough.value
            s.rough.bounds.ub = (1+SLD_rng)*s.rough.value
    if not fit: 
        return objective, qrng
    fitter = CurveFitter(objective)
    start_chi = objective.chisqr()
    fitter.fit("differential_evolution")
    params = process_objective(objective)
    params['start_chi'] = start_chi

    if plot: 
        plot_obj([objective], [''], path=img_path, fname=fname, save=save)

    return objective, params

def process_objective(obj): 
    thk =[]
    rough =[]
    sld = []
    lays = []
    for i in range(1,len(obj.parameters[1])):
        check_rng(obj.parameters[1][i][0])
        check_rng(obj.parameters[1][i][2])
        check_rng(obj.parameters[1][i][1][0])
        lays.append(obj.parameters[1][i].name)
        rough.append(obj.parameters[1][i][2].value)
        thk.append(obj.parameters[1][i][0].value)
        sld.append(obj.parameters[1][i][1][0].value)

    params = {'thk':thk, 'rough':rough, 'sld':sld, 'layers':lays}

    for item in params:
        params[item]=np.array(params[item])
    # Add the model params to the dict. 
    for i in range(len(obj.parameters[0])):
        params[obj.parameters[0][i].name] = obj.parameters[0][i].value
    params['chisq'] = obj.chisqr()
    return params 

def compare_models(pars):
    pp = {'sld':{}, 'thick':{}, 'rough':{},'chisq':[], 'dq - resolution':[], 'q_offset':[], 'scale':[], 'bkg':[]} 
    layer_list = pars[-1]['layers']

    for l in layer_list:
        indices = [i for i, layer in enumerate(layer_list) if layer == l]  
        for i, index in enumerate(indices):
            if i>0: l = l + str(i)
            pp['rough'][l]=[]
            pp['thick'][l]=[]
            pp['sld'][l]=[]

    for p in pars:
        pp['chisq'].append(p['chisq'])
        pp['dq - resolution'].append(p['dq - resolution'])
        pp['q_offset'].append(p['q_offset'])
        pp['scale'].append(p['scale'])
        pp['bkg'].append(p['bkg'])

        
        for l in set(layer_list): 
            if l in p['layers']:
                #index = p['layers'].index(l)
                indices = [i for i, layer in enumerate(p['layers']) if layer == l]
                for i, index in enumerate(indices):
                    if i>0: l = l + str(i)
                    pp['rough'][l].append(p['rough'][index])
                    pp['thick'][l].append( p['thk'][index])
                    pp['sld'][l].append(p['sld'][index])
    return pp

def check_rng(par):
    if par.value > 0.95 * par.bounds.ub: 
        print(f"Warning: {par.name} is at the upper edge of its bounds")
    if par.value < 1.05 * par.bounds.lb:
        print(f"Warning: {par.name} is at the lower edge of its bounds")

def collect_pars(out):
    full_pars = [None]*len(out)
    for i, o in enumerate(out): 
        full_pars[i]=[oo[1] for oo in o]
    return full_pars

def plot_obj(obj, labs, multi=True, save=True, path=None, fname=''):

    if multi: 
        #fig, ax = plt.subplots(2,2, figsize=(12,6))
        #ax=ax.flatten()
        fig, ax = plt.subplot_mosaic([['a', 'a'],['b', 'c']], figsize=(8,7),
                              constrained_layout=True)

        ax['a'].set_xlabel('q (1/Å)')
        ax['a'].set_ylabel('Normalized Counts')
        ax['b'].set_xlabel('q (1/Å)')
        ax['b'].set_ylabel('Residuals')
        ax['c'].set_ylabel('SLD /$10^{-6} \\AA^{-2}$')
        ax['c'].set_xlabel('distance / $\\AA$')
        if len(obj)==1:
            ax['a'].loglog(obj[0].data.x, obj[0].data.y, '.-', markersize=4, label='Data', color='#b51d14')
        else: 
            ax['a'].loglog(obj[0].data.x, obj[0].data.y, '.-', markersize=4, label='Data', color='k')
        for l, objective in zip(labs,obj):
            ax['a'].semilogy(objective.data.x, objective.model(objective.data.x), label=l)
            ax['b'].plot(objective.data.x, objective.residuals(), label=l)
            ax['c'].plot(*objective.model.structure.sld_profile(), label=l)
        ax['b'].axhline(y=0, color='black', linestyle='--')
        ax['a'].legend()
        if len(labs[0])>0: 
            ax['b'].legend()
            ax['c'].legend()
        fig.suptitle(fname)
    else: 
        fig = plt.figure()             
        plt.xlabel("q (1/Å)")
        plt.ylabel("Normalized Counts")
        #plt.loglog(data.x, model(data.x), linewidth=0.5, label='guess')
        plt.loglog(obj[0].data.x, obj[0].data.y, '.-', markersize=4, label='Data', color='#b51d14')
        for l, objective in zip(labs,obj):
            plt.semilogy(objective.data.x, objective.model(objective.data.x), label=l)
        
        plt.legend()
        fig2 = plt.figure()
        for l, objective in zip(labs,obj):
            plt.plot(objective.data.x, objective.residuals(), label=l)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend()
        plt.figure()
        for l, objective in zip(labs,obj):
            plt.plot(*objective.model.structure.sld_profile(), label=l)
        plt.legend()
        plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
        plt.xlabel("distance / $\\AA$")
    
    fig.tight_layout()
    plt.show()
    if save:
        save_fig(fig, path, fname)


    return fig

def plot_vary(var, vals, name, level=1, cfg={}, q=None): 
    sns.set_palette('coolwarm',len(vals))
    if q is None: 
        q = np.linspace(4e-2, 1,5000)
    if len(vals)>4:
        fig, ax = plt.subplots(2,3,sharey=True, sharex=True, figsize=(10,6)) 
    else:
        fig, ax = plt.subplots(2,2,sharey=True, sharex=True, figsize=(7,6)) 

    ax[1,0].set_xlabel('q (1/Å)')
    ax[1,1].set_xlabel('q (1/Å)')
    ax[0,0].set_ylabel('Normalized Counts')
    ax[1,0].set_ylabel('Normalized Counts')

    ax = ax.flatten()
    fig2, ax2 = plt.subplots(1,1)

    for i in range(len(vals)):
        cfg[var] = vals[i]
        model = make_model_nb(level, cfg)
        ax[i].loglog(q, model(q), linewidth=1)
        ax[i].set_title(f'{vals[i]:.1f}')
        ax2.semilogy(q, model(q), linewidth=1, label=f'{vals[i]:.1f}')

    fig.suptitle(name)
    fig2.suptitle(name)
    fig.tight_layout()
    ax2.legend()
    
    fig.savefig(name+'.png')
    
    ax2.set_xlabel('q (1/Å)')
    ax2.set_ylabel('Normalized Counts')
    
    fig2.tight_layout()
    fig2.savefig(name+'2.png')

def plot_model(cfg={}, data=None, name=None, level='nb', q=None, ax=None, fig=None):
    
    if q is None: 
        q = np.linspace(1e-2, 1,5000)

    if ax is None:
        fig, ax = plt.subplots(1,1)

    model = make_model_nb(level, cfg)
    
    ax.semilogy(q, model(q), linewidth=0.5, label=name)
    if data is not None:
        ax.semilogy(data.x, data.y, '.-', color='#b51d14', linewidth=0.5, markersize=4, label=name)
    ax.legend()
    fig.tight_layout()
    #if name is not None:
        #fig.suptitle(name)

        #fig.savefig(name+'.png')
    
    return fig, ax
    