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

#'Al2O3': 33.385,
cfg_model = {'bkg': 3e-8, 'dq': 2, 'scale': 1, 'q_offset': 0}
cfg_sld = {'al':22.425, 'nb':64.035, 'alox':28.419, 'nbox':17.819,'al2o3':33.319,'buff':12}
cfg_isld = {'al':-0.411, 'nb':-3.928, 'alox':-0.37, 'nbox':-.3,'al2o3':-0.388,'buff':-0.2,'air':0}
#cfg_isld = {'al':0, 'nb':0, 'alox':0, 'nbox':0,'al2o3':0,'buff':0,'air':0}
cfg_rough = {'al':1, 'nb':1, 'alox':1, 'nbox':1,'al2o3':5,'buff':1}
cfg_thk = {'al':200, 'nb':500, 'alox':10, 'nbox':15,'buff':20}
#cfg_mat = {'al': {'rough':2, 'sld':22.425}, 'nb':{'rough':1, 'sld':64.035}, 'alumina':{'rough':2}, 'bottom_oxide_rough':1, 'substrate_rough':2, 
#               'al_sld':22.425, 'nb_sld':64.035, 'top_oxide_sld':8.419, 'bottom_oxide_sld':27.819,'substrate_sld':33.385}

cfg_vary = {'bkg':True, 'dq':True, 'scale':True, 'q_offset':True, 
                'al_thk':True, 'nb_thk':True, 'top_oxide_thk':True, 'bottom_oxide_thk':True, 
                'al_rough':True,'al_sld':True,'nb_rough':True, 'nb_sld':True, 'top_oxide_rough':True, 'top_oxide_sld':True, 
                'bottom_oxide_rough':True, 'bottom_oxide_sld':True, 'substrate_sld':False, 'substrate_rough':True, 'middle_oxide_thk':True, 'middle_oxide_rough':True, 'middle_oxide_sld':True}

def save_fig(fig, path, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + "/_" + name + "_" + timestr+ ".png"
    fig.savefig(path_out)
    return str(path_out)

def save_text(string, path, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path_out = os.path.splitext(path)[0] + "_" + name + "_" + timestr + ".txt"
    text_file = open(path_out, "w")
    text_file.write(string)
    text_file.close()
    return str(path_out)

def loadRefData(path,x_start=None, x_end=None, qrng=None):
    data = np.loadtxt(path) #load text
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = len(data)
    x = data[x_start:x_end,0].T
    y = data[x_start:x_end,1].T/max(data[:,1])
    #conv t-th to q using 0.72769 A wavelength
    x_new = refnx.util.q(x/2, 0.72769) 
    if qrng is not None:
        y = y[(x_new>qrng[0]) & (x_new<qrng[1])]
        x_new = x_new[(x_new>qrng[0]) & (x_new<qrng[1])]        
    compiled = (x_new,y)
    #returns the ReflectDataset object
    #print (f"Model will fit:\nd_min: {min(x_new)} A\n"
    #   f"d_max: {max(x_new)} A\n")

    return ReflectDataset(compiled)

def generate_table(data):
    # Create a DataFrame from the provided data
    df = pd.DataFrame({
        'Layer': data['layers'],
        'Thickness (A)': data['thk'],
        'Roughness (A)': data['rough'],
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

def make_dict(cfg): 
    SLD_dict = dict()
    #cfg['buff']=1/2*cfg['alox']+1/2*cfg['al']
    SLD_val_dict = {'Substrate':cfg['al2o3'],'Air': 0,
        'nbox':cfg['nbox'],
        'Nb':cfg['nb'],
        'Al' : cfg['al'],'Al_top' : cfg['al'],   
        'AlOx_bot': cfg['alox'],'AlOx_top': cfg['alox'], 'AlOx_JJ': cfg['alox'],
        'Buff_JJ_bot': cfg['buff'], 'Buff_JJ_top': cfg['buff'], 'Buff_top': cfg['buff']}
    for material in SLD_val_dict:
        SLD_dict[material] = SLD(SLD_val_dict[material], name = material)

    return SLD_dict

def make_layer2(SLD_dict, cfg_mat, material, layer): 
    return SLD_dict[material](cfg_mat[layer]['thk'],cfg_mat[layer]['rough'])

def make_model_par(level, par):
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
    model.dq.setp(bounds = (0.001, 3), vary = cfg_vary['dq']) # due to error?
    model.scale.setp(bounds = (0.8, 2), vary = cfg_vary['scale']) # maximum value
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
    layer_list = level.split('/')
    ox_indices = [i+1 for i, layer in enumerate(layer_list) if 'ox' in layer]
    metal_indices = [i+1 for i, layer in enumerate(layer_list) if 'al' == layer or 'nb' == layer]
    buff_indices = [i+1 for i, layer in enumerate(layer_list) if 'buff' in layer]
    cat = {'ox':ox_indices, 'metal':metal_indices, 'buff':buff_indices}
    return cat

def sim(file, level, params, qrng=[0.025, 0.3], plot=True, SLD_rng=[0.6, 0.15], careful=False, fit=True): 
    # possible layer names are al, nb, nbox, alox, buff 
    structure, model = make_model_par(level, params)
    cats = check_cat(level)

    data = loadRefData(file, qrng=qrng) #this function loads to the data and converts the x-data from 2-theta to q
    objective = Objective(model, data, transform=Transform("logY"))   

    if plot:
        fig1 = plt.figure()             
        plt.xlabel("q (1/Å)")
        plt.ylabel("Normalized Counts")
        #plt.loglog(data.x, objective.model(data.x),'k')
        plt.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)
        #print(objective)
    
    # Defining the bounds, this can definitely be improved.
    for i, s in enumerate(structure):
        if i in cats['ox']:
            s.sld.real.setp(bounds=((1-SLD_rng[0]/1.5)*s.sld.real.value,(1+SLD_rng[0])*s.sld.real.value), vary=True)
            if careful: 
                s.thick.setp(bounds = (s.thick.value/2,1.5*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
            else:
                s.thick.setp(bounds = (0,45), vary = True)
                s.rough.setp(bounds = (0,10), vary = True)
        elif i in cats['metal']:
            s.sld.real.setp(bounds=((1-SLD_rng[1])*s.sld.real.value,(1+SLD_rng[1])*s.sld.real.value), vary=True)
            if careful: 
                s.thick.setp(bounds = (0.8*s.thick.value,1.2*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
            else:
                s.thick.setp(bounds = (s.thick.value/2,s.thick.value*2), vary = True)
                s.rough.setp(bounds = (0,8), vary = True)
        elif i in cats['buff']:
            s.sld.real.setp(bounds=((1-SLD_rng[0])*s.sld.real.value,(1+SLD_rng[0])*s.sld.real.value), vary=True)
            if careful: 
                s.thick.setp(bounds = (0.8*s.thick.value,1.2*s.thick.value), vary = True)
                s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
            else:
                s.thick.setp(bounds = (0,40), vary = True)
                s.rough.setp(bounds = (0,25), vary = True)

        rough = params['sub_rough_max']
        structure[-1].rough.setp(bounds = (0,rough), vary = True)

    if fit:
        fitter = CurveFitter(objective)
        fitter.fit("differential_evolution", maxiter=2000)

        if plot:
            plt.semilogy(data.x, objective.model(data.x))

            fig2 = plt.figure()
            plt.plot(data.x, objective.residuals())

            plt.figure()
            plt.plot(*structure.sld_profile())
            plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
            plt.xlabel("distance / $\\AA$")

        params = process_objective(objective)
        params['sub_rough_max'] = rough
        
    
    return objective, params

def sim_al(file, level='alox/al', qrng=[0.025, 0.3], save_name='test', cfg_in={}, cfg_in2={}, cfg_sld_in={}, run_mc=False, SLD_rng=[1,0.3], plot=True, no_fit=False, careful=False):
    cfg_mat = {'nb':{'rough':6,'thk':80},'al': {'rough':1, 'thk':300},'al_top': {'rough':1, 'thk':210}, 
               'top_oxide':{'rough':1, 'thk':10},'JJ':{'rough':2, 'thk':20},'bottom_oxide':{'rough':1, 'thk':5},
                'buffer':{'rough':2, 'thk':5},
               'substrate':{'rough':5,'rough_max':15}}
    # Carefull will only allow small changes in variables. 
    cfg_sld.update(cfg_sld_in)
    cfg_mat.update(cfg_in)
    cfg_model.update(cfg_in2)
    # Make SLD dict 
    SLD_dict = make_dict(cfg_sld)

    # Make layers 
    air = SLD_dict['Air'](np.inf,0)
    substrate = SLD_dict['Substrate'](np.inf,0) # this is crystalline? 
    
    al = make_layer2(SLD_dict, cfg_mat, 'Al', 'al')
    nb = make_layer2(SLD_dict, cfg_mat, 'Nb', 'nb')
    al_top = make_layer2(SLD_dict, cfg_mat, 'Al_top', 'al_top')

    bottom_oxide = make_layer2(SLD_dict, cfg_mat, 'AlOx_bot', 'bottom_oxide')
    top_oxide = make_layer2(SLD_dict, cfg_mat, 'AlOx_top', 'top_oxide')
    jj = make_layer2(SLD_dict, cfg_mat, 'AlOx_JJ', 'JJ')

    buffer = make_layer2(SLD_dict, cfg_mat, 'Buff_top', 'buffer')
    jj_buffer_top = make_layer2(SLD_dict, cfg_mat, 'Buff_JJ_top', 'buffer')
    jj_buffer_bot = make_layer2(SLD_dict, cfg_mat, 'Buff_JJ_bot', 'buffer')
    
    if level == 'al': 
        structure =  air | al | substrate
        ox_layers = []
        met_layers = [1]
    elif level == 'alox/al':
        structure =  air | top_oxide | al | substrate
        ox_layers = [1]
        met_layers=[2]
    elif level == 'alox/al/alox':
        structure =  air | top_oxide | al | bottom_oxide | substrate
        ox_layers = [1,3]
        met_layers=[2]
    elif level == 'al/alox': 
        structure =  air | al | bottom_oxide | substrate
        ox_layers = [2]
        met_layers=[1]
    elif level == 'alox/buff/al/alox': 
        structure = air | top_oxide | buffer | al | bottom_oxide | substrate
        ox_layers = [1,2,4]
        met_layers=[3]
    elif level =='al/alox/al':
        structure = air | al | jj | al | substrate
        ox_layers = [2]
        met_layers=[1,3]
    elif level == 'alox/al/alox/al': 
        structure = air | top_oxide | al_top | jj | al | substrate
        ox_layers = [1,3]
        met_layers=[2,4]
    elif level == 'alox/al/alox/al/buff': 
        structure = air | top_oxide | al_top | jj | al | buffer | substrate
        ox_layers = [1,3,5]
        met_layers=[2,4]
    elif level == 'alox/al/buff/alox/buff/al/alox': 
        structure = air | top_oxide | al_top | jj_buffer_top | jj | jj_buffer_bot | al | bottom_oxide | substrate
        ox_layers = [1,3,4,5,7]
        met_layers=[2,6]
    elif level == 'alox/al/buff/alox/al/buff': 
        structure = air | top_oxide | al_top | jj_buffer_top | jj | al |  jj_buffer_bot | substrate
        ox_layers = [1,3,4,6]
        met_layers=[2,5]
    elif level == 'alox/al/buff/alox/buff/al': 
        structure = air | top_oxide | al_top | jj_buffer_top | jj | jj_buffer_bot | al | substrate
        ox_layers = [1,3,4,5]
        met_layers=[2,6]
    elif level == 'al/nb':
        structure = air | al | nb | substrate
        ox_layers = []
        met_layers = [1,2]
    elif level == 'ox/al/nb':
        structure = air |top_oxide | al | nb | substrate
        ox_layers = [1]
        met_layers = [2,3]
    elif level == 'ox/al/nb/ox':
        structure = air |top_oxide | al | nb | bottom_oxide | substrate
        ox_layers = [1,4]
        met_layers = [2,3]
    elif level == 'ox/al/buff/nb/ox':
        structure = air |top_oxide | al | buffer | nb | bottom_oxide | substrate
        ox_layers = [1,3,5]
        met_layers = [2,4]
    elif level == 'ox/al/buff/buff/nb/ox':
        structure = air |top_oxide | al | buffer |jj_buffer_bot | nb | bottom_oxide | substrate
        ox_layers = [1,3,4,6]
        met_layers = [2,5]
        #ox_layers = [1,3,5,6]
        #met_layers = [2,4]
    elif level == 'ox/al/buff/nb/ox/buff':  
        structure = air |top_oxide | al | buffer |jj_buffer_bot | nb | bottom_oxide | substrate
        ox_layers = [1,3,5,6]
        met_layers = [2,4]


    model = make_model(structure, cfg_model)
    for i in ox_layers:
        s=structure[i]   
        s.sld.real.setp(bounds=((1-SLD_rng[0]/1.5)*s.sld.real.value,(1+SLD_rng[0])*s.sld.real.value), vary=True)
        if careful: 
            s.thick.setp(bounds = (s.thick.value/2,1.5*s.thick.value), vary = True)
            s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
        else:
            #s.thick.setp(bounds = (0,s.thick.value*4), vary = True)
            s.thick.setp(bounds = (0,40), vary = True)
            s.rough.setp(bounds = (0,10), vary = True)                    

    for i in met_layers:
        s=structure[i]
        s.sld.real.setp(bounds=((1-SLD_rng[1])*s.sld.real.value,(1+SLD_rng[1])*s.sld.real.value), vary=True)
        if careful: 
            s.thick.setp(bounds = (0.8*s.thick.value,1.2*s.thick.value), vary = True)
            s.rough.setp(bounds = (s.rough.value/2,1.5*s.rough.value), vary = True)
        else:
            s.thick.setp(bounds = (s.thick.value/2,s.thick.value*2), vary = True)
            s.rough.setp(bounds = (0,8), vary = True)
    
    structure[-1].rough.setp(bounds = (0,cfg_mat['substrate']['rough_max']), vary = True)
    data = loadRefData(file, qrng=qrng) #this function loads to the data and converts the x-data from 2-theta to q
    objective = Objective(model, data, transform=Transform("logY"))    
    
    if plot:
        fig1 = plt.figure()             
        plt.xlabel("q (1/Å)")
        plt.ylabel("Normalized Counts")
        plt.semilogy(data.x, objective.model(data.x))
        plt.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)
        print(objective)
    if no_fit: 
        return objective, structure, cfg_mat
    fitter = CurveFitter(objective)
    fitter.fit("differential_evolution")

    if plot:
        plt.semilogy(data.x, objective.model(data.x))

        fig2 = plt.figure()
        plt.plot(data.x, objective.residuals())

        plt.figure()
        plt.plot(*structure.sld_profile())
        plt.ylabel("SLD /$10^{-6} \\AA^{-2}$")
        plt.xlabel("distance / $\\AA$")
    
    params = process_objective(objective)

    if run_mc: 
        fitter.sample(400, pool=1)
        fitter.reset()
        res = fitter.sample(30, nthin=20, pool=1)
        #objective.corner()
        print(objective)
        #fig,ax = structure.plot(samples = 100)
        #corner_plot = objective.corner()
    return objective, params

def refit(obj, file, qrng=[0.025, 0.3], plot=True, reset=False, fit=True): 
    data = loadRefData(file, qrng=qrng) #this function loads to the data and converts the x-data from 2-theta to q
    obj.data = data
    if plot:
        fig1 = plt.figure()             
        plt.xlabel("q (1/A)")
        plt.ylabel("logR")
        plt.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)
        plt.semilogy(data.x, obj.model(data.x))
    # have it reset the bounds 
    if reset: 
        for s in obj.model.structure:
            s.sld.real.bounds.lb = (1-0.2)*s.sld.real.value
            s.sld.real.bounds.ub = (1+0.2)*s.sld.real.value
            s.thick.bounds.lb = (1-0.2)*s.thick.value
            s.thick.bounds.ub = (1+0.2)*s.thick.value
            s.rough.bounds.lb = (1-0.2)*s.rough.value
            s.rough.bounds.ub = (1+0.2)*s.rough.value
    if not fit: 
        return obj, qrng
    fitter = CurveFitter(obj)
    start_chi = obj.chisqr()
    fitter.fit("differential_evolution")
    params = process_objective(obj)
    params['start_chi'] = start_chi

    if plot:
        plt.semilogy(data.x, obj.model(data.x))
        fig2 = plt.figure()
        plt.plot(data.x, obj.residuals())

    return obj, params

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

    params = {'thk':thk, 'rough':rough, 'sld':sld, 'layers':lays, 'chisq':obj.chisqr()}

    for item in params:
        params[item]=np.array(params[item])
    # Add the model params to the dict. 
    for i in range(len(obj.parameters[0])):
        params[obj.parameters[0][i].name] = obj.parameters[0][i].value
    
    return params 

def compare_models(pars):
    pp = {'sld':{}, 'thick':{}, 'rough':{},'chisq':[]} 
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

def plot_obj(obj, labs):

    fig1 = plt.figure()             
    plt.xlabel("q (1/A)")
    plt.ylabel("logR")
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
    
def run_simulation(file, level=3, xrng=[50,300], save_name='test', cfg_in={}, cfg_vary_in={}, run_mc=True, SLD_rng=0.2, plot=True, fit=True, objective=None, verbose=True): 

    cfg = {'bkg': 3e-8, 'dq': 0.05, 'scale': 1, 'q_offset': 0, 
            'al_thk':50, 'nb_thk':600, 'top_oxide_thk':15, 'bottom_oxide_thk':5, 
                'al_rough':2, 'nb_rough':1, 'top_oxide_rough':2, 'bottom_oxide_rough':1, 'substrate_rough':2, 
                'al_sld':22.425, 'nb_sld':64.035, 'top_oxide_sld':8.419, 'bottom_oxide_sld':7.819,'substrate_sld':8.419}
                #'al_sld':2.078, 'nb_sld':3.919, 'top_oxide_sld':1.436, 'bottom_oxide_sld':0.977, 'substrate_sld':1.436}
                #
    cfg_vary = {'bkg':True, 'dq':False, 'scale':True, 'q_offset':True, 
                'al_thk':True, 'nb_thk':True, 'top_oxide_thk':True, 'bottom_oxide_thk':True, 
                'al_rough':True,'al_sld':True,'nb_rough':True, 'nb_sld':True, 'top_oxide_rough':True, 'top_oxide_sld':True, 
                'bottom_oxide_rough':True, 'bottom_oxide_sld':True, 'substrate_sld':False, 'substrate_rough':True, 'middle_oxide_thk':True, 'middle_oxide_rough':True, 'middle_oxide_sld':True}
    cfg.update(cfg_in)
    cfg_vary.update(cfg_vary_in)
    if objective is None: 
        SLD_dict = dict()
        SLD_val_dict = {'Al2O3':cfg['substrate_sld'],
                    'Nb2O5_bot':cfg['bottom_oxide_sld'],
                    'Nb2O5_mid':cfg['bottom_oxide_sld'],
                    'Nb':cfg['nb_sld'],
                'Al' : cfg['al_sld'],
                'air': 0,
                'native': cfg['top_oxide_sld']}
        for material in SLD_val_dict:
                SLD_dict[material] = SLD(SLD_val_dict[material], name = material)

        al = SLD_dict['Al'](cfg['al_thk'],cfg['al_rough'])
        air = SLD_dict['air'](np.inf,0)
        nb = SLD_dict['Nb'](cfg['nb_thk'],cfg['nb_rough'])
        top_oxide = SLD_dict['native'](cfg['top_oxide_thk'], cfg['top_oxide_rough']) # this is al2o3 but hermal? 
        bottom_oxide = SLD_dict['Nb2O5_bot'](cfg['bottom_oxide_thk'], cfg['bottom_oxide_rough']) # this is al2o3 but hermal? 
        middle_oxide = SLD_dict['Nb2O5_mid'](cfg['bottom_oxide_thk'], cfg['bottom_oxide_rough']) # this is al2o3 but hermal? 
        substrate = SLD_dict['Al2O3'](np.inf,0) # this is crystalline? 
        
        if level == 1: 
            structure =  air | al | nb | substrate  
        elif level==2: 
            structure =  air | top_oxide | al | nb | substrate  
        elif level==4: 
            structure =  air | al | nb | bottom_oxide | substrate
        elif level==3:         
            structure =  air | top_oxide | al | nb | bottom_oxide | substrate  
        elif level==5: 
            structure =  air | top_oxide | al | middle_oxide | nb | bottom_oxide | substrate
        
        model = ReflectModel(structure, bkg=cfg['bkg'], dq=cfg['dq'], scale = cfg['scale'], q_offset=cfg['q_offset'])
        model.bkg.setp(bounds = (1e-9, 5e-6), vary = cfg_vary['bkg']) #background
        model.dq.setp(bounds = (0.001, 0.1), vary = cfg_vary['dq']) # due to error?
        model.scale.setp(bounds = (0.9, 1.5), vary = cfg_vary['scale']) # maximum value
        model.q_offset.setp(bounds = (-.1, 0.1), vary = cfg_vary['q_offset']) # due to error? 

        al.thick.setp(bounds = (40,120), vary = cfg_vary['al_thk'])
        al.sld.real.setp(bounds = (5,45), vary = cfg_vary['al_sld'])
        al.rough.setp(bounds = (0,20), vary = cfg_vary['al_rough'])
            
        nb.thick.setp(bounds=(300, 900), vary=cfg_vary['nb_thk'])
        nb.sld.real.setp(bounds=(40, 90), vary=cfg_vary['nb_sld'])
        nb.rough.setp(bounds=(0, 10), vary=cfg_vary['nb_rough'])

        substrate.rough.setp(bounds=(0, 20), vary=cfg_vary['substrate_rough'])
        substrate.sld.real.setp(bounds=(2, 60), vary=cfg_vary['substrate_sld'])

        if level > 1 and level!=4:
            top_oxide.thick.setp(bounds=(0, 25), vary=cfg_vary['top_oxide_thk'])
            top_oxide.rough.setp(bounds=(0, 25), vary=cfg_vary['top_oxide_rough'])
            top_oxide.sld.real.setp(bounds=(20, 34), vary=cfg_vary['top_oxide_sld'])
        if level > 2:
            bottom_oxide.thick.setp(bounds=(0, 25), vary=cfg_vary['bottom_oxide_thk'])
            bottom_oxide.rough.setp(bounds=(0, 75), vary=cfg_vary['bottom_oxide_rough'])
            bottom_oxide.sld.real.setp(bounds=(10, 80), vary=cfg_vary['bottom_oxide_sld'])
        if level > 4: 
            middle_oxide.thick.setp(bounds=(0, 25), vary=cfg_vary['middle_oxide_thk'])
            middle_oxide.rough.setp(bounds=(0, 75), vary=cfg_vary['middle_oxide_rough'])
            middle_oxide.sld.real.setp(bounds=(10, 80), vary=cfg_vary['middle_oxide_sld'])

        for s in structure: 
            s.sld.real.bounds.lb = (1-SLD_rng)*s.sld.real.value
            s.sld.real.bounds.ub = (1+SLD_rng)*s.sld.real.value

        data = loadRefData(file, xrng[0], xrng[1]) #this function loads to the data and converts the x-data from 2-theta to q
        objective = Objective(model, data, transform=Transform("logY"))
    else:
        data = loadRefData(file, xrng[0], xrng[1]) #this function loads to the data and converts the x-data from 2-theta to q
        model = []
    if plot:
            fig1 = plt.figure()             
            plt.xlabel("q (1/A)")
            plt.ylabel("logR")
            plt.loglog(data.x, data.y, '.-', color='#b51d14', linewidth=1, markersize=4)
    if fit: 
        print(objective)        
        #Does the curve fitting
        fitter = CurveFitter(objective)
        fitter.fit("differential_evolution")
        if verbose: 
            print(objective)
        if run_mc: 
            fitter.sample(400, pool=1)
            fitter.reset()
            res = fitter.sample(25, nthin=200, pool=1)
            structure.plot(samples = 100)
            fig,ax = structure.plot(samples = 100)
            corner_plot = objective.corner()

            print(objective)
        
    if plot:        
            plt.semilogy(data.x, objective.model(data.x))
            fig2 = plt.figure()
            plt.plot(data.x, objective.residuals())

    if plot: #toggle True to save fit figure, False to not save.
        save_name += "_reflectivity_"
        out = save_fig(fig1, path, save_name) #save file with filename specified here
        print ("Figure saved at:", out)

    if plot: #toggle True to save fit file, False to not save.
        save_name += "_objective_"
        save_text(str(objective), path, save_name)
        print ("Text file saved at:", out)

    return objective, model

def make_model_nb(level=3, cfg_in={}):

    cfg = {'bkg': 1e-9, 'dq': 0.05, 'scale': 1, 'q_offset': 0, 
            'al_thk':60, 'nb_thk':570, 'top_oxide_thk':15, 'bottom_oxide_thk':5, 'middle_oxide_thk':5, 
            'nb2_thk':300, 'nb2_rough':1,'middle_oxide_rough':1,'middle_oxide_sld':7.819,
                'al_rough':2, 'nb_rough':1, 'top_oxide_rough':2, 'bottom_oxide_rough':1, 'substrate_rough':5, 
                #'al_sld':2.078, 'nb_sld':3.919, 'top_oxide_sld':1.436, 'bottom_oxide_sld':0.977, 'substrate_sld':1.436}
                'al_sld':22.425, 'nb_sld':64.035, 'top_oxide_sld':8.385, 'bottom_oxide_sld':27.819,'substrate_sld':33.385}
    cfg.update(cfg_in)

    
    SLD_dict = dict()
    SLD_val_dict = {'Al2O3':cfg['substrate_sld'],
                'Nb2O5_bot':cfg['bottom_oxide_sld'],
                'Nb2O5_mid':cfg['middle_oxide_sld'],
                'Nb':cfg['nb_sld'],
                'Nb2':cfg['nb_sld'],
            'Al' : cfg['al_sld'],
            'air': 0,
            'native': cfg['top_oxide_sld']}
    for material in SLD_val_dict:
            SLD_dict[material] = SLD(SLD_val_dict[material], name = material)

    al = SLD_dict['Al'](cfg['al_thk'],cfg['al_rough'])
    air = SLD_dict['air'](np.inf,0)
    nb = SLD_dict['Nb'](cfg['nb_thk'],cfg['nb_rough'])
    nb2 = SLD_dict['Nb2'](cfg['nb2_thk'],cfg['nb2_rough'])
    top_oxide = SLD_dict['native'](cfg['top_oxide_thk'], cfg['top_oxide_rough']) # this is al2o3 but hermal? 
    bottom_oxide = SLD_dict['Nb2O5_bot'](cfg['bottom_oxide_thk'], cfg['bottom_oxide_rough']) # this is al2o3 but hermal? 
    middle_oxide = SLD_dict['Nb2O5_mid'](cfg['middle_oxide_thk'], cfg['middle_oxide_rough']) # this is al2o3 but hermal? 
    substrate = SLD_dict['Al2O3'](np.inf,cfg['substrate_rough']) # this is crystalline? 
    if level == 'nb': #0
        structure =  air | nb | substrate
    if level == 'ox/nb': #0.5 
        structure =  air | top_oxide | nb | substrate
    if level == 'nb/ox':
        structure =  air |  nb | bottom_oxide |substrate
    if level == 'ox/nb/ox': #0.75
        structure =  air | top_oxide | nb | bottom_oxide | substrate
    if level=='al/ox/nb': 
        structure =  air | al | top_oxide | nb | substrate
    if level=='nb/ox/nb': 
        structure =  air | nb2 | top_oxide | nb | substrate
    if level == 'al/nb': #1  
        structure =  air | al | nb | substrate  
    elif level=='ox/al/nb': #2
        structure =  air | top_oxide | al | nb | substrate  
    elif level=='al/nb/ox': # 4
        structure =  air | al | nb | bottom_oxide | substrate
    elif level=='ox/al/nb/ox': #3        
        structure =  air | top_oxide | al | nb | bottom_oxide | substrate  
    elif level=='ox/al/ox/nb/ox': # 5
        structure =  air | top_oxide | al | middle_oxide | nb | bottom_oxide | substrate
    
    model = ReflectModel(structure, bkg=cfg['bkg'], dq=cfg['dq'], scale = cfg['scale'], q_offset=cfg['q_offset'])

    return model 

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
    