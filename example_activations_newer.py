import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from plotting import plot_response_heatmaps
import config_file as c
from utility import get_timing_responses,save_stats
from model_fitting_stats import load_fit_data
from model_fitting import run_fit, tuned_function,mono_function




results_dir = c.RESULTS_DIR
control_condition_list = ["16nodes"]
shuffled_bool = False

ind_rnn_bool = True
     
depth = [5]
nums = [16]        

# get info repetition   
tuned_params = ["pref_x0", "pref_y0", "sigma_x0", "sigma_y0", "theta", "exponent","tuned_slope","tuned_intercept"]
mono_params = ["x0_exp", "y0_exp", "ratio_x0", "ratio_y0","mono_slope","mono_intercept"]

# hard coded examples, based on finding different response functions
tuned_rep = 28 # network replica
tuned_node = 8 # node
tuned_layer = 4 # layer
tuned_split_fit = 0 # data split
tuned_tt = '_ISI_period' # best fitting response function space

mixed_rep = 29
mixed_node = 5
mixed_layer = 4
mixed_split_fit = 1
mixed_tt = '_duration_period'

mono_rep = 29
mono_node = 4
mono_layer = 0
mono_split_fit = 0
mono_tt = '_duration_period'

layers = [tuned_layer, mixed_layer, mono_layer]
reps = [tuned_rep, mixed_rep, mono_rep]
nodes = [tuned_node, mixed_node, mono_node]
splits = [tuned_split_fit, mixed_split_fit, mono_split_fit]
timing_types = [tuned_tt, mixed_tt, mono_tt]
types = ["tuned","mixed","mono"]

overwrite = True
for i in range(len(types)):
    
    repetition = {"num_layers": 5, "num_hidden": 16, "norm": "layer_norm", "batch_size": 50, "ind_rnn": ind_rnn_bool, "weight_constraint": False,'LR':2e-3,'dropout_prob': c.DROPOUT_PROB,'assessed_per':'event','cross_v': True,'weight_reg':1e-08, 'scrambled_control': shuffled_bool, 'counter': reps[i]}   
    
    node = nodes[i]

    timing_type = timing_types[i]
    x0 = timing_type.split('_')[1]
    y0 = timing_type.split('_')[2]
    
    layer = layers[i]
    fit_split = splits[i]
   
    # get response and plot it
    accuracy_path, fig_file_name = save_stats([],name_dict,overwrite=overwrite)
    timing_res, timing_labels = get_timing_responses(results_dir, repetition, assessed_per="event",x0=x0,y0=y0,control_condition="16nodes")
    plot_response_heatmaps(timing_res, timing_labels, neuron_idx=int(node),x0=x0,y0=y0, normalize_per_plot=True, layer_id = layer,split = fit_split)
        
    name_dict = {'results_section':'exampleActivations','type':types[i],'split': 'fit'}
    accuracy_path, fig_file_name = save_stats([],name_dict)
    plt.savefig(os.path.join(accuracy_path,fig_file_name))    
    plt.show()

    # get responses in the other data split and plot them
    plot_response_heatmaps(timing_res, timing_labels, neuron_idx=int(node),x0=x0,y0=y0, normalize_per_plot=True, layer_id = layer,split = 1-fit_split)
    name_dict = {'results_section':'exampleActivations','type':types[i],'split': 'evaluate'}
    accuracy_path, fig_file_name = save_stats([],name_dict)
    plt.savefig(os.path.join(accuracy_path,fig_file_name))
        
    # get the function fits for the responses/ nodes
    for x0_name in ["duration","ISI"]:
        mono_fits, tuned_fits = run_fit(repetition,x0=x0_name,y0="period",control_condition="16nodes")
        first_cols = mono_fits.iloc[:,0:3]
        mono_fits_reduced = mono_fits.drop(columns = ['node','layer','split'])
        mono_fits_reduced = mono_fits_reduced.add_suffix("_" + x0_name + "_period")
        tuned_fits_reduced = tuned_fits.drop(columns = ['node','layer','split'])
        tuned_fits_reduced = tuned_fits_reduced.add_suffix("_" + x0_name + "_period")

        # blend dataframes
        if x0_name == "duration":
            all_fits = pd.concat([first_cols, mono_fits_reduced, tuned_fits_reduced], axis='columns')
        else:
            all_fits = pd.concat([all_fits,mono_fits_reduced, tuned_fits_reduced], axis='columns')

    tlabels = np.transpose(timing_labels/1000)
    all_fits_row = all_fits[np.logical_and(np.logical_and(all_fits["layer"]==layer,all_fits["node"]==node),all_fits["split"]==fit_split)]
    
    # make tuned responses of right space
    tuned_param_vals = [list(all_fits_row[param + timing_type])[0] for param in tuned_params]
    predicted_res = tuned_function(tlabels,*tuned_param_vals)

    t_res = predicted_res.reshape(-1, 1)
    data = np.concatenate((t_res, timing_labels), axis=1)
    frame = pd.DataFrame(data, columns=["value", x0, y0])
    frame = pd.pivot_table(frame, index=y0, columns=x0, values="value")
    frame = frame.reindex(frame.index.sort_values(ascending=False))
    fig, axes = plt.subplots(1, 1, figsize=(15, 1 * 10))                    
    sns.heatmap(frame, cmap="coolwarm", vmin=-0.1, vmax = 1.1, annot=False,cbar = True,ax=axes)
    
    name_dict = {'results_section':'exampleActivations','type':types[i],'split': 'tunedPrediction'}
    accuracy_path, fig_file_name = save_stats([],name_dict)
    plt.savefig(os.path.join(accuracy_path,fig_file_name))

    # make mono responses of right space
    mono_param_vals = [list(all_fits_row[param + timing_type])[0] for param in mono_params]
    predicted_res = mono_function(tlabels,*mono_param_vals)

    t_res = predicted_res.reshape(-1, 1)
    data = np.concatenate((t_res, timing_labels), axis=1)
    frame = pd.DataFrame(data, columns=["value", x0, y0])
    frame = pd.pivot_table(frame, index=y0, columns=x0, values="value")
    frame = frame.reindex(frame.index.sort_values(ascending=False))
  
    fig, axes = plt.subplots(1, 1, figsize=(15, 1 * 10))                    
    sns.heatmap(frame, cmap="coolwarm", vmin=-0.1, vmax = 1.1, annot=False,cbar = True,ax=axes)
    
    name_dict = {'results_section':'exampleActivations','type':types[i],'split': 'monoPrediction'}
    accuracy_path, fig_file_name = save_stats([],name_dict)
    plt.savefig(os.path.join(accuracy_path,fig_file_name))
    

