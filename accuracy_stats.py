import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
import pickle
import seaborn as sns
from scipy.stats import kruskal, shapiro,mannwhitneyu,wilcoxon,levene
from scikit_posthocs import posthoc_dunn
import matplotlib
import statsmodels

from pipeline import evaluate_net,plot_loss_functions_all_depths, combined_grid_plot
from create_special_conditions import evaluate_special_conditions
from utility import get_result_paths,save_stats



def value_to_color(val):
    '''
    Transforms values to colors along red-yellow-green scale
    '''

    n_colors = 256
    palette = sns.color_palette("RdYlGn",n_colors=n_colors)
    for color_id in range(n_colors):
        palette[color_id] = sns.set_hls_values(palette[color_id],l=.4)
   
    if val == val:
        ind = int(val * (n_colors - 1))
    else:
        ind = 0

    return palette[ind]

def plot_performance_heatmap_size_std(performance_measure, labels, stds:list=None, do_plotting:bool = True,ax = None):
    '''
        Plots triangular heatmap for timing accuracies. With sizes of the heatmap scales
            by variance across the measurements for this timing
    '''
        
    not_show = False

    try:
        data = np.concatenate((labels, performance_measure.reshape(-1, 1)), axis=1)
    except:
        data = np.concatenate((labels, performance_measure.T), axis=1)
    frame = pd.DataFrame(data, columns=["duration", "period", "value"])
    frame = frame.astype({"duration": int, "period": int, "value": float})
    frame = pd.pivot_table(frame, index="period",
                           columns="duration", values="value")
    frame = frame.reindex(frame.index.sort_values(ascending=False))
    
    frame = frame.unstack().reset_index(name='value')
    
    if do_plotting == True:   

        x = frame.loc[:,'duration']
        y = frame.loc[:,'period']
        color = frame.loc[:,'value']
        
        if stds is not None:
            data_stds = np.concatenate((labels, stds.reshape(-1, 1)), axis=1)
            frame_stds = pd.DataFrame(data_stds, columns=["duration", "period", "value"])
            frame_stds = frame_stds.astype({"duration": int, "period": int, "value": float})
            frame_stds = pd.pivot_table(frame_stds, index="period",
                                   columns="duration", values="value")
            frame_stds = frame_stds.reindex(frame_stds.index.sort_values(ascending=False))
            frame_stds = frame_stds.unstack().reset_index(name='value')
            size_scale = 1-frame_stds.loc[:,'value']*2
        else:
            size_scale = 1

    
        x_labels = list(range(50,1000,50))
        y_labels = list(range(100,1050,50))
        x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
        y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}
            
        if ax == None:
            fig,ax = plt.subplots()
        else:
            not_show = True
            fig = ax.figure
            
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        width *=fig.dpi 
        size = (width-40) /2.5
        
            
        ax.scatter(x=x.map(x_to_num),y=y.map(y_to_num),s = size*size_scale, marker = 's',c=color.apply(value_to_color).tolist(),edgecolors='none')
        ax.set_xticks([x_to_num[v] for v in x_labels])
        ax.set_xticklabels(x_labels,rotation=90)
        ax.set_yticks([y_to_num[v] for v in y_labels])
        ax.set_yticklabels(y_labels)
        
        ax.grid(False)
        ax.set_xticks([t for t in ax.get_xticks()])
        ax.set_yticks([t for t in ax.get_yticks()])
        
        ax.set_xlim([-0.5,max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5,max([v for v in y_to_num.values()]) + 0.5])
        
        ax.set_xlabel('Duration')
        ax.set_ylabel('Period')
        
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        for key, spine in ax.spines.items():
            spine.set_visible(False)    
        if not_show == False:
            plt.show()
        
    return frame


def compare_accuracy_model_depths(results_dir, all_repetitions, condition_list = None,top = 50):
    '''
        Compares the accuracy between different model depths

        top: if you want to compare it only between the top performing networks

    '''
   
    event_accuracies_per_depth = {}
    state_change_accuracies_per_depth = {}
    event_accuracies_per_depth_sorted = {}
    state_change_accuracies_per_depth_sorted = {}
    event_repetitions_best_accuracies = {}
    state_change_repetitions_best_accuracies = {}
    
    collect_depths = []

    for layer_id,repetitions_per_layer in enumerate(all_repetitions):
        if condition_list is not None:
            control_condition = condition_list[layer_id]
        else:
            control_condition = None
        
        mean_event_accuracies = []
        mean_state_change_accuracies = []
        for repetition in repetitions_per_layer:
            if len(repetition) == 1:
                repetition = repetition[0]
                
            # evaluate or get stored accuracies
            if control_condition == "special_condition":
                evaluate_special_conditions(results_dir, repetition, assessed_per="event", do_plotting = False,control_condition=control_condition)
            else:
                evaluate_net(results_dir, repetition,do_plotting = False,control_condition=control_condition)
            _, test_results_path, _, _, _, _, _, _ = get_result_paths(results_dir, repetition,control_condition=control_condition)
    
            with open(test_results_path, 'rb') as f:
                # get concatenated losses and fit_labels
                _, _, _, _, _, _, _, _, _,_, acc_proportions_event, acc_proportions_state_change = pickle.load(f)
                
            # for each model repetition get mean accuracy over all timings
            mean_event_accuracies.append(np.mean(acc_proportions_event))
            mean_state_change_accuracies.append(np.mean(acc_proportions_state_change))
        
        if control_condition == None or all(el=='16nodes' for el in condition_list):
            amount_layers = str(repetition['num_layers']) + ' layers'
        else:
            amount_layers = control_condition
            
        collect_depths.append(repetition['num_layers'])
            
        event_accuracies_per_depth[amount_layers] = mean_event_accuracies
        state_change_accuracies_per_depth[amount_layers] = mean_state_change_accuracies
    
    # sort them so if you want to make a selection of max accuracy you can (comparisons are not done between types of accuracy
    # so not paired anyway so this is fine)
    for layer_id,layer in enumerate(event_accuracies_per_depth.keys()):
        sort_ids_event = sorted(range(len(event_accuracies_per_depth[layer])),key = event_accuracies_per_depth[layer].__getitem__)
        event_accuracies_per_depth_sorted[layer] = [event_accuracies_per_depth[layer][id_event] for id_event in sort_ids_event][-top:]
        event_repetitions_best_accuracies[layer] = [all_repetitions[layer_id][id_event] for id_event in sort_ids_event][-top:]
        
        sort_ids_change = sorted(range(len(state_change_accuracies_per_depth[layer])),key = state_change_accuracies_per_depth[layer].__getitem__)
        state_change_accuracies_per_depth_sorted[layer] = [state_change_accuracies_per_depth[layer][id_event] for id_event in sort_ids_change][-top:]
        state_change_repetitions_best_accuracies[layer] = [all_repetitions[layer_id][id_event] for id_event in sort_ids_change][-top:]
        
        if top != 50:
            # DATA IS NO LONGER PAIRED!
            state_change_accuracies_per_depth[layer] = state_change_accuracies_per_depth_sorted[layer]
            event_accuracies_per_depth[layer] = event_accuracies_per_depth_sorted[layer]
                
    # do anovas
    stats = {}
    
    for accuracy_type in ["per_event", "per_state_change"]:
        if accuracy_type == "per_event":
            accuracies = event_accuracies_per_depth
        else:
            accuracies = state_change_accuracies_per_depth
        
        stats[accuracy_type] = {}
        stats[accuracy_type]["anova"] = {}
        stats[accuracy_type]["variance"] = {}

        if (not any(condition_list) or all(el=='16nodes' for el in condition_list)) and len(condition_list) > 1:
            eval_string = "kruskal(" 
            for key in list(accuracies.keys()):
                eval_string = eval_string + "accuracies['" + key + "'],"
            eval_string = eval_string[:-1] + ")"
            [stats[accuracy_type]["anova"]["F"],stats[accuracy_type]["anova"]["p"]]= eval(eval_string)
            print(stats[accuracy_type]["anova"])
            
            p_vals = []
            eval_string = eval_string.replace("kruskal","levene").replace(")",", center='median')")
            stats[accuracy_type]["variance"]["levene"] = eval(eval_string)
            for key_id, key in enumerate(list(accuracies.keys())[:-1]):
                stats[accuracy_type]["variance"][key] = {}
                for key_id_2 in np.arange(key_id + 1,len(list(accuracies.keys()))):
                    stats[accuracy_type]["variance"][key][list(accuracies.keys())[key_id_2]],p = levene(accuracies[key],accuracies[list(accuracies.keys())[key_id_2]])
                    p_vals.append(p)
                    print(stats[accuracy_type]["variance"][key][list(accuracies.keys())[key_id_2]])
            _,stats[accuracy_type]["variance"]["posthoc_corrected"],_,_=statsmodels.stats.multitest.multipletests(p_vals, method='holm-sidak',alpha = 0.05)
            
            print(stats[accuracy_type]["variance"]["levene"] )
            print(stats[accuracy_type]["variance"]["posthoc_corrected"])
                    
                    
            
            # post hocs    
            # posthoc dunn test, with correction for multiple testing
            eval_string = "posthoc_dunn([" 
            for key in list(accuracies.keys()):
                eval_string = eval_string + "accuracies['" + key + "'],"
            eval_string = eval_string[:-1] + "],p_adjust='holm-sidak')"
            
            stats[accuracy_type]["post_hoc"] = eval(eval_string)
            print(accuracy_type,':', stats[accuracy_type]["post_hoc"])
            
            
            eval_string = "kruskal(" 
            for key in list(accuracies.keys()):
                eval_string = eval_string + "accuracies['" + key + "'],"
            eval_string = eval_string[:-1] + ")"
            [stats[accuracy_type]["anova"]["F"],stats[accuracy_type]["anova"]["p"]]= eval(eval_string)
            print(stats[accuracy_type]["anova"])
            
            # post hocs    
            # posthoc dunn test, with correction for multiple testing
            eval_string = "posthoc_dunn([" 
            for key in list(accuracies.keys()):
                eval_string = eval_string + "accuracies['" + key + "'],"
            eval_string = eval_string[:-1] + "],p_adjust='holm-sidak')"
            
            stats[accuracy_type]["post_hoc"] = eval(eval_string)
            print(accuracy_type,':', stats[accuracy_type]["post_hoc"])
            
        elif len(condition_list) == 1:
            pass
            
        # comparisons with control conditions are paired with its "regular indRNN" version
        # I'm not doing anything else than top 50 here, so data are still paired
        else:
            
            stats[accuracy_type]["MannWU"] = {}
            stats[accuracy_type]["Wilcoxon"] = {}
            stats[accuracy_type]['shapiro'] = {}

            acc_regular = accuracies["5 layers"]
            stats[accuracy_type]["shapiro"]["regular"] = shapiro(acc_regular)
           
            # compare 5 layer network with not recurrent
            stats[accuracy_type]["MannWU"]["NN"] = mannwhitneyu(acc_regular,accuracies["no_recurrency"])
            stats[accuracy_type]["shapiro"]["NN"] = shapiro(accuracies["no_recurrency"])
            print('NN:',stats[accuracy_type]["MannWU"]["NN"])
    
            # compare 5 layer network with 16 nodes per layer
            stats[accuracy_type]["MannWU"]["16nodes"] = mannwhitneyu(acc_regular,accuracies["16nodes"])
            stats[accuracy_type]["shapiro"]["16nodes"] = shapiro(accuracies["16nodes"])
            print('16:',stats[accuracy_type]["MannWU"]["16nodes"])
            
            # compare 5 layer network with network trained on shuffled data
            stats[accuracy_type]["Wilcoxon"]["shuffled"] = wilcoxon(acc_regular,accuracies["shuffled"])
            stats[accuracy_type]["shapiro"]["shuffled_diff"] = shapiro(np.array(acc_regular) - np.array(accuracies["shuffled"]))
            print('shuffled:',stats[accuracy_type]["Wilcoxon"]["shuffled"])
            
            # compare 5 layer network with network trained on shuffled data
            stats[accuracy_type]["Wilcoxon"]["init"] = wilcoxon(acc_regular,accuracies["init"])
            stats[accuracy_type]["shapiro"]["init_diff"] = shapiro(np.array(acc_regular) - np.array(accuracies["init"]))
            print('init:',stats[accuracy_type]["Wilcoxon"]["shuffled"])
     

    # plot including scatter points of data
    violin_list = [(event_accuracies_per_depth[key][rep], "event", key) for key in event_accuracies_per_depth.keys() for rep in range(len(event_accuracies_per_depth[key]))]
    violin_list.extend([(state_change_accuracies_per_depth[key][rep], "state change", key) for key in event_accuracies_per_depth.keys() for rep in range(len(event_accuracies_per_depth[key]))])
    plot_df = pd.DataFrame(violin_list,columns=("accuracy","accuracy type", "condition"))
    print(plot_df)
    violin = sns.violinplot(data=plot_df, x="accuracy type", y="accuracy", hue = "condition",hue_order = event_accuracies_per_depth.keys(),color = [0,0,0],cut=0,inner='quartile',linecolor="white",scale="width", width = 0.8)
    
    for p in violin.lines[-3*len(all_repetitions)*2:]:
        p.set_linestyle('-')
        p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
        p.set_color('white')  # Sets the color of the quartile lines
        p.set_alpha(1)            
 
    for i, v in enumerate(violin.findobj(PolyCollection)):
       if i < len(event_accuracies_per_depth):
           v.set_facecolor('0.8')
       else:
           v.set_facecolor('0')
           
           
    grouped_df = plot_df.groupby(['accuracy type', 'condition'],sort=False)
    print(grouped_df)
    mean = grouped_df['accuracy'].mean().to_list()
    median = grouped_df['accuracy'].median().to_list()
    Q25 = grouped_df['accuracy'].quantile(.25).to_list()
    Q75 = grouped_df['accuracy'].quantile(.75).to_list()
    stats['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]
    print(stats['median_info'])
    x_values_summary_stats = [np.where(np.asarray(["event", "state change"]) == plotted_acc)[0][0] for plotted_acc in grouped_df.mean().index.get_level_values(0)]
   
    if len(event_accuracies_per_depth) == 5:
        x_values_violins = [-0.32, -0.16, 0, 0.16, 0.32]
    elif len(event_accuracies_per_depth) == 4:
        x_values_violins = [-0.3, -0.1, 0.1, 0.3]
    elif len(event_accuracies_per_depth) == 1:
        x_values_violins = [0]

    previous_x_value = x_values_summary_stats[0]
    layer_id = 0
    for x_id, x_value in enumerate(x_values_summary_stats):
        if x_value != previous_x_value:
            layer_id = 0
        previous_x_value = x_value
        x_values_summary_stats[x_id] = x_value + x_values_violins[layer_id]
        layer_id += 1
    
    plt.errorbar(x_values_summary_stats,median,fmt='.',color='white',capsize=0) 
    plt.errorbar(x_values_summary_stats,mean,fmt='.',color='red',capsize=0) 

    
    plt.ylim(0,1)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    
    pickle_info = {'repetitions':repetitions_per_layer, 'event_accuracies_per_depth_sorted':event_accuracies_per_depth_sorted, 'event_repetitions_best_accuracies':event_repetitions_best_accuracies, 'state_change_accuracies_per_depth_sorted':state_change_accuracies_per_depth_sorted, 'state_change_repetitions_best_accuracies':state_change_repetitions_best_accuracies, 'stats':stats}
    overwrite = False
    save_stats(pickle_info, {'results_section':'accuracy_NEW16nodesTRY','top':top,'depths':collect_depths, 'conditions': condition_list,'incl':'Levene'},overwrite=overwrite)
    accuracy_path, fig_file_name = save_stats([], {'results_section':'accuracy_NEW16nodesTRY','top':top,'depths':collect_depths, 'conditions': condition_list,"scale":"width"},overwrite=overwrite)
    if os.path.exists(os.path.join(accuracy_path, fig_file_name)) and overwrite == False:
        print('file ', fig_file_name, ' already exists. Not overwriting')
    else:    
        plt.savefig(os.path.join(accuracy_path,fig_file_name))
    
    plt.show()
    
    
def compare_variance_accuracy_maps(results_dir, all_repetitions, assessed_per="event",condition_list = [],top = 50):
    '''
    creates accuracy heatmaps for each duration-period combination and compares variance within these spaces
    between network depths
    
    results_dir: directory in which the  runrepetitions of the models are stored
    all_repetitions: list of repetition information you want to assess. Different network depths are stacked on each other
    assessed_per: variable over which average summed activation is calculated (event, state_change, movie; string)
    control_list: list of control conditions, like no_recurrency, shuffled, 16nodes, init, None. Same length as the stacked repetitions
    
    '''

    ev_acc = []
    sc_acc = []
    var_acc = []
    all_layer_names = []
    
    fig, axs = plt.subplots(2, len(all_repetitions),squeeze = False)
    plt.subplots_adjust(wspace=0.1,hspace = 0)
    
    for network_depth_id, repetitions_per_layer in enumerate(all_repetitions):      
        if len(condition_list) >0:
            control_condition = condition_list[network_depth_id]
        else:
            control_condition = None
        
        repetitions = []
        event_accuracies = [];
        event_values = [];
        sc_values = [];
        variance_heatmap = [];
        
        for repetition in repetitions_per_layer:
            if len(repetition) == 1:
                repetition = repetition[0]
                
            repetitions.append(repetition)
               
            # get accuracies all timing presentations
            if control_condition == "special_condition":
                evaluate_special_conditions(results_dir, repetition, do_plotting = False)
            else:
                evaluate_net(results_dir, repetition, assessed_per,do_plotting = False,control_condition=control_condition)
            _, test_results_path, _, _, _, _, _, _ = get_result_paths(results_dir, repetition,control_condition=control_condition) 
            with open(test_results_path, 'rb') as f:
                # get concatenated losses and fit_labels
                _, _, _, _, _, _, _, labels, _,_, acc_proportions_event, acc_proportions_state_change = pickle.load(f)

                
            if len(event_accuracies)==0:
                event_accuracies = acc_proportions_event
                change_accuracies = acc_proportions_state_change

                
                frame = plot_performance_heatmap_size_std(acc_proportions_state_change,labels,do_plotting = False)
                variance_heatmap = [np.var(frame['value'])]
                if len(repetitions_per_layer) == 1:
                    event_accuracies = [acc_proportions_event]
                    change_accuracies =[acc_proportions_state_change]
            else:
                event_accuracies = np.vstack((event_accuracies, acc_proportions_event))
                change_accuracies = np.vstack((change_accuracies, acc_proportions_state_change))
                frame = plot_performance_heatmap_size_std(acc_proportions_state_change,labels,do_plotting = False)
                variance_heatmap.append(np.var(frame['value']))
                
        if control_condition == "special_condition":
            amount_layers = repetition['special_condition']
            amount_layers = amount_layers.replace('_', ' ')
        elif control_condition == None:
            amount_layers = str(repetition['num_layers']) + ' layers'
        else:
            amount_layers = control_condition
             
        sort_ids_change = sorted(range(len(np.mean(change_accuracies,axis = 1))),key = np.mean(change_accuracies,axis = 1).__getitem__)
        change_accuracies_sorted = [change_accuracies[:][id_event] for id_event in sort_ids_change][-top:]
        state_change_repetitions_best_accuracies = [repetitions[id_event] for id_event in sort_ids_change][-top:]        
        sort_ids_event = sorted(range(len(np.mean(event_accuracies,axis = 1))),key = np.mean(event_accuracies,axis = 1).__getitem__)
        event_accuracies_sorted = [event_accuracies[:][id_event] for id_event in sort_ids_event][-top:]
        event_repetitions_best_accuracies = [repetitions[id_event] for id_event in sort_ids_event][-top:]        
        if top != 50:
            variance_heatmap = [variance_heatmap[id_event] for id_event in sort_ids_change][-top:]
      
        var_acc.append(variance_heatmap)
        all_layer_names.append(amount_layers)
        
        df_event = plot_performance_heatmap_size_std(np.mean(event_accuracies_sorted,axis=0), labels,np.std(event_accuracies_sorted,axis=0),ax=axs[0,network_depth_id])
        df_sc = plot_performance_heatmap_size_std(np.mean(change_accuracies_sorted,axis=0),labels,np.std(change_accuracies_sorted,axis=0),ax=axs[1,network_depth_id])

        durations = df_event.columns.to_list()
        for per in np.arange(df_event.shape[0]):
            for dur in np.arange(df_event.shape[1]):
                event_values.append(df_event[durations[dur]].iat[per])
                sc_values.append(df_sc[durations[dur]].iat[per])
                
        ev_acc.append(event_values)
        sc_acc.append(sc_values)

  
    if "special_condition" not in condition_list and len(condition_list) > 1:
        eval_string = "kruskal(" 
        for var_list in var_acc:
            eval_string = eval_string + str(var_list) + ","
        eval_string = eval_string[:-1] + ")"
        stats_var = eval(eval_string)
              
        eval_string = eval_string.replace("kruskal(","posthoc_dunn([").replace(")","],p_adjust='holm-sidak')")
        post_hoc_var = eval(eval_string)
        print(stats_var)
        print(post_hoc_var)
    
      
        pickle_info = {'ev_timing_values':ev_acc, 'sc_timing_values':sc_acc, 'variances':var_acc, 'layers':all_layer_names, 'kruskal':stats_var, 'posthoc':post_hoc_var,'sc_rep':state_change_repetitions_best_accuracies,'sc_acc':change_accuracies_sorted,'ev_acc':event_accuracies_sorted,'ev_rep':event_repetitions_best_accuracies}
    else:
        pickle_info = []
    
    overwrite = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    accuracy_path, fig_file_name = save_stats(pickle_info, {'results_section':'variance_acc','depths':all_layer_names, 'conditions': condition_list,'top': top},overwrite=overwrite)
    if os.path.exists(os.path.join(accuracy_path, fig_file_name)) and overwrite == False:
        print('file ', fig_file_name, ' already exists. Not overwriting')
    else:    
        plt.savefig(os.path.join(accuracy_path,fig_file_name))
    
    plt.show()
    return pickle_info

    


    
def do_accuracy_evaluations(results_dir, repetitions_all_depths, elem_idx=118, condition_list = None):

    '''
    Selects the data for which to do analyses and runs the analyses. 
    I have played around (commented and uncommented) with these a lot, depending on what I needed
    '''

    if condition_list is None:
        condition_list = []

    # initially, parameter-matched were seen as "normal", later, layer-size-matched (16nodes) could also be normal
    # so I 
    # select relevant repetitions to plot loss
    normal_ids = np.where(np.array(condition_list) == None)[0]
    repetitions_normal = [repetitions_all_depths[i] for i in normal_ids]
    condition_list_normal = [condition_list[i] for i in normal_ids]
    pickle_info = plot_loss_functions_all_depths(results_dir, repetitions_normal,control_condition_list = condition_list_normal)
    
    # select (other) relevant repetitions to plot loss
    normal_id_5_layer = [normal_id for normal_id in normal_ids if repetitions_all_depths[normal_id][0][0]["num_layers"] == 5]
    normal_id_5_layer.extend(np.where(np.logical_or(np.logical_or(np.logical_or(np.array(condition_list) == "init",np.array(condition_list) == "no_recurrency"),np.array(condition_list) == "shuffled"),np.array(condition_list) == "16nodes"))[0])
                             
    repetitions_5_layers = [repetitions_all_depths[i] for i in normal_id_5_layer]
    condition_list_5_layers = [condition_list[i] for i in normal_id_5_layer]
    pickle_info = plot_loss_functions_all_depths(results_dir, repetitions_5_layers,control_condition_list = condition_list_5_layers)

    # make figure 3
    combined_grid_plot(results_dir, elem_idx, repetitions_all_depths, control_condition_list = condition_list)

    # make accuracy stats and plots per-event and per-state change
    compare_accuracy_model_depths(results_dir, repetitions_normal, condition_list = condition_list_normal)
    compare_accuracy_model_depths(results_dir, repetitions_normal, condition_list = condition_list_normal,top = 25)
    

    normal_id_5_layer = [normal_id for normal_id in normal_ids if repetitions_all_depths[normal_id][0][0]["num_layers"] == 5]
    normal_id_5_layer.extend(np.where(np.array(condition_list) == "init")[0])
    normal_id_5_layer.extend(np.where(np.array(condition_list) == "shuffled")[0])
    normal_id_5_layer.extend(np.where(np.array(condition_list) == "no_recurrency")[0])
    normal_id_5_layer.extend(np.where(np.array(condition_list) == "16nodes")[0])
    repetitions_5_layers = [repetitions_all_depths[i] for i in normal_id_5_layer]
    condition_list_5_layers = [condition_list[i] for i in normal_id_5_layer]
    pickle_info = compare_accuracy_model_depths(results_dir, repetitions_5_layers, condition_list = condition_list_5_layers)
    
    
    pickle_info = compare_variance_accuracy_maps(results_dir,repetitions_all_depths,condition_list= condition_list)
    pickle_info = compare_variance_accuracy_maps(results_dir,repetitions_all_depths,condition_list= condition_list,top=25)
    
    return pickle_info

  