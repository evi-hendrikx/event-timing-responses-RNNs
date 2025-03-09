import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scikit_posthocs import posthoc_dunn
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib
from scipy.stats import shapiro
import itertools
from matplotlib.collections import PolyCollection

from model_fitting import run_fit,tuned_function,find_peak_tuning
from utility import save_stats
import config_file as c


def load_fit_data(results_dir,repetitions,assessed_per="event",x0=None,y0=None,control_condition_list=[], threshold = 0.2,to_assess = "prop"):
    '''
    loads fitting of monotonic and tuned models into pandas dataframe including parameters and fits
    
    results_dir: directory in which the  runrepetitions of the models are stored
    repetitions: list of repetition information you want to assess. Different network depths are stacked on each other
    assessed_per: variable over which average summed activation is calculated (event, state_change, movie; string)
    x0: x variable in the timing space (duration, occupancy; string; list of strings)
    y0: y variable in the timing space (period, isi; string; list of strings)
    control_condition_list: list of control conditions, like no_recurrency, shuffled, 16nodes, init, None. Same length as the stacked repetitions
    threshold: threshold for which the nodes are considered tuned or monotonic
    to_assess: variable you want to assess. Can be a parameter (as named in tuned_function or monotonic_function), ve, or prop
    
    returns 
    total_df: dataframe with proportions and ve info of tuned & monotonic model fits over layers
    '''
    print(x0, y0)
    if x0 is None:
        x0 = ["duration", "ISI"]
    if y0 is None:
        y0 = ["period", "period"]
        
    depth_arr,layer_arr,model_arr,prop_arr,ve_arr,rep_arr, cond_arr,node_arr,timing_type_arr,assessed_arr = [],[],[],[],[],[],[],[],[],{}
    timing_spaces = ['_' + x0[id_element] + '_' + y0[id_element] for id_element in range(len(x0))]
    tuned_params = {}
    
    if not isinstance(to_assess,list):
        to_assess = [to_assess]
    if not isinstance(x0,list):
        x0 = [x0]
        y0 = [y0]
     
    for network_depth_id, network_depth in enumerate(repetitions):
        print('network depth:', network_depth_id)
        
        if len(control_condition_list) == 0:
            control_condition = None
        else:
            control_condition = control_condition_list[network_depth_id]
            
        for repetition in network_depth:

            if len(repetition)==1:
                repetition= repetition[0]
            amount_layers = repetition["num_layers"]    
            rep_id = repetition["counter"]
            
            # combining multiple types of fits
            for timing_space_id,timing_space in enumerate(timing_spaces):
                mono_fits, tuned_fits = run_fit(repetition, assessed_per=assessed_per,x0=x0[timing_space_id],y0=y0[timing_space_id],control_condition=control_condition)
                first_cols = mono_fits.iloc[:,0:3]
                mono_fits_reduced = mono_fits.drop(columns = ['node','layer','split'])
                mono_fits_reduced = mono_fits_reduced.add_suffix(timing_space)
                tuned_fits_reduced = tuned_fits.drop(columns = ['node','layer','split'])
                tuned_fits_reduced = tuned_fits_reduced.add_suffix(timing_space)
                tuned_params[timing_space] = list(filter(lambda col: 'VE' not in col, tuned_fits_reduced.columns.values))

                # blend dataframes
                if timing_space_id == 0:
                    all_fits = pd.concat([first_cols, mono_fits_reduced, tuned_fits_reduced], axis='columns')
                else:
                    all_fits = pd.concat([all_fits,mono_fits_reduced, tuned_fits_reduced], axis='columns')

            for layer in range(amount_layers + 1):
               
                # per layer or for all layers together (in last loop)
                if layer == amount_layers:
                    layer_fits = all_fits
                else:
                    layer_fits =  all_fits.loc[all_fits['layer'] ==layer]
                    layer_fits = layer_fits.reset_index(drop = True)
                
                # per timing space indicate what invalid responses are
                invalid_tuned = {}
                invalid_mono = {}
                for timing_space in timing_spaces:
                    # tuned is invalid with huge sigmas or a tiny tiny slope 
                    # (positive betas don't allow below 1e-08, so shouldn't be an issue)
                    invalid_tuned[timing_space] = np.logical_or(np.logical_and(layer_fits['sigma_x0' + timing_space]>9.99,layer_fits['sigma_y0' + timing_space]>9.99),
                        np.logical_and(layer_fits['tuned_slope' + timing_space]>-1e-08,layer_fits['tuned_slope' + timing_space]<1e-08))

                    # mono is invalid if both factors are constant or have a tiny tiny slope 
                    invalid_mono[timing_space] = np.logical_and(np.logical_and(layer_fits['mono_slope' + timing_space] * layer_fits['ratio_x0' + timing_space] >-1e-08, \
                              layer_fits['mono_slope' + timing_space] * layer_fits['ratio_x0' + timing_space] <1e-08), \
                            np.logical_and(layer_fits['mono_slope' + timing_space] * layer_fits['ratio_y0' + timing_space] >-1e-08, \
                              layer_fits['mono_slope' + timing_space] * layer_fits['ratio_y0' + timing_space] <1e-08))     
               
                # make selection of valid, invalid, and out of range responses
                # empty lists for all nodes in layer
                mono_classified = [False] * len(layer_fits)
                tuned_classified = [False] * len(layer_fits)
                mixed_classified = [False] * len(layer_fits)
                only_intercept = [False] * len(layer_fits)
                timing_type = [''] * len(layer_fits)
                
                # select model with highest cross-validated VE
                VEs = layer_fits.filter(like='cv_VE')
                highest_VE = VEs.idxmax(axis=1)
                
                for row_id, winner in enumerate(highest_VE):
                    
                    try:
                        timing_space = winner.split('cv_VE')[1]
                    except:
                        # sometimes all values are nan (when node has no variance
                        # for example)
                        continue
                    
                    timing_type[row_id] = timing_space
                    above_threshold = VEs.at[row_id,winner] > threshold
                    
                    
                    if "tuned" in winner and above_threshold:
    
                        # find out where the peak of the tuned function is and whether it's outside of the presented range
                        tuned_param_values = [layer_fits.at[row_id,param] for param in tuned_params[timing_space]]
                        response_labels_x = np.arange(-0.05,1.1,0.05)
                        response_labels_y = np.arange(-0.05,1.1,0.05)
                        # little hack to not get a problem with dividing by 0
                        response_labels_x[response_labels_x==0] = 1e-8
                        response_labels_y[response_labels_y==0] = 1e-8
                        
                        combis_x = [x for x in response_labels_x for y in response_labels_y]
                        combis_y = [y for x in response_labels_x for y in response_labels_y]
                        response_labels = [combis_x, combis_y]
                        tuned_prediction = tuned_function(response_labels, *tuned_param_values)
                                              
                        guess_peak_x = combis_x[np.nanargmax(tuned_prediction)]
                        guess_peak_y = combis_y[np.nanargmax(tuned_prediction)]
                        
                        # make peak guess more precise
                        peak_x, peak_y = find_peak_tuning(guess_peak_x, guess_peak_y, *tuned_param_values)
                        if timing_space == "_duration_period" or timing_space == "_ISI_period":
                            out_range = (peak_x>0.95) or (peak_x<0.05) or (peak_y>1) or (peak_y<0.1) or (peak_x > peak_y - 0.05)
                        elif timing_space == "_duration_ISI":
                            out_range = (peak_x>0.95) or (peak_x<0.05) or (peak_y>0.95) or (peak_y<0.05) or (peak_x + peak_y > 1)
                        elif timing_space == "_occupancy_period" or timing_space == "_occupancyISI_period" :
                            out_range = (peak_x>0.95) or (peak_x<0.05) or (peak_y>1) or (peak_y<0.1) or (peak_x > 1-(0.05/peak_y))
                        
                        # check if it is valid
                        if invalid_tuned[timing_space][row_id] == False and out_range == False:
                            tuned_classified[row_id] = True
                        elif invalid_tuned[timing_space][row_id] == False and out_range == True:
                            mixed_classified[row_id] = True
                        else:
                            only_intercept[row_id] = True
                    
                    elif "mono" in winner and above_threshold:
                        if invalid_mono[timing_space][row_id] == False:
                            mono_classified[row_id] = True
                        else:
                            only_intercept[row_id] = True
                    
                    
                for model_name in ["mono","tuned","mixed"]:
                    # gather general information
                    rep_arr.append(rep_id)            
                    depth_arr.append(amount_layers) 
                    model_arr.append(model_name)
                    if control_condition != None:
                        cond_arr.append(control_condition)
                    else:
                        cond_arr.append("regular")        
                  
                    # per layer or for all layers together (in last loop)
                    if layer == amount_layers and to_assess != ["classifications"]:
                            # classifications will have all layers added,
                            # happens later
                            layer_arr.append("overall")
                    elif layer != amount_layers:
                        layer_arr.append(layer)
                    
                        
                    classified_string = model_name + '_classified'
                    timing_type_arr.append(np.array(timing_type)[eval(classified_string)])
                    
                    if to_assess != None and to_assess != ["prop"] and to_assess != ["ve"]:
                       if to_assess != ["classifications"]:
                            for t_a in to_assess:
                                if t_a != 'node' and t_a != 'split':
                                    relevant_parameter = np.array([float(layer_fits.loc[tt_id,t_a + tt]) if len(tt)>0 else np.nan for tt_id, tt in enumerate(timing_type)])
                                else:
                                    relevant_parameter = np.array([float(layer_fits.loc[tt_id,t_a]) if len(tt)>0 else np.nan for tt_id, tt in enumerate(timing_type)])
                                if not t_a in assessed_arr.keys():
                                    assessed_arr[t_a] = []
                                assessed_arr[t_a].append(relevant_parameter[eval(classified_string)])
                       else:
                            node_arr.append(layer_fits['node'][eval(classified_string)].values)
                            if layer == amount_layers:
                                layer_arr.append(layer_fits['layer'][eval(classified_string)].values)
                                
                    else:
                        # also tried with proportion of all classified nodes. Gives similar results
                        prop_arr.append(sum(eval(classified_string))/len(layer_fits))

                        if model_name == "mono":
                            relevant_ve = np.array([float(layer_fits.loc[tt_id,'mono_cv_VE' + tt]) if len(tt)>0 else np.nan for tt_id, tt in enumerate(timing_type)])
                            ve_arr.append(relevant_ve[eval(classified_string)])
                        else:
                            relevant_ve = np.array([float(layer_fits.loc[tt_id,'tuned_cv_VE' + tt]) if len(tt)>0 else np.nan for tt_id, tt in enumerate(timing_type)])
                            ve_arr.append(relevant_ve[eval(classified_string)])   
                
                        
    
    if to_assess != ["prop"] and to_assess != ["ve"]:
        if to_assess != ["classifications"]:
            assessed_par_values = list(assessed_arr.values())
            assessed_par = list(assessed_arr.keys())
            total_df = pd.DataFrame(list(zip(depth_arr,cond_arr,layer_arr,model_arr,rep_arr,*assessed_par_values,timing_type_arr)),columns = ['depth','condition','layer','model','rep',*assessed_par,'timing_type'])
        else:
            total_df = pd.DataFrame(list(zip(depth_arr,cond_arr,layer_arr,model_arr,rep_arr,node_arr,timing_type_arr)),columns = ['depth','condition','layer','model','rep', 'node','timing_type'])
    else:
        total_df = pd.DataFrame(list(zip(depth_arr,cond_arr,layer_arr,model_arr,prop_arr,ve_arr,rep_arr,timing_type_arr)),columns = ['depth','condition','layer','model','prop','ve','rep','timing_type'])
    print(total_df)
    return total_df


def kruskal_and_posthocs(total_df, dependent_var, between_what, additional_factors = None):
    
    '''
    total_df: pandas data frame for which you want to do the stats (like the one made by load_fit_data)
    dependent_var: dependent variables you want to compare in this test
    between_what: column name of groups you want to compare
    additional_factors: other including factors. dict where key is the column and value is a list with the to be included value
    
    returns 
    stats: dictionary with outcome statistics kruskal & posthoc Dunns; tested data with labels; input information
    '''
    if additional_factors is None:
        additional_factors = {}
        
    stats = {}
    stats['input_args'] = {}
    stats['input_args']['dependent_var'] = dependent_var
    stats['input_args']['independent_var'] = between_what
    stats['input_args']['selection'] = additional_factors
    
    stats["shapiro"] = {}
    
    group_by = [between_what]
    if len(additional_factors) > 0:
        for additional_key in additional_factors.keys():
            group_by.extend([additional_key]) 
    else:
        group_by = group_by[0]
        
    get_whats = [new_group[dependent_var].values for key,new_group in total_df.groupby(group_by)]
    get_whats_titles = [key for key,new_group in total_df.groupby(group_by)]

    # if we only want specific groups (e.g., model == "mono")
    if len(additional_factors) > 0:
        new_get_whats = []
        new_get_whats_titles = []
        
        for tup_id, tup in enumerate(get_whats_titles):  
            # between factor has to remain
            new_tup = [tup[0]]
            include_point = True
            for avs_id, additional_factor_values in enumerate(additional_factors.values()):
                if tup[avs_id + 1] not in additional_factor_values:
                    include_point = False
                else:
                    new_tup.append(tup[avs_id+1])
                    
            
            if include_point == True:
                new_get_whats.append(get_whats[tup_id])
                new_get_whats_titles.append(tuple(new_tup))
        get_whats = new_get_whats
        get_whats_titles = new_get_whats_titles
    
    stats['data'] = get_whats
    stats['group_names'] = get_whats_titles
    
    if len(get_whats) > 1:
        for get_what_id,get_what in enumerate(get_whats):
            # print(get_whats_titles[get_what_id])
            # print(get_what)
            stats["shapiro"][get_whats_titles[get_what_id]] = shapiro(get_what)
            print(stats["shapiro"])

        # kruskal test
        try:
            stats['kruskal'] = kruskal(*get_whats)
            print(stats['kruskal'])
        except:
            print('kruskal impossible')
    if len(get_whats) > 2:
        # posthocs
        stats['Dunn_posthoc'] = posthoc_dunn(get_whats,p_adjust='holm-sidak')
        print(stats['Dunn_posthoc'])
        
    
    return stats

def plot_violins(plot_df, stats, hue=None,hue_order = None,y_x_ratio = 1,save_path = ''):
    
    '''
    make violin plots from dataframe
    plot_df: pandas data frame with selection of data you want to plot
    stats: dictionary with outcome statistics kruskal & posthoc Dunns; tested data with labels; input information    between_what: column name of groups you want to compare
        (created by kruskal_and_posthocs)
    hue: string, column that you want separate violins for
    y_x_ratio: if you want the y and x axis to be different sizes
        
    returns 
    
    '''
    matplotlib.rcParams['pdf.fonttype'] = 42

    if hue == None:
        hues = ['None']
        grouped_df = plot_df.groupby([stats['input_args']['independent_var']],sort=False)
    else:
        grouped_df = plot_df.groupby([hue, stats['input_args']['independent_var']],sort=False)
        hues = np.unique(plot_df[stats['input_args']['independent_var']])
    
    plt.figure(figsize=(5*1.5,5*1.5))    
   
    
    if not hue_order:
        violin = sns.violinplot(data=plot_df, x=hue, y=stats['input_args']['dependent_var'],color=[0,0,0],hue= stats['input_args']['independent_var'],cut=0,inner='quartile',linecolor="white",scale = "width",dodge=True,width=0.8)
    else:
        violin = sns.violinplot(data=plot_df, x=hue, y=stats['input_args']['dependent_var'],color=[0,0,0],hue= stats['input_args']['independent_var'],hue_order=hue_order,cut=0,inner='quartile',linecolor="white",scale = "width",dodge=True,width=0.8)

   
    mean = grouped_df[stats['input_args']['dependent_var']].mean().to_list()
    median = grouped_df[stats['input_args']['dependent_var']].median().to_list()
    Q25 = grouped_df[stats['input_args']['dependent_var']].quantile(.25).to_list()
    Q75 = grouped_df[stats['input_args']['dependent_var']].quantile(.75).to_list()
    median_info = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]
    
    x_values_summary_stats = [np.where(np.array(['mono','mixed','tuned']) == plotted_model)[0][0] for plotted_model in grouped_df.mean().index.get_level_values(0)]
    
    x_values_5_violins = [-0.32, -0.16, 0, 0.16, 0.32]
    previous_x_value = x_values_summary_stats[0]
    layer_id = 0
    for x_id, x_value in enumerate(x_values_summary_stats):
        if x_value != previous_x_value:
            layer_id = 0
        previous_x_value = x_value
        x_values_summary_stats[x_id] = x_value + x_values_5_violins[layer_id]
        layer_id += 1
    
  
    for p in violin.lines[-3*len(x_values_summary_stats):]:
        p.set_linestyle('-')
        p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
        p.set_color('white')  # Sets the color of the quartile lines
        p.set_alpha(1)            
        
    for i, v in enumerate(violin.findobj(PolyCollection)):
        if i < layer_id:
            v.set_facecolor('0.8')
        elif i < layer_id * 2:
            v.set_facecolor('0.4')
        else:
            v.set_facecolor('0')
        
    plt.errorbar(x_values_summary_stats,median, fmt='.',color='white',zorder=4)
    plt.errorbar(x_values_summary_stats,mean,fmt='.',color='red',zorder=4)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylim(0,1)
    overwrite = False
    if len(save_path) > 0:
        if os.path.exists(os.path.join(save_path)) and overwrite == False:
            print('file ', save_path, ' already exists. Not overwriting')
        else:    
            plt.savefig(os.path.join(save_path))
    
    plt.show()
    
    return median_info
   

    
def stats_fits_between_depths(results_dir,repetitions,assessed_per="event",x0=None,y0=None,control_condition_list=[], threshold = 0.2, model_list = ["mono","mixed","tuned"], last_layer = True, to_assess= "prop",first_layer= False):
    '''
    Stats and plots between different network depths
    or different conditions of the 5 layer network

    Parameters
    ----------
   
    results_dir: directory in which the  runrepetitions of the models are stored
    repetitions: list of repetition information you want to assess. Different network depths are stacked on each other
    assessed_per: variable over which average summed activation is calculated (event, state_change, movie; string)
    x0: x variable in the timing space (duration, occupancy, ISI; string or list of strings)
    y0: y variable in the timing space (period, ISI; string or list of strings)
    control_condition_list: list of control conditions, like no_recurrency, shuffled, 16nodes, init, None. Same length as the stacked repetitions
    threshold: threshold for which the nodes are considered tuned or monotonic
 

    Returns
    -------
    None.

    '''
    
    if x0 is None:
        x0 = ["duration", "ISI"]
    if y0 is None:
        y0 = ["period", "period"]
    
    to_save_stats = {}
    to_save_stats[to_assess] = {}
    
    condition = control_condition_list[0]
    if condition == None:
        condition = 'regular'

    
    # proportions
    total_df = load_fit_data(results_dir, repetitions,assessed_per,x0,y0,control_condition_list, threshold,to_assess=to_assess)
    if last_layer:
        total_df = total_df[total_df['layer'] == total_df['depth']-1]
        print(total_df)
    
    elif first_layer:
        total_df = total_df[total_df['layer'] == 0]
        print(total_df)
        
    if to_assess != "prop":
        total_df = total_df.explode(to_assess)
        total_df = total_df[total_df['prop']!=0]

    amount_layers = []
    for network_depth in repetitions:
        amount_layers.append(network_depth[0][0]["num_layers"])
  
    for model in model_list:
        if last_layer or first_layer:
            stats = kruskal_and_posthocs(total_df,to_assess, 'depth',{'condition':[condition],'model':[model]})
        else:
            stats = kruskal_and_posthocs(total_df,to_assess, 'depth',{'condition':[condition],'model':[model],'layer':['overall']})
        to_save_stats[to_assess][model] = stats
        
        # create dataframe with same selection as stats for plotting
        list_tuples = []
        for tup_id, tup in enumerate(stats['group_names']):
            for datapoint in stats['data'][tup_id]:
                list_tuples.append(tup + tuple([datapoint]))
                
        # combine dataframes for monotonic and tuned models
        if model == "mono":
            plot_df = pd.DataFrame.from_records(list_tuples,columns=[stats['input_args']['independent_var'], *list(stats['input_args']['selection'].keys()),stats['input_args']['dependent_var']])
        else:
            new_df = pd.DataFrame.from_records(list_tuples,columns=[stats['input_args']['independent_var'], *list(stats['input_args']['selection'].keys()),stats['input_args']['dependent_var']])
            plot_df = pd.concat([plot_df, new_df])
  
    overwrite = False
    fit_stat_path, fig_file_name = save_stats([], {'results_section':to_assess +'PerDepth','depths':amount_layers, 'conditions': control_condition_list,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'lastLayer':last_layer,"scale":"width",'threshold': threshold},overwrite=overwrite)
    to_save_stats['median_info'] = plot_violins(plot_df, stats,hue="model",hue_order = list(range(min(amount_layers),min(amount_layers)+5)), save_path = os.path.join(fit_stat_path, fig_file_name))
    
    
    pickle_info = {'total_df': total_df, 'stats': to_save_stats, 'plot_df': plot_df}
    overwrite = False
    save_stats(pickle_info, {'results_section':to_assess +'PerDepth','depths':amount_layers, 'conditions': control_condition_list,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'lastLayer':last_layer,'firstLayer':first_layer,'threshold': threshold},overwrite=overwrite)

    
def stats_fits_between_layers(results_dir,repetitions,assessed_per="event",x0=None,y0=None,control_condition_list=[], threshold = 0.2, to_assess = "prop",model_list = ["mono","mixed","tuned"]):
    '''
    Stats and plots between different layers of a network
   
    Seems that kruskal wallis is fine with unequal sample sizes (more than ANOVA at least)
   

    Parameters
    ----------
   
    results_dir: directory in which the  runrepetitions of the models are stored
    repetitions: list of repetition information you want to assess. Different network depths are stacked on each other
        (e.g., [[50 reps 1 layer][50 reps 2 layers]])
    assessed_per: variable over which average summed activation is calculated (event, state_change, movie; string)
    x0: x variable in the timing space (duration, occupancy; string)
    y0: y variable in the timing space (period, isi; string)
    control_condition_list: list of control conditions, like no_recurrency, shuffled, 16nodes, init, None. Same length as the stacked repetitions
    threshold: threshold for which the nodes are considered tuned or monotonic
 

    Returns
    -------
    None.

    '''
    
    if x0 is None:
        x0 = ["duration", "ISI"]
    if y0 is None:
        y0 = ["period", "period"]
        
    stats_to_save = {}
    stats_to_save[to_assess]= {}
    df_plots = {}
    df_plots[to_assess]= {}
    all_layer_names = []
    stats_to_save['median_info'] = {}
    # proportion
    total_df = load_fit_data(results_dir, repetitions,assessed_per,x0,y0,control_condition_list, threshold, to_assess=to_assess)
    
    if to_assess != "prop":
        total_df = total_df.explode(to_assess)
    
    for network_id, network_depth in enumerate(repetitions):
        condition = control_condition_list[network_id]
        amount_layers = network_depth[0][0]["num_layers"]
        if condition == None:
            condition = 'regular'
            condition_key = amount_layers
        else:
            condition_key = condition
            
        
        all_layer_names.append(amount_layers)
        stats_to_save[to_assess][condition_key] = {}
        df_plots[to_assess][condition_key] = {}

        
        for model in model_list:
            stats = kruskal_and_posthocs(total_df,to_assess, 'layer',{'layer':[0,1,2,3,4],'condition':[condition],'model':[model],'depth':[amount_layers]})
            stats_to_save[to_assess][condition_key][model] = stats
            
            # create dataframe with same selection as stats for plotting
            list_tuples = []
            for tup_id, tup in enumerate(stats['group_names']):
                for datapoint in stats['data'][tup_id]:
                    list_tuples.append(tup + tuple([datapoint]))
                    
            # combine dataframes for monotonic and tuned models
            if model == model_list[0]:
                plot_df = pd.DataFrame.from_records(list_tuples,columns=[stats['input_args']['independent_var'], *list(stats['input_args']['selection'].keys()),stats['input_args']['dependent_var']])
                plot_df = plot_df.T.drop_duplicates().T   
                plot_df[to_assess] = plot_df[to_assess].astype(float)
            else:
                new_df = pd.DataFrame.from_records(list_tuples,columns=[stats['input_args']['independent_var'], *list(stats['input_args']['selection'].keys()),stats['input_args']['dependent_var']])
                new_df = new_df.T.drop_duplicates().T   
                new_df[to_assess] = new_df[to_assess].astype(float)
                plot_df = pd.concat([plot_df, new_df])
        
        overwrite = False
        if all_layer_names == [5] and (control_condition_list == ["init"] or control_condition_list == ["shuffled"]) and repetitions[0][0][0]['num_hidden'] == 16:
            accuracy_path, fig_file_name = save_stats([], {'results_section': to_assess + 'PerLayer','depths':amount_layers, 'numHidden':16,'conditions': condition,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'scale':'width','threshold':threshold},overwrite=overwrite)
        else:
            accuracy_path, fig_file_name = save_stats([], {'results_section': to_assess + 'PerLayer','depths':amount_layers, 'conditions': condition,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'scale':'width','threshold':threshold},overwrite=overwrite)

        stats_to_save['median_info'][network_id] = plot_violins(plot_df, stats,hue="model",hue_order = [0,1,2,3,4], y_x_ratio = 5/amount_layers,save_path=os.path.join(accuracy_path, fig_file_name))
        
        df_plots[to_assess][condition_key][model] = plot_df

    
    
      
    pickle_info = {'total_df': total_df, 'stats': stats_to_save, 'df_plots': df_plots}
  
    
    overwrite = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    if all_layer_names == [5] and (control_condition_list == ["init"] or control_condition_list == ["shuffled"]) and repetitions[0][0][0]['num_hidden'] == 16:
        save_stats(pickle_info, {'results_section':to_assess + 'PerLayer','depths':all_layer_names, 'numHidden':16,'conditions': control_condition_list,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'threshold':threshold},overwrite=overwrite)
    else:
        save_stats(pickle_info, {'results_section':to_assess + 'PerLayer','depths':all_layer_names, 'conditions': control_condition_list,'negative': c.NEGATIVE,'x0':x0,'y0':y0,'threshold':threshold},overwrite=overwrite)

    return pickle_info

def do_mono_tuned_fit_evaluations(results_dir,repetitions_all_depths,assessed_per="event",x0=["duration","ISI"],y0=["period","period"],control_condition_list=[], threshold = 0.2):
    
    # whether you need this depends on what you added in main
    normal_ids = np.where(np.array(control_condition_list) == None)[0]
    repetitions_compare_acc_model_depths = [repetitions_all_depths[i] for i in normal_ids]
    condition_list_compare_acc_model_depths = [control_condition_list[i] for i in normal_ids]
    
    pickle_info = stats_fits_between_depths(results_dir,repetitions_all_depths,assessed_per,x0,y0,control_condition_list, threshold,last_layer=True)

    stats_fits_between_layers(results_dir,repetitions_compare_acc_model_depths,assessed_per,x0,y0,condition_list_compare_acc_model_depths,threshold)
    
    return pickle_info

