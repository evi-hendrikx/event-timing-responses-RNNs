import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
from matplotlib.collections import PolyCollection
from scipy.stats import kruskal,mannwhitneyu,skew, kurtosis, levene
from scikit_posthocs import posthoc_dunn
import seaborn as sns
import statsmodels
import numpy as np

# Add missing np.NaN for compatibility with pycircstat
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pycircstat as circstat
import diptest

from model_fitting_stats import load_fit_data
from utility import get_timing_responses, save_stats
import config_file as c



def data_stats(data, first_vs_last = True,variable='distances',parameters_type=[],name_dict={}):
    """
    Do statistics on the provided data and plot in violin plot
    """
    
    stats = {}

    # between layers for each network
    stats['between_layers'] = {}
    plot_info = []
    for network_depth in data.keys():
        stats['between_layers'][network_depth] = {}            
    
        layer_order = [0,1,2,3,4,5]
        stats['between_layers'][network_depth]['anova'] = {}
        
        if variable == "fit_theta":
            eval_string = "circstat.tests.watson_williams("
        else:
            eval_string = "kruskal(" 
        for key in layer_order:
            if key in list(data[network_depth].keys()):
                eval_string = eval_string + "data[" + str(network_depth) + "][" + str(key) + "],"
                for node in data[network_depth][key]:
                    plot_info.append((node,key,network_depth))

        # networks with only one layer cannot be compared over layers
        if network_depth == 1:
            continue
        
                    
        eval_string = eval_string[:-1] + ")"
        # theta has different statistics, because it's a circular variable
        if variable == "fit_theta":
            [stats['between_layers'][network_depth]["anova"]["p"],stats['between_layers'][network_depth]["anova"]["all"]]= eval(eval_string)
            eval_string = eval_string.replace('watson_williams','cmtest') 
            stats['between_layers'][network_depth]['cmtest'] = eval(eval_string)
            print(stats['between_layers'][network_depth]['cmtest'])
        else:
            [stats['between_layers'][network_depth]["anova"]["F"],stats['between_layers'][network_depth]["anova"]["p"]]= eval(eval_string)
        
        
        # post hocs    
        # posthoc dunn test, with correction for multiple testing
        stats['between_layers'][network_depth]['post_hoc'] = {}
        p_vals = []
        p_vals_cm = []
        if variable != "fit_theta":
         
            eval_string = eval_string.replace("kruskal(","posthoc_dunn([").replace(")","],p_adjust='holm-sidak')")
            stats['between_layers'][network_depth]["post_hoc"]['vals'] = eval(eval_string)
    
            stats['between_layers'][network_depth]['variances'] = {}
            
            if variable != "fit_betaRatio":
                eval_string = eval_string.replace("posthoc_dunn([","levene(").replace("],p_adjust='holm-sidak')",", center='median')")
                stats['between_layers'][network_depth]["variances"]["main"] = eval(eval_string)
            
            stats['between_layers'][network_depth]["variances"]["posthoc"]= {}
            stats['between_layers'][network_depth]["variances"]["posthoc"]['stat']= {}
            stats['between_layers'][network_depth]["variances"]["posthoc"]['p']= {}
        else:
            stats['between_layers'][network_depth]['post_hoc']['print'] = {}
            stats['between_layers'][network_depth]["post_hoc_cm"] = {}
            stats['between_layers'][network_depth]["post_hoc_cm"]["print"]={}
            
            
        for key_id, key in enumerate(layer_order):
            
            for key2 in layer_order[key_id+1:]:
                if key in list(data[network_depth].keys()) and key2 in list(data[network_depth].keys()) :
                    
                    if variable != "fit_theta":
                        if not key in list(stats['between_layers'][network_depth]["variances"]["posthoc"]['stat'].keys()):
                            stats['between_layers'][network_depth]["variances"]["posthoc"]['stat'][key],stats['between_layers'][network_depth]["variances"]["posthoc"]['p'][key]={},{}
                        posthoc_name_field = "variances"
                        if variable != "fit_betaRatio":
                            eval_string = "levene(data[" + str(network_depth) + "][" + str(key) + "]," + "data[" + str(network_depth) + "][" + str(key2) + "],center = 'median')"
                            stats['between_layers'][network_depth]["variances"]["posthoc"]['stat'][key][key2],stats['between_layers'][network_depth]["variances"]["posthoc"]['p'][key][key2]=eval(eval_string)
                            p_vals.append(stats['between_layers'][network_depth]["variances"]["posthoc"]['p'][key][key2])
                            
                    else:
                        if not key in list(stats['between_layers'][network_depth]["post_hoc"]['print'].keys()):
                            stats['between_layers'][network_depth]["post_hoc"]['print'][key] ={}
                        eval_string = "circstat.tests.watson_williams(data[" + str(network_depth) + "][" + str(key) + "]," + "data[" + str(network_depth) + "][" + str(key2) + "])"
                        p,stats['between_layers'][network_depth]["post_hoc"]['print'][key][key2]=eval(eval_string)
                        p_vals.append(p)
                        posthoc_name_field = "post_hoc"
                        
                        
                        if not key in list(stats['between_layers'][network_depth]["post_hoc_cm"]['print'].keys()):
                            stats['between_layers'][network_depth]["post_hoc_cm"]['print'][key] ={}
                        eval_string = "circstat.tests.cmtest(data[" + str(network_depth) + "][" + str(key) + "]," + "data[" + str(network_depth) + "][" + str(key2) + "])"
                        p,stats['between_layers'][network_depth]["post_hoc_cm"]['print'][key][key2]=eval(eval_string)
                        p_vals_cm.append(p)
                        posthoc_name_field = "post_hoc"
                        
        if variable != "fit_betaRatio":
            _,stats['between_layers'][network_depth][posthoc_name_field]["posthoc_corrected"],_,_=statsmodels.stats.multitest.multipletests(p_vals, method='holm-sidak',alpha = 0.05)
        else:
            stats['between_layers'][network_depth]["variances"]["posthoc_corrected"] = {}
        if variable =="fit_theta":
            _,stats['between_layers'][network_depth]["post_hoc_cm"]["posthoc_corrected"],_,_=statsmodels.stats.multitest.multipletests(p_vals_cm, method='holm-sidak',alpha = 0.05)
            print(stats['between_layers'][network_depth]["post_hoc_cm"]["posthoc_corrected"])
                        
    
    if first_vs_last == True:
        # between first and last layer of the same network (cannot be paired, because different amount of nodes)
        all_first_layers = []
        all_last_layers = []
        stats['first_last_layer'] = {}
        for network_depth in data.keys():
            stats['first_last_layer'][network_depth] = {}
            first_layer = data[network_depth][0]
            all_first_layers.append(first_layer)
            last_layer = data[network_depth][network_depth-1]
            all_last_layers.append(last_layer)
            
            stats['first_last_layer'][network_depth]["MannWU"] = mannwhitneyu(first_layer,last_layer)
            print(network_depth)
            print(stats['first_last_layer'][network_depth]["MannWU"])
    
        
        # between first layers networks
        stats['first_layers'] = {}
        stats['first_layers']["anova"] = {}
        stats['first_layers']["post_hoc"] = {}
        stats['first_layers']["anova"]["F"],stats['first_layers']["anova"]["p"]= kruskal(*all_first_layers)
        stats['first_layers']["post_hoc"]['vals'],stats['first_layers']["post_hoc"]['print'] = posthoc_dunn(all_first_layers,p_adjust='holm-sidak')
        print(stats['first_layers']["post_hoc"]['print'])
        
        
        # between last layers networks
        stats['last_layers'] = {}
        stats['last_layers']["anova"] = {}
        stats['last_layers']["post_hoc"] = {}
        stats['last_layers']["anova"]["F"],stats['last_layers']["anova"]["p"]= kruskal(*all_last_layers)
        stats['last_layers']["post_hoc"]['vals'],stats['last_layers']["post_hoc"]['print'] = posthoc_dunn(all_last_layers,p_adjust='holm-sidak')
        print(stats['last_layers']["post_hoc"]['print'])


    # # plot including scatter points of data
    plot_df = pd.DataFrame(plot_info,columns=(variable,"layer", "depth"))
    print(variable)
    
    if variable == "fit_exp" or variable =="fit_expDur" or variable =="fit_expPer":
        stats['bimodality_coef'] = {}
        stats['diptest'] = {}
        stats['diptest']['ps']=[]
        for layer in range(max(plot_df['layer'])+1):
            stats['bimodality_coef'][layer] = bimodality_coefficient(plot_df[variable][plot_df['layer']==layer])
            stats['diptest'][layer] = diptest.diptest(plot_df[variable][plot_df['layer']==layer])
            stats['diptest']['ps'].append(stats['diptest'][layer][1])
        
        # fdr correction for exponents bimodality tests
        stats['diptest']['adj_ps']=statsmodels.stats.multitest.fdrcorrection(stats['diptest']['ps'])
        
    if variable == 'fit_betaRatio':
        inf_or_0_or_normal = plot_df.copy() 
        inf_or_0_or_normal[variable] = np.log10(inf_or_0_or_normal[variable])

        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == np.inf,'group'] = 'Inf'
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == -np.inf,'group'] = 'Zero'
        inf_or_0_or_normal.loc[np.logical_and(inf_or_0_or_normal[variable] != -np.inf,inf_or_0_or_normal[variable] != np.inf),'group'] = 'normal'
                       
       
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == np.inf, variable] = 20
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == -np.inf, variable] = -20
      
        for group in ["Inf","normal","Zero"]:

            for layer in range(max(list(inf_or_0_or_normal['layer']))+1):
                
       
                normal_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="normal",inf_or_0_or_normal["layer"]==layer))/ sum(inf_or_0_or_normal['layer'] == layer)
                inf_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="Inf",inf_or_0_or_normal["layer"]==layer))/ sum(inf_or_0_or_normal['layer'] == layer)
                zero_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="Zero",inf_or_0_or_normal["layer"]==layer))/ sum(inf_or_0_or_normal['layer'] == layer)                  
                if normal_ratio>=inf_ratio and normal_ratio>=zero_ratio:
                    max_ratio = normal_ratio
                elif inf_ratio>=normal_ratio and inf_ratio>=zero_ratio:
                    max_ratio = inf_ratio
                else:
                    max_ratio = zero_ratio
                if group == "Inf":
                    width_par = inf_ratio/max_ratio*0.8
                elif group == "Zero":
                    width_par = zero_ratio/max_ratio*0.8
                elif group == "normal":
                    width_par = normal_ratio/max_ratio *0.8 
                        
                if eval(group.lower() + "_ratio") != 0:
                    plot_layer_df = inf_or_0_or_normal[np.logical_and(inf_or_0_or_normal["group"]==group,inf_or_0_or_normal["layer"]==layer)]
                    
                    # Identify unique combinations of 'layer' and 'group'
                    unique_combinations = inf_or_0_or_normal[['layer']].drop_duplicates()
                    
                    # Create a DataFrame with these combinations and np.nan for the 'variable' and 'depth' columns
                    nan_rows = unique_combinations.copy()
                    nan_rows['variable'] = np.nan
                    nan_rows['depth'] = np.nan
                    
                    # Concatenate the original DataFrame with the new DataFrame containing NaN values
                    plot_layer_df = pd.concat([plot_layer_df, nan_rows], ignore_index=True)
                    
                    violin = sns.violinplot(data=plot_layer_df, x="layer", y=variable, color = [0,0,0],cut=0,inner = None,scale="width", width = width_par,dodge = False)
                    
        # currently only works per depth
        grouped_df = inf_or_0_or_normal.groupby(['depth', 'layer'],sort=False)
        median = grouped_df[variable].median().to_list()       
        Q25 = grouped_df[variable].quantile(.25).to_list()
        Q75 = grouped_df[variable].quantile(.75).to_list()
        stats['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]       
        stats['median_info_notLog'] = [str(round(10**median[idx],2)) + ' [' + str(round(10**Q25[idx],2)) + ', ' + str(round(10**Q75[idx],2)) +']' for idx in range(len(median))]       
        print([str(10**median[idx]) + ' [' + str(10**Q25[idx]) + ', ' + str(10**Q75[idx]) +']' for idx in range(len(median))])

        x_values_summary_stats = range(max(list(inf_or_0_or_normal['layer']))+1)
        plt.errorbar(x_values_summary_stats, Q25, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, median, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, Q75, xerr=0.4, elinewidth=0.5, linestyle='',color='white')

        
    else:
       
        
        grouped_df = plot_df.groupby(['depth', 'layer'],sort=False)
        
        if variable == "fit_theta":
            violin = sns.violinplot(data=plot_df, x="depth", y=variable, hue = "layer",hue_order = [0,1,2,3,4],color = [0,0,0],cut=0,inner=None,scale="width", width = 0.8)


            mean = []
            median = []
            lower_pct = []
            higher_pct = []
            grouped_df = plot_df.groupby(['depth', 'layer'],sort=False)
            for depth in range(min(plot_df['depth']),max(plot_df['depth'])+1):
                for layer in range(max(plot_df['layer'])+1):
                    if len(list(plot_df[variable][np.logical_and(plot_df['depth']==depth,plot_df['layer']==layer)]))>0:
                        mean.append(circstat.descriptive.mean(plot_df[variable][np.logical_and(plot_df['depth']==depth,plot_df['layer']==layer)]))
                        median_and_ci = circstat.descriptive.median(plot_df[variable][np.logical_and(plot_df['depth']==depth,plot_df['layer']==layer)],ci = 0.95)
                        
                        median.append(np.float64(median_and_ci[0]))
                        lower_pct.append(median_and_ci[1][0])
                        higher_pct.append(median_and_ci[1][1])

            stats['median_info']= [str(round(np.rad2deg(median[idx]/2),2)) + ' [' + str(round(np.rad2deg(lower_pct[idx]/2),2)) + ', ' + str(round(np.rad2deg(higher_pct[idx]/2),2)) +']' for idx in range(len(median))]
            
                  
            for i, v in enumerate(violin.findobj(PolyCollection)):
                v.set_facecolor('0')

        else:
            violin = sns.violinplot(data=plot_df, x="depth", y=variable, hue = "layer",hue_order = [0,1,2,3,4],color = [0,0,0],cut=0,inner='quartile',linecolor="white",scale="width", width = 0.8)

            for p in violin.lines[-3*15:]:
                p.set_linestyle('-')
                p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
                p.set_color('white')  # Sets the color of the quartile lines
                p.set_alpha(1)        
         
      
            for i, v in enumerate(violin.findobj(PolyCollection)):
                v.set_facecolor('0')
                
            mean = grouped_df[variable].mean().to_list()
            median = grouped_df[variable].median().to_list()
           
            Q25 = grouped_df[variable].quantile(.25).to_list()
            Q75 = grouped_df[variable].quantile(.75).to_list()
            stats['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]
       
        x_values_summary_stats = [np.where(np.asarray([1,2,3,4,5]) == plotted_dist)[0][0] for plotted_dist in grouped_df.mean().index.get_level_values(0)]
        x_values_violins = [-0.32, -0.16, 0, 0.16, 0.32]
    
        previous_x_value = x_values_summary_stats[0]
        layer_id = 0
        for x_id, x_value in enumerate(x_values_summary_stats):
            available_layers = np.unique(plot_df['layer'][plot_df['depth']==x_value + 1])
            if x_value != previous_x_value:
                layer_id = 0
            previous_x_value = x_value
            if len(list(stats['between_layers'].keys())) == 1:
                x_value = 0
            x_values_summary_stats[x_id] = x_value + x_values_violins[available_layers[layer_id]]
                
            layer_id += 1
            
        plt.errorbar(x_values_summary_stats,mean,fmt='.',color='red',capsize=0) 

    
    plt.errorbar(x_values_summary_stats,median,fmt='.',color='white',capsize=0) 
    if variable == "fit_theta":
        plt.errorbar(x_values_summary_stats, median, xerr=0.08, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, lower_pct, xerr=0.08, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, higher_pct, xerr=0.08, elinewidth=0.5, linestyle='',color='white')
    
    if name_dict['variable'] == 'distances':
        plt.plot([0,4],[0.2636,0.2636],'b--')
        plt.ylim(0,0.27)
    
    if name_dict['variable'] == 'peakY' or name_dict['variable'] == 'peakX' or name_dict['variable'] == 'fit_x0' or name_dict['variable'] == 'refit_x0' or name_dict['variable'] == 'fit_y0' or name_dict['variable'] == 'refit_y0':
        plt.ylim(-0.05,1.1)
    elif name_dict['variable'] == 'pca_ratios' or name_dict['variable'] == 'fit_ratio' or name_dict['variable'] == 'refit_ratio':
        plt.ylim(0,1)
    elif name_dict['variable'] == 'orientations' or name_dict['variable'] == 'fit_theta' or name_dict['variable'] == 'refit_theta':
        plt.ylim(0,2*np.pi)
        plt.yticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi],[0,45,90,135,180])
    elif name_dict['variable'] == 'pca1':
        plt.ylim(0,3)
    elif name_dict['variable'] == 'pca0':
        plt.ylim(0,5)
    elif name_dict['variable'] == 'fit_sigmaMajor' or name_dict['variable'] == 'refit_sigmaMajor':
        plt.ylim(0,10)
    elif name_dict['variable'] == 'fit_sigmaMinor' or name_dict['variable'] == 'refit_sigmaMinor':
        plt.ylim(0,7)
    elif name_dict['variable'] == 'fit_betaRatio':
        plt.ylim(-21, 21)
        plt.yticks([-20, -18, -12, -6, 0, 6, 12, 18, 20],['0','10^-18','10^-12','10^-6','1','10^6','10^12','10^18','inf'])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    matplotlib.rcParams['pdf.fonttype'] = 42
     
    overwrite = False
    
    accuracy_path, fig_file_name = save_stats(stats,name_dict,overwrite=overwrite)
    if os.path.exists(os.path.join(accuracy_path, fig_file_name)) and overwrite == False:
        print('file ', fig_file_name, ' already exists. Not overwriting')
    else:    
        plt.savefig(os.path.join(accuracy_path,fig_file_name))
    
    plt.show()



def compare_maps_tuned_predictions(results_dir,all_repetitions,assessed_per="event",x0=["duration","ISI"],y0=["period","period"],control_condition_list=[[None]], threshold = 0.2,parameters_type = ["tuned","mixed","mono"]):
    """
    Load in data about model parameters and run stats on them
    """

    overwrite = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    fit_x0,fit_y0,fit_sigmaMajor,fit_sigmaMinor,fit_theta,fit_ratio,fit_exp = {},{},{},{},{},{},{}
    fit_expDur, fit_expPer, fit_betaRatio = {},{},{}
    
    num_per_violin = {}

    mono_params = ['x0_exp','y0_exp','ratio_x0','ratio_y0','mono_slope','mono_intercept']
    tuned_params = ['pref_x0','pref_y0','sigma_x0','sigma_y0','theta','exponent','tuned_slope','tuned_intercept']
    params = [*mono_params, *tuned_params]
    total_df = load_fit_data(results_dir, all_repetitions,assessed_per,x0,y0,control_condition_list, threshold,to_assess = params)
    

    for response_type in ["tuned","mixed","mono"]:    

        for timing_type in ['_' + x0[timing_type_id] + '_'+ y0[timing_type_id] for timing_type_id, _ in enumerate(x0)]:
            
            for row_id, row in total_df.iterrows():
                if row['model'] != response_type or row['layer'] == "overall":
                    continue
 
                for node_id, _ in enumerate(row['timing_type']):
                    
                    if row['timing_type'][node_id] != timing_type:
                        continue                 

                    if response_type in parameters_type:
                        
                        if not row['depth'] in num_per_violin.keys():
                            num_per_violin[row['depth']] = {}
                            if "tuned" in parameters_type:
                                fit_sigmaMinor[row['depth']],fit_sigmaMajor[row['depth']] = {},{}
                                fit_ratio[row['depth']],fit_theta[row['depth']] = {},{}
                                fit_x0[row['depth']] = {}
                                fit_y0[row['depth']] = {}
                                fit_exp[row['depth']] = {}
                            if "mono" in parameters_type:
                                fit_expDur[row['depth']] = {}
                                fit_expPer[row['depth']] = {}
                                fit_betaRatio[row['depth']] = {}
                        if not row['layer'] in num_per_violin[row['depth']].keys():
                            num_per_violin[row['depth']][row['layer']] = 0
                            if "tuned" in parameters_type:

                                fit_sigmaMinor[row['depth']][row['layer']],fit_sigmaMajor[row['depth']][row['layer']] = [],[]
                                fit_ratio[row['depth']][row['layer']],fit_theta[row['depth']][row['layer']] = [],[]
                                fit_x0[row['depth']][row['layer']] = []
                                fit_y0[row['depth']][row['layer']] = []
                                fit_exp[row['depth']][row['layer']] = []
                            if "mono" in parameters_type:
                                fit_expDur[row['depth']][row['layer']] = []
                                fit_expPer[row['depth']][row['layer']] = []
                                fit_betaRatio[row['depth']][row['layer']] = []

                 
                        num_per_violin[row['depth']][row['layer']] += 1                        
                        
                        if response_type == "tuned":
                             # make sure the largest component is sigmaMajor and corresponding angulation is correct
                            if [float(row[param][node_id]) for param in tuned_params][3] > [float(row[param][node_id]) for param in tuned_params][2]:
                                old_major = [float(row[param][node_id]) for param in tuned_params][3]
                                old_minor = [float(row[param][node_id]) for param in tuned_params][2]
                                old_theta = 90 - [float(row[param][node_id]) for param in tuned_params][4]
                            else:
                                old_major = [float(row[param][node_id]) for param in tuned_params][2]
                                old_minor = [float(row[param][node_id]) for param in tuned_params][3]
                                old_theta = -[float(row[param][node_id]) for param in tuned_params][4]
                                
                            # angulation-wise: 10 and 190 degrees are the same orientations
                            if old_theta > 180:
                                old_theta = old_theta-180
                            elif old_theta < 0:
                                old_theta = old_theta + 180
                            
                            # Scale from 0 to 360 for circular statistics to get it
                            old_theta = old_theta * 2
                            old_theta = np.radians(old_theta)
                                
                            fit_x0[row['depth']][row['layer']].append([float(row[param][node_id]) for param in tuned_params][0])
                            fit_y0[row['depth']][row['layer']].append([float(row[param][node_id]) for param in tuned_params][1])
                            fit_theta[row['depth']][row['layer']].append(old_theta)
                            fit_sigmaMajor[row['depth']][row['layer']].append(old_major)
                            fit_sigmaMinor[row['depth']][row['layer']].append(old_minor)
                            fit_ratio[row['depth']][row['layer']].append(old_minor/old_major)
                            fit_exp[row['depth']][row['layer']].append([float(row[param][node_id]) for param in tuned_params][5])
                        elif response_type == "mono":
                            expDur = float(row[mono_params[0]][node_id])
                            expPer = float(row[mono_params[1]][node_id])
                           
                            
                            # ratio only makes sense when looking at two normalized components 
                            # 1 for duration and 1 for period in this case
                            # when fitting I did not normalize each component separately
                            # if we multiply the beta by the maximum of the prediction of the component
                            # it should be the same as when we would have fitted with normalized components
                            # normalized_a = (a - min(a))/max(a)
                            # response = (ax + by) * slope + intercept
                            # response = (normalized_a * (max(a) * x) + normalized_b * (max(b) * y) ) * slope + intercept + min(a) * x * slope + min(b) * y * slope
                            # new_x = max(a) * x and new_y = max(b) * y
                            # when fitting I would normalize the datapoints I have for the timings I have
                            # so also here I'm not going to interpolate the max, rather use the presented points
                            x0_type = timing_type.split('_')[1]
                            y0_type = timing_type.split('_')[2]
                            tmp, response_labels = get_timing_responses(results_dir, all_repetitions[0][0][0],assessed_per=assessed_per,gate_idx=0,x0=x0_type,y0=y0_type,control_condition=control_condition_list[0]) 
                            response_labels = response_labels/1000
                            response_labels = np.asarray(response_labels.T)
                            durations, periods = response_labels
                            max_x0 = max(durations**expDur)
                            min_x0 = min(durations**expDur)
                            max_y0 = max(periods/(periods**expPer))
                            min_y0 = min(periods/(periods**expPer))
                            unnormalized_beta_x0 = float(row[mono_params[2]][node_id])
                            unnormalized_beta_y0 = float(row[mono_params[3]][node_id])
                            beta_x0 = unnormalized_beta_x0 * (max_x0-min_x0)
                            beta_y0 = unnormalized_beta_y0 * (max_y0-min_y0)
                        
                            
                            if expDur != 0 and expPer != 1:
                                fit_betaRatio[row['depth']][row['layer']].append(beta_x0/beta_y0)
                            # period component contant --> betas can become anything. Ratio is only about duration
                            elif expPer == 1:
                                fit_betaRatio[row['depth']][row['layer']].append(np.inf)
                            # duration component contant --> betas can become anything. Ratio is only about period
                            elif expDur == 0:
                                fit_betaRatio[row['depth']][row['layer']].append(0)
                                
                            if beta_x0 == 0 and expDur != 0:
                                expDur = 0
                            elif beta_y0 == 0 and expPer != 1:
                                expPer = 1

                                
                            fit_expDur[row['depth']][row['layer']].append(expDur)
                            fit_expPer[row['depth']][row['layer']].append(expPer)
                                

    name_dict = {'results_section':'parmeterPlotsInfo_afterBerkeley','depths':total_df['depth'].unique(),'conditions': control_condition_list,'negative': c.NEGATIVE,'x0':x0,'y0':y0, 'responseFunctions':parameters_type,'numHidden':all_repetitions[0][0][0]['num_hidden']}
    pickle_info = {'total_df':total_df,'num_per_violin':num_per_violin}
    
    if "tuned" in parameters_type:
        for var in ['fit_x0', 'fit_y0','fit_sigmaMajor','fit_sigmaMinor','fit_theta','fit_ratio','fit_exp']:
            name_dict['variable'] = var
            print(var)
            pickle_info[var] = eval(var)
            if control_condition_list != ['no_recurrency']:
                data_stats(eval(var), first_vs_last = False,variable=var,parameters_type=parameters_type,name_dict=name_dict)
    if "mono" in parameters_type:
        for var in ['fit_expDur', 'fit_expPer','fit_betaRatio']:
            name_dict['variable'] = var
            print(var)
            pickle_info[var] = eval(var)
            data_stats(eval(var), first_vs_last = False,variable=var,parameters_type=parameters_type,name_dict=name_dict)

    name_dict['variable'] = 'all'
    
    
    save_stats(pickle_info, name_dict,overwrite=overwrite)

    return pickle_info


def bimodality_coefficient(data):
    """
    Run bimodality statistics (eventually we didn't use this: we did the diptest)
    """

    n = sum(np.isnan(data))
    skewness = skew(data)
    kurt = kurtosis(data,fisher = False)
    sample_size_adjustment = ((n-1)**2)/((n-2)*(n-3))
    bc = (skewness**2 + 1)/(kurt+3*sample_size_adjustment)
    return bc
