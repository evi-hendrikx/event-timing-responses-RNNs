
import scipy
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

from utility import save_stats

# datasets made in Matlab from previous studies
# provided in the github as well
dataNatComms =  scipy.io.loadmat('2022NatCommsResults/meanSub_new_exp.mat')

plot_info = {'subj':[k[0] for k in list(dataNatComms['meanSubSubj'].reshape(-1))],'hemi':[k[0] for k in list(dataNatComms['meanSubHemi'].reshape(-1))],'map':[k[0] for k in list(dataNatComms['meanSubRoi'].reshape(-1))],'xs':dataNatComms['meanSubXs'].reshape(-1),'ys':dataNatComms['meanSubYs'].reshape(-1),'ratio':dataNatComms['meanSubRatio'].reshape(-1)}
plot_df = pd.DataFrame(plot_info)
layer_order = ['V1','V2','V3','V4','LO1','LO2','TO1','TO2','V3AB','IPS0','IPS1','IPS2','IPS3','IPS4','IPS5','sPCS1','sPCS2','iPCS']
plot_df = plot_df[np.logical_and(plot_df["map"]!="VO1",plot_df["map"]!="VO2")]

plot_df.map = plot_df.map.astype("category")
plot_df.map = plot_df.map.cat.set_categories(layer_order)
plot_df = plot_df.sort_values(["map"])

stats = {}

for variable in ["xs","ys","ratio"]:
    print(variable)
    stats[variable]= {}

    stats[variable]['between_layers'] = {}            

    stats[variable]['between_layers']['anova'] = {}
   
    eval_string = "kruskal(" 
    for key in layer_order:
        if key in list(plot_df['map']):
            list_vals = list(plot_df[variable][plot_df['map']== key])
            exec(key + 'list = np.asarray(list_vals)[~np.isnan(list_vals)]')
            eval_string = eval_string + key + "list,"
   
    eval_string = eval_string[:-1] + ")"
    print(eval(eval_string))
    if variable == "theta":
        [stats[variable]['between_layers']["anova"]["p"],stats[variable]['between_layers']["anova"]["all"]]= eval(eval_string)
    else:
        [stats[variable]['between_layers']["anova"]["F"],stats[variable]['between_layers']["anova"]["p"]]= eval(eval_string)
    
    
    # post hocs    
    # posthoc dunn test, with correction for multiple testing
    stats[variable]['between_layers']['post_hoc'] = {}
    p_vals = []
 
    eval_string = eval_string.replace("kruskal(","posthoc_dunn([").replace(")","],p_adjust='holm-sidak')")
    stats[variable]['between_layers']["post_hoc"]['vals'],stats[variable]['between_layers']["post_hoc"]['print'] = eval(eval_string)
    print(stats[variable]['between_layers']["post_hoc"]['print'])
    
    if variable == "ratio":

        # this is for plotting 
        # this part of the script makes sure that the violin is about the values that are not 0 or infinite
        # but that the widths of all the parts are still scaled proportionally
        inf_or_0_or_normal = plot_df.copy() 
        inf_or_0_or_normal[variable] = np.log10(inf_or_0_or_normal[variable])
    
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == np.inf,'group'] = 'Inf'
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == -np.inf,'group'] = 'Zero'
        inf_or_0_or_normal.loc[np.logical_and(inf_or_0_or_normal[variable] != -np.inf,inf_or_0_or_normal[variable] != np.inf),'group'] = 'normal'

        max_value = inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] != np.inf, variable].max()
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == np.inf, variable] = 3
        min_value = inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] != -np.inf, variable].min()
        inf_or_0_or_normal.loc[inf_or_0_or_normal[variable] == -np.inf, variable] = -3
      
        for group in ["Inf","normal","Zero"]:
            for layer in layer_order:
       
                normal_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="normal",inf_or_0_or_normal["map"]==layer))/ sum(inf_or_0_or_normal['map'] == layer)
                inf_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="Inf",inf_or_0_or_normal["map"]==layer))/ sum(inf_or_0_or_normal['map'] == layer)
                zero_ratio = sum(np.logical_and(inf_or_0_or_normal["group"]=="Zero",inf_or_0_or_normal["map"]==layer))/ sum(inf_or_0_or_normal['map'] == layer)                  
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
                print(width_par)
                        
                if eval(group.lower() + "_ratio") != 0:
                    plot_layer_df = inf_or_0_or_normal[np.logical_and(inf_or_0_or_normal["group"]==group,inf_or_0_or_normal["map"]==layer)]
                    
                    # Identify unique combinations of 'layer' and 'group'
                    unique_combinations = inf_or_0_or_normal[['map']].drop_duplicates()
                    
                    # Create a DataFrame with these combinations and np.nan for the 'variable' and 'depth' columns
                    nan_rows = unique_combinations.copy()
                    nan_rows['variable'] = np.nan
                    nan_rows['depth'] = np.nan
                    
                    # Concatenate the original DataFrame with the new DataFrame containing NaN values
                    plot_layer_df = pd.concat([plot_layer_df, nan_rows], ignore_index=True)
                    
                    violin = sns.violinplot(data=plot_layer_df, x="map", y=variable, color = [0,0,0],cut=0,inner='quartile',linecolor="white",scale="width", width = width_par,dodge = False)
                    
        # the medians and quartiles are still calculated over the whole data
        grouped_df = inf_or_0_or_normal.groupby(['map'],sort=False)
        median = grouped_df[variable].median().to_list()       
        Q25 = grouped_df[variable].quantile(.25).to_list()
        Q75 = grouped_df[variable].quantile(.75).to_list()
        layer_id = 0
        x_values_summary_stats = range(len(layer_order))
        plt.errorbar(x_values_summary_stats, Q25, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, median, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(x_values_summary_stats, Q75, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(range(len(np.unique(plot_df['map']))),median,fmt='.',color='white',capsize=0) 

        stats[variable]['median_info'] = [str(round(10**median[idx],2)) + ' [' + str(round(10**Q25[idx],2)) + ', ' + str(round(10**Q75[idx],2)) +']' for idx in range(len(median))]       


    else:
        violin = sns.violinplot(data=plot_df, x="map", y=variable, color = [0,0,0],cut=0,inner='quartile',linecolor="white",scale="width", width = 0.8,order=layer_order)
    

        for p in violin.lines[-3*len(np.unique(plot_df['map'])):]:
            p.set_linestyle('-')
            p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
            p.set_color('white')  # Sets the color of the quartile lines
            p.set_alpha(1)            
    
           
        grouped_df = plot_df.groupby(['map'],sort=False)
        mean = grouped_df[variable].mean().to_list()
        Q25 = grouped_df[variable].quantile(.25).to_list()
        Q75 = grouped_df[variable].quantile(.75).to_list()
        median = grouped_df[variable].median().to_list()
        
        plt.errorbar(range(len(np.unique(plot_df['map']))),median,fmt='.',color='white',capsize=0) 
        plt.errorbar(range(len(np.unique(plot_df['map']))),mean,fmt='.',color='red',capsize=0)

        stats[variable]['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]       

    if variable == 'xs' or variable == 'ys':
        plt.ylim(0,1)
 
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    name_dict = {'results_section':'parmeterPlotsInfo','plots':'Hendrikx2022','variable': variable}

    overwrite = False
    accuracy_path, fig_file_name = save_stats(stats,name_dict,overwrite=overwrite)
    if os.path.exists(os.path.join(accuracy_path, fig_file_name)) and overwrite == False:
        print('file ', fig_file_name, ' already exists. Not overwriting')
    else:    
        plt.savefig(os.path.join(accuracy_path,fig_file_name))
    
    
    plt.show()
