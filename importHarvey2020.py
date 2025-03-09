import scipy
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import os
import statsmodels
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import pycircstat as circstat

from utility import save_stats

# datasets made in Matlab from previous studies
# provided in the github as well
dataCurBio =  scipy.io.loadmat('2020CurBioResults/dataPointsCurBio.mat')

plot_info = []
for datapoint_id in range(len(dataCurBio['SigmaMajor'])):
   
    plot_info.append((dataCurBio['subjectLabels'][0][datapoint_id][0],dataCurBio['hemisphereLabels'][0][datapoint_id][0],dataCurBio['mapLabels'][0][datapoint_id][0], dataCurBio['Q2D'][datapoint_id][0],dataCurBio['Q2P'][datapoint_id][0],dataCurBio['SigmaMajor'][datapoint_id][0],dataCurBio['SigmaMinor'][datapoint_id][0],dataCurBio['SigmaRatio'][datapoint_id][0],90-np.degrees(dataCurBio['SigmaTheta'][datapoint_id][0]), dataCurBio['Exps'][datapoint_id][0]))
    
plot_df = pd.DataFrame(plot_info,columns=("subj","hemi", "map","mean_X","mean_Y","maj","min","ratio","theta","exp"))
stats = {}

for variable in ["mean_X","mean_Y","maj","min","ratio","theta","exp"]:
    print(variable)
    stats[variable]= {}

    stats[variable]['between_layers'] = {}            

    layer_order = ['TLO','TTOP','TTOA','TPO','TLS','TPCI','TPCM','TPCS','TFI','TFS']
    stats[variable]['between_layers']['anova'] = {}
    
    if variable == "theta":
        eval_string = "circstat.tests.watson_williams("
    else:
        eval_string = "kruskal(" 
    for key in layer_order:
        if key in list(plot_df['map']):
            list_vals = list(plot_df[variable][plot_df['map']== key])
            exec(key + 'list = np.asarray(list_vals)[~np.isnan(list_vals)]')
            if variable == "theta":
                exec(key + "list = np.radians(" + key + "list) * 2")
            eval_string = eval_string + key + "list,"
   
    eval_string = eval_string[:-1] + ")"
    if variable == "theta":
        [stats[variable]['between_layers']["anova"]["p"],stats[variable]['between_layers']["anova"]["all"]]= eval(eval_string)
        eval_string = eval_string.replace("watson_williams","cmtest")
        stats[variable]['between_layers']["cmtest"] = eval(eval_string)
        print(stats[variable]['between_layers']["cmtest"])
    else:
        [stats[variable]['between_layers']["anova"]["F"],stats[variable]['between_layers']["anova"]["p"]]= eval(eval_string)
        print(eval(eval_string))
    
    
    # post hocs    
    # posthoc dunn test, with correction for multiple testing
    stats[variable]['between_layers']['post_hoc'] = {}
    p_vals = []
    p_vals_cm = []
    if variable != "theta":
     
        eval_string = eval_string.replace("kruskal(","posthoc_dunn([").replace(")","],p_adjust='holm-sidak')")
        stats[variable]['between_layers']["post_hoc"]['vals'],stats[variable]['between_layers']["post_hoc"]['print'] = eval(eval_string)
        
    else:
        stats[variable]['between_layers']['post_hoc']['print'] = {}
        stats[variable]['between_layers']["post_hoc_cm"] = {}
        
    # theta should be done with circular statistics
    for key_id, key in enumerate(layer_order):
        
        for key2 in layer_order[key_id+1:]:
                
            if variable == "theta":
                if not key in list(stats[variable]['between_layers']["post_hoc"]['print'].keys()):
                    stats[variable]['between_layers']["post_hoc"]['print'][key] ={}
                    stats[variable]['between_layers']["post_hoc_cm"][key] = {}
                p,stats[variable]['between_layers']["post_hoc"]['print'][key][key2]=eval("circstat.tests.watson_williams("+key + "list," + key2 + "list)")
                p_vals.append(p)
                posthoc_name_field = "post_hoc"
                
                p_cm,stats[variable]['between_layers']["post_hoc_cm"][key][key2]=eval("circstat.tests.cmtest("+key + "list," + key2 + "list)")
                p_vals_cm.append(p_cm)

    if variable == "theta":
        _,stats[variable]['between_layers'][posthoc_name_field]["posthoc_corrected"],_,_=statsmodels.stats.multitest.multipletests(p_vals, method='holm-sidak',alpha = 0.05)
        _,stats[variable]['between_layers']["post_hoc_cm"]["posthoc_corrected"],_,_=statsmodels.stats.multitest.multipletests(p_vals_cm, method='holm-sidak',alpha = 0.05)
        print(stats[variable]['between_layers']["post_hoc_cm"]["posthoc_corrected"])

    else:
        print(stats[variable]['between_layers']["post_hoc"]['print'])

    if variable != "theta":
        violin = sns.violinplot(data=plot_df, x="map", y=variable, color = [0,0,0],cut=0,inner='quartile',linecolor="white",scale="width", width = 0.8)
    
        for p in violin.lines[-3*len(np.unique(plot_df['map'])):]:
            p.set_linestyle('-')
            p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
            p.set_color('white')  # Sets the color of the quartile lines
            p.set_alpha(1)            
           
        grouped_df = plot_df.groupby(['map'],sort=False)
        mean = grouped_df[variable].mean().to_list()
        median = grouped_df[variable].median().to_list()
        
        Q25 = grouped_df[variable].quantile(.25).to_list()
        Q75 = grouped_df[variable].quantile(.75).to_list()
        
        stats[variable]['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(Q25[idx],2)) + ', ' + str(round(Q75[idx],2)) +']' for idx in range(len(median))]

    # theta is circular degrees, so gets a slightly different approach
    else:
        violin = sns.violinplot(data=plot_df, x="map", y=variable, color = [0,0,0],cut=0,inner=None,scale="width", width = 0.8)

        mean = []
        median = []
        lower_pct = []
        higher_pct = []
        for key in layer_order:
            mean.append(np.rad2deg(circstat.descriptive.mean(eval(key+"list")))/2)
            median_and_ci = circstat.descriptive.median(eval(key+"list"),ci = 0.95)
            median.append(np.rad2deg(np.float64(median_and_ci[0]))/2)
            lower_pct.append(np.rad2deg(median_and_ci[1][0])/2)
            higher_pct.append(np.rad2deg(median_and_ci[1][1])/2)
            
        stats['median_info'] = [str(round(median[idx],2)) + ' [' + str(round(lower_pct[idx],2)) + ', ' + str(round(higher_pct[idx],2)) +']' for idx in range(len(median))]
        plt.errorbar(range(len(np.unique(plot_df['map']))), median, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(range(len(np.unique(plot_df['map']))), lower_pct, xerr=0.4, elinewidth=0.5, linestyle='',color='white')
        plt.errorbar(range(len(np.unique(plot_df['map']))), higher_pct, xerr=0.4, elinewidth=0.5, linestyle='',color='white')

        
    plt.errorbar(range(len(np.unique(plot_df['map']))),median,fmt='.',color='white',capsize=0) 
    plt.errorbar(range(len(np.unique(plot_df['map']))),mean,fmt='.',color='red',capsize=0)
    
    if variable == 'mean_Y' or variable == 'mean_X' or variable == 'ratio':
        plt.ylim(0,1)
    elif variable == 'theta':
        plt.ylim(0,180)
    elif variable == 'min':
        plt.ylim(0,8)
    elif variable == 'maj':
        plt.ylim(0,10)
    elif variable == 'exp':
        plt.ylim(0,1)    
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    name_dict = {'results_section':'parmeterPlotsInfo','plots':'Harvey2020','variable': variable}

    overwrite = False
    accuracy_path, fig_file_name = save_stats(stats,name_dict,overwrite=overwrite)
    if os.path.exists(os.path.join(accuracy_path, fig_file_name)) and overwrite == False:
        print('file ', fig_file_name, ' already exists. Not overwriting')
    else:    
        plt.savefig(os.path.join(accuracy_path,fig_file_name))
    
    
    plt.show()
