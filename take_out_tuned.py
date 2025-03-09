import numpy as np
import random
from scipy.stats import wilcoxon
import torch
import itertools
import copy
import matplotlib.pyplot as plt

from generate_dataset import EventTimingDataset
from torch.utils.data import DataLoader
from utility import save_stats,get_result_paths
from pipeline import predict_dataset
from model_fitting_stats import load_fit_data
import config_file as c  

    
def switch_off_nodes_last_layer(results_dir,repetitions,switch_off="tuned",num_layers=5,num_hidden=16,assessed_per="event",x0=None,y0=None,control_condition_list=None, threshold = 0.2):
    
    '''
    switch_off_nodes_last_layer assesses performance of the network with target group weights set to 0
    
    results_dir: directory in which the repetitions of the models are stored
    repetitions: list of repetition information you want to assess. Different network depths are stacked on each other
    assessed_per: variable over which average summed activation is calculated (event, state_change, movie; string)
    x0: x variable in the timing space (duration, occupancy; string)
    y0: y variable in the timing space (period, isi; string)
    control_condition_list: list of control conditions, like no_recurrency, shuffled, 16nodes, init, None. Same length as the stacked repetitions
    threshold: threshold for which the nodes are considered tuned or monotonic
    
    returns percentiles for the state change accuracy
    '''
    if x0 is None:
        x0 = ["duration","ISI"]
    if y0 is None:
        y0 = ["period","period"]
    if control_condition_list is None:
        control_condition_list = ['16nodes']
    
    stats = {}
    stats['repetitions'] = repetitions
    stats['target']=switch_off
    stats['target_off'] = {}
    stats['rest_off'] = {}
    stats['target_off']['nodes'] = []
    stats['target_off']['per_event'] = []
    stats['target_off']['per_state_change'] = []
    stats['rest_off']['per_event'] = []
    stats['rest_off']['per_state_change'] = []
    stats['rest_off']['combis'] = []
    stats['percentiles'] = {}
    stats['percentiles']['per_event'] = []
    stats['percentiles']['per_state_change'] = []
    stats['distribution_percentiles'] = {}
    stats['distribution_percentiles']['per_event'] = []
    stats['distribution_percentiles']['per_state_change'] = []

    store_percentiles = []
    store_percentiles_event = []
    
    # Assess accuracy of all timings
    dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                 events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                 cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control=False)
   
    # ASSUMES ALL REPETITIONS IN ALL NETWORKS HAVE THE SAME BATCH SIZE (which they do in my case)
    eval_dataloader = DataLoader(dataset, batch_size=repetitions[0][0][0]["batch_size"], shuffle=False, num_workers=0)
    
    # load the types of nodes for each network repetition
    total_df = load_fit_data(results_dir, repetitions,assessed_per,x0,y0,control_condition_list, threshold,to_assess="classifications")

    # Just the last layer, because otherwise last tuned also might not get input? and that might cause issues
        # ?? However if only kicking out the last layer it might be that tuned from before-last layer still gets information through by monotonic node??
        # eventually did go with only last layer
    nodes_to_turn_off = {}
    for row in total_df.index:
        rep = total_df.iloc[row]['rep']
        if rep in nodes_to_turn_off.keys():    
            if total_df.iloc[row]['model']==switch_off and not hasattr(total_df.iloc[row]['layer'],"__len__") and total_df.iloc[row]['layer']==num_layers-1 and total_df.iloc[row]['depth']==num_layers and len(total_df.iloc[row]['node'])>0:
                nodes_to_turn_off[rep].append(total_df.iloc[row]['node'])
        else:
            if total_df.iloc[row]['model']==switch_off and not hasattr(total_df.iloc[row]['layer'],"__len__") and total_df.iloc[row]['layer']==num_layers-1 and total_df.iloc[row]['depth']==num_layers and len(total_df.iloc[row]['node'])>0:
                nodes_to_turn_off[rep] = total_df.iloc[row]['node']
                
    # Most biased to NOT finding an effect that we DO want to find (so probably most fair):
        # We assessed the category of the nodes in odd and even data splits
        # Here, we only count as target if considered target in both data splits
            # if it's actually target then there is a target node still in the network 
            # might increase the accuracy that we compare it against                
        # nodes will be in the list twice if they are classified as target in both cases
    for rep in nodes_to_turn_off:
        all_nodes = nodes_to_turn_off[rep]
        limited_nodes = list(set([node for node in all_nodes if list(all_nodes).count(node)>1]))
        nodes_to_turn_off[rep]=limited_nodes
   
    for rep in nodes_to_turn_off:    
        print(f"repetition: {rep}")
        repetition = {"num_layers": num_layers, "num_hidden": num_hidden, "norm": "layer_norm", "batch_size": 50, "ind_rnn": True, "weight_constraint": False,'LR':2e-3,'dropout_prob': c.DROPOUT_PROB,'assessed_per':'event','cross_v': True,'weight_reg':1e-08, 'scrambled_control': False, 'counter': rep}   
        
        net_path, _, _, _, _, _, _, _ = get_result_paths(results_dir, repetition)            
        checkpoint = torch.load(net_path, map_location=c.DEVICE)
        network_weights = checkpoint["model_state_dict"]

        # shapes out weights (transposed compared to shape in between nodes!)
        string_get_weights = 'out.0.weight'
        
        # set target weights to 0
        copy_weight_info = copy.deepcopy(network_weights)
        for node in nodes_to_turn_off[rep]:
            copy_weight_info[string_get_weights][0][int(node)] = 0
        
        # evaluate accuracy of network with weights set to 0
        print("target off")
        c.NET_TYPE = "RNN"
        net = c.RNNetwork(c.INPUT_SIZE, repetition["num_hidden"], repetition["num_layers"], c.NONLINEARITY, c.BIAS, c.BATCH_FIRST, repetition["dropout_prob"], repetition["norm"], repetition["ind_rnn"], repetition['weight_constraint'])
        net.load_state_dict(copy_weight_info)
        net.eval()
        net.to(c.DEVICE)
        _, _, _, _, _, _, _, _, _, _, _, _, no_target_acc_proportions_event, no_target_acc_proportions_state_change = predict_dataset(net, eval_dataloader, c.LOSS_FN, return_movies=True)
        
        stats['target_off']['per_event'].append(no_target_acc_proportions_event)
        stats['target_off']['per_state_change'].append(no_target_acc_proportions_state_change)
        stats['target_off']['nodes'].append(nodes_to_turn_off[rep])

        # other combinations of nodes to turn the weights off
        remaining_nodes = [i for i in np.arange(num_hidden) if i not in nodes_to_turn_off[rep]]
        get_possible_combinations = list(itertools.combinations(remaining_nodes,len(nodes_to_turn_off[rep])))
        not_target_off_distribution = []
        not_target_off_distribution_event = []
        
        # shuffle all combis in case you will only try 500 combinations
        random_indices = random.sample(range(len(get_possible_combinations)),len(get_possible_combinations))
        get_possible_combinations = [get_possible_combinations[r_i] for r_i in random_indices]
        
        combi_counter = 0
        for combi in get_possible_combinations:
            print(f"combi off: {combi_counter}/{len(get_possible_combinations)}")
            if combi_counter < 500:

                copy_weight_info = copy.deepcopy(network_weights)
            
                for node_id in combi:
                    copy_weight_info[string_get_weights][0][int(node_id)] = 0
                
                c.NET_TYPE = "RNN"
                net = c.RNNetwork(c.INPUT_SIZE, repetition["num_hidden"], repetition["num_layers"], c.NONLINEARITY, c.BIAS, c.BATCH_FIRST, repetition["dropout_prob"], repetition["norm"], repetition["ind_rnn"], repetition['weight_constraint'])
                net.load_state_dict(copy_weight_info)
                net.eval()
                net.to(c.DEVICE)
                _, _, _, _, _, _, _, _, _, _, _, _, acc_proportions_event, acc_proportions_state_change = predict_dataset(net, eval_dataloader, c.LOSS_FN, return_movies=True)
                stats['rest_off']['per_event'].append(acc_proportions_event)
                stats['rest_off']['per_state_change'].append(acc_proportions_state_change)
                stats['rest_off']['combis'].append(combi)

                
                not_target_off_distribution.append(np.mean(acc_proportions_state_change))
                not_target_off_distribution_event.append(np.mean(acc_proportions_event))
                
                combi_counter += 1
            else:
                break

           
        n = sum(not_target_off_distribution<=np.mean(no_target_acc_proportions_state_change))
        percentile_score_target = n/len(not_target_off_distribution)
        stats['percentiles']['per_state_change'].append(percentile_score_target)
        
        n = sum(not_target_off_distribution_event<=np.mean(no_target_acc_proportions_event))
        percentile_score_target_event = n/len(not_target_off_distribution_event)
        stats['percentiles']['per_event'].append(percentile_score_target_event)
        
        if rep < 5:
            plt.hist(not_target_off_distribution,50)
            plt.axvline(np.mean(no_target_acc_proportions_state_change))
            plt.xlim((0,1))
            plt.show()
            
        store_percentiles.append(percentile_score_target)
        store_percentiles_event.append(percentile_score_target_event)


    plt.hist(store_percentiles)
    plt.show()
    
    plt.hist(store_percentiles_event)
    plt.show()
    
    stats["distribution_percentiles"]['per_state_change'] = wilcoxon(np.asarray(store_percentiles)-0.5)
    stats["distribution_percentiles"]['per_event'] = wilcoxon(np.asarray(store_percentiles_event)-0.5)

    
    save_stats(stats,{'results_section':'turnOff','target':switch_off,'layers':num_layers,'hidden':num_hidden,'lastLayer':True},overwrite = False)

    return store_percentiles