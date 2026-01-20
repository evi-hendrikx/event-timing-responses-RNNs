import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from generate_dataset import EventTimingDataset
from pipeline import train_net
from utility import to_numpy, get_result_dirs, get_result_paths
import config_file as c
from model_fitting import run_fit
from create_special_conditions import evaluate_special_conditions
from accuracy_stats import do_accuracy_evaluations
from model_fitting_stats import do_mono_tuned_fit_evaluations
from parameter_stats import compare_maps_tuned_predictions
from take_out_tuned import switch_off_nodes_last_layer


# create data and result directories if necessary
dirs = [c.DATASET_DIR, c.RESULTS_DIR, c.STATS_DIR]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)


def run_model(args):
    '''
    Initiates making dataset and training networks to predict next frame. Saves state of the network    
    Also continues running a network that stopped training early (if c.USED_CACHED_NETWORKS == True)
  
    args: specifications of the properties of this repetition, dict with:
    "num_layers", "num_hidden", "norm" (type normalization), "batch_size", 
        "ind_rnn" (bool about type of RNN, we stuck with indRNN (Li et al., 2018), so this is True), 
        "weight_constraint" (bool),'LR' (learning rate),'dropout_prob': c.DROPOUT_PROB,
        'assessed_per' (over which range you determine the accuracy, string: "event", "state change" or "movie"),
        'cross_v' (bool),'weight_reg','scrambled_control' (bool),'counter' (number of the repetition)   
    '''
  
        
    # setup directory for this experimental configuration
    exp_dir, sub_results_dir = get_result_dirs(c.RESULTS_DIR, args)
    net_path, _, _, _, _, _, _, _ = get_result_paths(c.RESULTS_DIR, args)
    print(net_path)
    dir_list = [exp_dir, sub_results_dir]
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

    # get dataset
    # use seed: every repetition gets same test-train data ==> want to look at whether structure of the network matters, not necessarily whether it becomes really good at generalizing the task
    generator = torch.Generator().manual_seed(c.SEED)
    dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                 events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                 cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control = c.CONTROL_MOVIES)
    
    trainset_size = int(len(dataset) * c.TRAIN_SPLIT_RATIO)
    train_set, test_set = torch.utils.data.random_split(dataset, [trainset_size, len(dataset) - trainset_size],
                                                        generator)
    
    train_dataloader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=c.SHUFFLE, pin_memory=False,
                                  num_workers=c.NUM_WORKERS, persistent_workers=True if c.NUM_WORKERS > 0 else False)


    if 'weight_constraint' not in args:
        w_c = None
    else:
        w_c = args['weight_constraint']
        
    
    if c.USE_CACHED_NETWORKS and os.path.exists(net_path):    
        net = c.NET_CLASS(input_size=c.INPUT_SIZE, hidden_size=args["num_hidden"], num_layers=args["num_layers"],
                          nonlinearity=c.NONLINEARITY, bias=c.BIAS, batch_first=c.BATCH_FIRST, dropout=args["dropout_prob"],
                          norm=args["norm"], ind_rnn=args["ind_rnn"], weight_constraint=w_c)
        net.to(c.DEVICE)
        if 'weight_reg' in args:
            wd = args["weight_reg"]
        else:
            wd = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=args["LR"], weight_decay=wd)
        
        # eventually decided against exponential learning rate --> set it to 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)
        checkpoint = torch.load(net_path, map_location=c.DEVICE, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        train_losses = checkpoint['loss']        
        if isinstance(train_losses, torch.Tensor):
            train_losses = list(to_numpy(train_losses))
        
        trained_epochs = len(train_losses)
        remaining_epochs = c.NUM_EPOCHS - trained_epochs

        if remaining_epochs > 0:        
            # Network in training mode (enable stochastic num_layers, e.g. dropout)
            net.train()
            train_losses = train_net(net, train_dataloader, optimizer, scheduler, c.LOSS_FN, remaining_epochs, net_path,
                                     c.EARLY_STOPPING, train_losses, w_c)

            # save network (overwrite existing cache)
            state = {
                'model_state_dict': net.state_dict(),
                'loss': train_losses,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(state, net_path)

    else:        
        # define network and optimizer
        net = c.NET_CLASS(input_size=c.INPUT_SIZE, hidden_size=args["num_hidden"], num_layers=args["num_layers"],
                      nonlinearity=c.NONLINEARITY, bias=c.BIAS, batch_first=c.BATCH_FIRST, dropout=args["dropout_prob"],
                      norm=args["norm"], ind_rnn=args["ind_rnn"], weight_constraint=w_c)
        
        net.to(c.DEVICE)
        if 'weight_reg' in args:
            wd = args["weight_reg"]
        else:
            wd = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=args["LR"], weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,1)
        
        # use same initializations as network trained on normal stimuli for control stimuli
        if c.CONTROL_MOVIES == True:
            net_path_unscrambled_init = net_path.replace('_/net','_init/net').replace('scrambled_control_True','scrambled_control_False').replace('scrambled_control_AFTERSHIFT','scrambled_control_False')
            if c.USE_CACHED_NETWORKS and os.path.exists(net_path_unscrambled_init):
                print('loading unscrambled initialization')
                checkpoint = torch.load(net_path_unscrambled_init, map_location=c.DEVICE)
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
        # save initialization of the network
        state = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }
        save_initial_state_path = net_path.replace('_/net','_init/net')
        os.makedirs(save_initial_state_path.replace('_init/net','_init'))
        torch.save(state, save_initial_state_path)
        
        net.train()
        train_losses = train_net(net, train_dataloader, optimizer, scheduler, c.LOSS_FN, c.NUM_EPOCHS, net_path, c.EARLY_STOPPING, None, w_c)

        # end of training: save network (overwrite existing cache)
        state = {
            'model_state_dict': net.state_dict(),
            'loss': train_losses,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(state, net_path)
    



if __name__ == "__main__":
    
    repetitions_all_depths = []
    condition_list = []
    special_conditions_list = []
    eval_controls = False
    special_conditions = False
    results_dir = c.RESULTS_DIR
      
    
    # train networks with different depths with similar amount of parameters 
    # (so more layers becomes less nodes per layer)
    # (eventually these were also used for the layer-size-matched by simply
    # setting the num_hidden to 16)
    for layers in [1,2,3,4,5]:
        ind_rnn_bool = True
        if layers == 1:
            num_hidden = 77 #16
        elif layers == 2:
            num_hidden = 16 #16 #8 #64
        elif layers == 3:
            num_hidden = 12 #16
        elif layers ==4:
            num_hidden = 9 #16
        elif layers == 5:
            num_hidden = 64 #16 #8 #64
        
        repetitions = []
        for ii in range(50):
            # information repetition
            repetition = {"num_layers": layers, "num_hidden": num_hidden, "norm": "layer_norm", "batch_size": 50, "ind_rnn": ind_rnn_bool, "weight_constraint": False,'LR':2e-3,'dropout_prob': c.DROPOUT_PROB,'assessed_per':'event','cross_v': True,'weight_reg':1e-08, 'scrambled_control': False, 'counter': ii}   
            
            # train network on the task
            run_model(repetition)
            
            # determine monotonic and tuned fits to the activation of the nodes of the network
            run_fit(repetition,control_condition=None)

            if len(repetitions) == 0:
                repetitions = repetition
            else:
                repetitions = np.vstack((repetitions, repetition))

        repetitions_all_depths.append(repetitions)
        condition_list.append(None)
    
    # special control cases
    # NOTE WHEN RUNNING THIS LOOP ON SHUFFLED WHEN TRAINING MODELS
    # YOU WILL ALSO WANT TO CHECK/ CHANGE SETTINGS IN CONFIG_FILE
    if eval_controls == True:
        for condition in ["init","shuffled","no_recurrency","16nodes"]:
            ind_rnn_bool = True
            shuffled_bool = False
            layers = 5
            num_hidden = 8
            
            if condition == "shuffled":
                shuffled_bool = True
            elif condition == "no_recurrency":
                ind_rnn_bool = False
                num_hidden = 9
            elif condition == "16nodes":
                num_hidden = 16
                
            if not isinstance(layers,list):
                layers = [layers]
                
            for layer in layers:
                repetitions = []
                for ii in range(50):
                    repetition = {"num_layers": layer, "num_hidden": num_hidden, "norm": "layer_norm", "batch_size": 50, "ind_rnn": ind_rnn_bool, "weight_constraint": False,'LR':2e-3,'dropout_prob': c.DROPOUT_PROB,'assessed_per':'event','cross_v': True,'weight_reg':1e-08, 'scrambled_control': shuffled_bool, 'counter': ii}   

                    # train the network
                    run_model(repetition)

                    # determine monotonic and tuned fits to the activation of the nodes of the network
                    run_fit(repetition,control_condition=condition)
        
                    if len(repetitions) == 0:
                        repetitions = repetition
                    else:
                        repetitions = np.vstack((repetitions, repetition))
            
                repetitions_all_depths.append(repetitions)
                condition_list.append(condition)
                
        
    # evaluate what would happen if the network adheres to specific strategies
    if special_conditions == True:
        for condition in ["Input","Only_black","Only_white","Occupancy","Random"]:
            repetitions = []

            if condition == "Random":
                n_repeats = 50
            else:
                n_repeats = 1

            for rep in range(n_repeats):
                single_rep = {"special_condition": condition, "batch_size": 50, "counter": rep}
                # create responses special conditions and evaluate them   
                evaluate_special_conditions(results_dir, single_rep, do_plotting = False)
                if len(repetitions) == 0:
                    if n_repeats == 50:
                        repetitions = single_rep
                    else:
                        repetitions = [single_rep]
                else:
                    repetitions = np.vstack((repetitions, single_rep))
                    
            repetitions_all_depths.append(repetitions)
            condition_list.append("special_condition")  
            

    # analyses on the results after the models are trained
    pickle_info = do_accuracy_evaluations(results_dir, repetitions_all_depths, condition_list = condition_list,elem_idx = 118)
    
    # look at the mono & tuned proportions
    pickle_info = do_mono_tuned_fit_evaluations(results_dir,repetitions_all_depths,control_condition_list=condition_list,threshold =0.2,x0=["duration","ISI"],y0=["period","period"])
    do_mono_tuned_fit_evaluations(results_dir,repetitions_all_depths,control_condition_list=condition_list,threshold =0.8,x0=["duration","ISI"],y0=["period","period"])

    # look at the parameters
    compare_maps_tuned_predictions(results_dir,repetitions_all_depths,assessed_per="event",x0=["duration","ISI"],y0=["period","period"],control_condition_list=condition_list, threshold = 0.2,parameters_type = ["mono","tuned"])

    # # look what happens without tuned nodes
    # switch_off_nodes_last_layer(results_dir,repetitions_all_depths)
