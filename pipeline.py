import os.path
from torch.utils.data import DataLoader
from scipy.stats import kruskal, shapiro,mannwhitneyu, wilcoxon
from scikit_posthocs import posthoc_dunn

from generate_dataset import EventTimingDataset
from plotting import plot_response_heatmaps, plot_performance_heatmap, \
    plot_train_loss, show_prediction_grid_rows
from utility import to_numpy, get_result_paths, Clipper
import torch
import numpy as np
import pickle
from utility import get_avg_res, get_timing_responses,save_stats
import config_file as c
from tqdm import tqdm
import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from create_special_conditions import evaluate_special_conditions
from network_models_new import RNNetwork, NNetwork





def train_net(net, dataloader, optimizer, scheduler, loss_fn, num_epochs, net_path, early_stopping=False, prev_losses=None, clip_weights=False):
    """
    Trains given network for given hyper parameters
    :param net: RNN or NN
    :param dataloader: dataloader
    :param optimizer: optimizer
    :param scheduler: scheduler lr
    :param loss_fn: loss function
    :param num_epochs: total number of epochs to train for
    :param net_path: where network state is saved
    :return: mean losses for each epoch and last trained epoch (int)
    """
    net = net.to(c.DEVICE)

    if prev_losses is None:
        losses = []
        prev_epochs = 0
    else:
        prev_epochs = len(prev_losses)
        losses = list(prev_losses)

    if clip_weights:
        clipper = Clipper(min_value = c.RWEIGHT_CONSTRAINT_MIN, max_value=c.RWEIGHT_CONSTRAINT)
    else:
        clipper = None

    for epoch_count in tqdm(range(num_epochs)):
        batch_losses = []

        for batch_movies, batch_targets, _, _ in dataloader:

            # movies straightened out now 
            # not a problem now, might be problematic if we want to use spatial location
            net_input = batch_movies.to(c.DEVICE)
            target = batch_targets.to(c.DEVICE)


            with torch.autocast(enabled=c.USE_AMP, device_type=str(c.DEVICE)):
                # Forward pass
                net_outs, _ = net(net_input, None)
               
                # average over all axis for backprop
                loss = loss_fn(net_outs, target).mean()        
               

            # Update network 
            loss.backward()
            optimizer.step()

            if clipper is not None: 
                net.apply(clipper)

            # reset gradients
            optimizer.zero_grad(set_to_none=True)

            # Save batch loss
            batch_losses.append(loss.item())
            
        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)

        # save checkpoint every 50 epochs
        if (prev_epochs + epoch_count) % 50 == 0:
            print(f"Checkpoint save epoch {prev_epochs + epoch_count}")
            print(f"Avg epoch loss: {epoch_loss}")
            state = {
                'model_state_dict': net.state_dict(),
                'loss': np.array(losses),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            if (prev_epochs + epoch_count) % 1000 == 0:
                save_state_of_net_path = net_path.replace('_/net','_epoch{}/net'.format(prev_epochs + epoch_count))
                os.makedirs(save_state_of_net_path.replace('epoch{}/net'.format(prev_epochs + epoch_count),'epoch{}'.format(prev_epochs + epoch_count)))
                torch.save(state, save_state_of_net_path)
            torch.save(state, net_path)
            

        if early_stopping and (prev_epochs + epoch_count) > 100:
            # if average loss over last 50 epochs did not increase by 1 percent, break
            if np.mean(losses[epoch_count - 100: epoch_count - 50]) / np.mean(losses[epoch_count - 50: epoch_count]) < 1.01:
                break
            

        epoch_count += 1
        scheduler.step()

    return np.array(losses)


def predict_movie(net, batch_movies, batch_targets, loss_fn):
    """
    Predicts movies for given batched tensor
    :param net: RNN or NN network
    :param batch_movies: flat input tensor of shape (bs, seq_len, image_height * image_width)
    :param batch_targets: flat input tensor of shape (bs, seq_len, image_height * image_width)
    :param loss_fn: loss function
    :return: tuple(batch_losses, batch_predictions, batch_targets, net_state) - everything in numpy
    """
    net = net.to(c.DEVICE)
    batch_movies = batch_movies.to(c.DEVICE)
    batch_targets = batch_targets.to(c.DEVICE)

    with torch.no_grad():
        # Forward pass
        net_out, state = net(batch_movies, None)
    
        loss = loss_fn(net_out, batch_targets)
        
        #average over movie length
        batch_losses = torch.mean(loss, dim=2)

        # Squeeze batches
        unsqueezed_state = state.unsqueeze(1)

        # predictions also in a linear shape. Only later reshaped
        # mainly important if >1 pixel
        batch_predictions = net_out.reshape(net_out.shape[0], net_out.shape[1], c.IMAGE_WIDTH, c.IMAGE_HEIGHT)

    return to_numpy(batch_losses), to_numpy(batch_predictions), to_numpy(batch_targets), to_numpy(state), to_numpy(unsqueezed_state)


def predict_dataset(net, dataloader, loss_fn, return_movies=True):
    """
    Predicts movies for a whole dataset given by dataloader
    :param net: RNN or NN network
    :param dataloader: torch.DataLoader containing dataset
    :param loss_fn: loss function
    :param return_movies: if True return 2d numpy arrays else flat torch tensors
    :return: tuple(losses, event_losses, predictions, targets, fit_labels, events, avg_event_responses)
    """
    net = net.to(c.DEVICE)
    losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, avg_event_responses, avg_movie_responses, avg_state_change_responses, acc_proportions_event,acc_proportions_state_change = [], [], [], [], [], [], [], [], [], [], [], [],[],[]
    dataset_len = len(dataloader.dataset)
    
    # loop over all batches
    for batch_idx, (batch_movies, batch_targets, batch_labels, batch_events) in enumerate(dataloader):
        # init
        bs, seq_len, _ = batch_movies.shape

        batch_losses, batch_predictions, batch_targets, net_state, responses = predict_movie(net, batch_movies, batch_targets, loss_fn)
        losses.append(batch_losses)
        if return_movies:
            predictions.append(batch_predictions)
            targets.append(batch_targets)
            movies.append(batch_movies)
        labels.append(batch_labels)
        events.append(batch_events)
        
        
        # everything for this has to be np and on cpu 
        avg_event_res, avg_event_loss, acc_prop_event = get_avg_res(responses, batch_labels, batch_events,batch_losses, batch_movies, batch_targets,batch_predictions,"event")
        avg_state_change_res, avg_state_change_loss, acc_prop_state_change = get_avg_res(responses, batch_labels, batch_events,batch_losses,  batch_movies,batch_targets,batch_predictions,"state_change")
        avg_movie_res, avg_movie_loss = get_avg_res(responses, batch_labels, batch_events,batch_losses, batch_movies,batch_targets,batch_predictions,"movie")

        avg_event_responses.append(avg_event_res)
        avg_state_change_responses.append(avg_state_change_res)
        avg_movie_responses.append(avg_movie_res)

        acc_proportions_event.append(acc_prop_event)
        acc_proportions_state_change.append(acc_prop_state_change)
        
        event_losses.append(avg_event_loss)
        state_change_losses.append(avg_state_change_loss)
        movie_losses.append(avg_movie_loss)

  
    # reshape
    losses = np.concatenate(losses, axis=0)
    event_losses = np.concatenate(event_losses, axis=0)
    state_change_losses = np.concatenate(state_change_losses, axis=0)
    acc_proportions_state_change = np.concatenate(acc_proportions_state_change,axis=0)
    acc_proportions_event = np.concatenate(acc_proportions_event,axis=0)

    if return_movies:
        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions.reshape(dataset_len, seq_len, c.IMAGE_WIDTH, c.IMAGE_HEIGHT)
        targets = np.concatenate(targets, axis=0)
        targets = targets.reshape(dataset_len, seq_len, c.IMAGE_WIDTH, c.IMAGE_HEIGHT)
        movies = np.concatenate(movies, axis=0)
        movies = movies.reshape(dataset_len, seq_len, c.IMAGE_WIDTH, c.IMAGE_HEIGHT)
    labels = np.concatenate(labels, axis=0)
    events = np.concatenate(events, axis=0)
    avg_event_responses = np.concatenate(avg_event_responses, axis=0)
    avg_state_change_responses = np.concatenate(avg_state_change_responses, axis=0)
    avg_movie_responses = np.concatenate(avg_movie_responses, axis=0)

    return losses, event_losses, state_change_losses, movie_losses, np.array(predictions), np.array(targets), labels, events, np.array(movies), avg_event_responses, avg_state_change_responses, avg_movie_responses, acc_proportions_event, acc_proportions_state_change


def evaluate_net(results_dir, repetitions, assessed_per="event",x0="duration",y0="period", do_plotting = True,control_condition=None):
    """
    Evaluates net by showing different plots
    :param results_dir: base results directory
   
    x0, y0 only used for plotting
    
    :return:
    """

    net_path, test_results_path, event_response_path, movie_response_path, response_labels_path, mon_fits_path, tun_fits_path, state_change_response_path = get_result_paths(results_dir, repetitions,control_condition)
    
    
    ########################################################### whole dataset

    # evaluations on the whole dataset
    # these are the activations I'll use for model fitting
    if not os.path.exists(event_response_path) or not os.path.exists(state_change_response_path) or not os.path.exists(movie_response_path) or not os.path.exists(response_labels_path):

        event_responses, res_labels = None, None

        # load network
        if 'weight_constraint' not in repetitions:
            w_c = None
        else:
            w_c = repetitions['weight_constraint']
            
        if control_condition == "no_recurrency" or c.NET_CLASS == NNetwork:
            c.NET_CLASS = NNetwork
            c.NET_TYPE = "NN"
        else:
            c.NET_CLASS = RNNetwork
            c.NET_TYPE = "RNN"
            
        net = c.NET_CLASS(c.INPUT_SIZE, repetitions["num_hidden"], repetitions["num_layers"], c.NONLINEARITY, c.BIAS, c.BATCH_FIRST, repetitions["dropout_prob"], repetitions["norm"], repetitions["ind_rnn"], w_c)
        checkpoint = torch.load(net_path, map_location=torch.device(c.DEVICE))
        net.load_state_dict(checkpoint['model_state_dict'])
        if control_condition =="init":
            train_losses = []
        else:
            if type(checkpoint['loss']) is not np.ndarray:
                train_losses = to_numpy(checkpoint['loss'])
            else:
                train_losses = checkpoint['loss']
    
        # put in eveluation mode
        net.eval()
        
        # analyse activations on full dataset
        generator = torch.Generator().manual_seed(c.SEED)
        dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                      events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                      cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control = False)
        
        full_dataloader = DataLoader(dataset, batch_size=repetitions["batch_size"], shuffle=False, pin_memory=False,
                                      num_workers=0)
        _, _, _, _, _, _, res_labels, _, _, event_responses, state_change_responses, movie_responses,_,_ = predict_dataset(net, full_dataloader, c.LOSS_FN,
                                                                      return_movies=False)
        
        np.save(event_response_path, event_responses)
        np.save(state_change_response_path, state_change_responses)
        np.save(movie_response_path, movie_responses)
        np.save(response_labels_path, res_labels)
        
    # want to analyze the responses on the whole dataset
    # plotting for one split right now, for now not an issue
    if do_plotting == True:
        if repetitions['cross_v']==True:
            not_cv_repetitions = repetitions.copy()
            not_cv_repetitions['cross_v'] = False
            timing_res, timing_labels = get_timing_responses(results_dir, not_cv_repetitions, assessed_per,x0=x0,y0=y0,control_condition=control_condition)
        else:
            timing_res, timing_labels = get_timing_responses(results_dir, repetitions, assessed_per,x0=x0,y0=y0,control_condition=control_condition)

        # # activations heatmap
        plot_response_heatmaps(timing_res, timing_labels, 0,x0,y0)        
        
    
    ########################################################### Test
        
    # evaluations on the test set
    if not os.path.exists(test_results_path):
        # load model
        if 'weight_constraint' not in repetitions:
            w_c = None
        else:
            w_c = repetitions['weight_constraint']
            
        if control_condition == "no_recurrency" or c.NET_CLASS == NNetwork:
            c.NET_CLASS = NNetwork
            c.NET_TYPE = "NN"
        else:
            c.NET_CLASS = RNNetwork
            c.NET_TYPE = "RNN"
            
        net = c.NET_CLASS(c.INPUT_SIZE, repetitions["num_hidden"], repetitions["num_layers"], c.NONLINEARITY, c.BIAS, c.BATCH_FIRST, repetitions["dropout_prob"], repetitions["norm"], repetitions["ind_rnn"], w_c)
        checkpoint = torch.load(net_path, map_location=torch.device(c.DEVICE))
        net.load_state_dict(checkpoint['model_state_dict'])
        if control_condition =="init":
            train_losses = []
        else:
            if type(checkpoint['loss']) is not np.ndarray:
                train_losses = to_numpy(checkpoint['loss'])
            else:
                train_losses = checkpoint['loss']
                
        net.eval()
        
        # can be done on unscrambled data set since the split has the same indexing (we set the seed)
        generator = torch.Generator().manual_seed(c.SEED)
        dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                     events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                     cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control=False)
        trainset_size = int(len(dataset) * c.TRAIN_SPLIT_RATIO)
        train_set, test_set = torch.utils.data.random_split(dataset, [trainset_size, len(dataset) - trainset_size],
                                                            generator)
        eval_dataloader = DataLoader(test_set, batch_size=repetitions["batch_size"], shuffle=False, num_workers=0)
    
        net.to(c.DEVICE)
        losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, _, _, _, acc_proportions_event, acc_proportions_state_change = predict_dataset(net, eval_dataloader, c.LOSS_FN, return_movies=True)
    
        table = [train_losses, losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, acc_proportions_event, acc_proportions_state_change]
        with open(test_results_path, "wb") as f:
            pickle.dump(table, f)

    if do_plotting == True:
        with open(test_results_path, 'rb') as f:
            # get concatenated losses and fit_labels
            train_losses, losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, acc_proportions_event, acc_proportions_state_change = pickle.load(f)
            
        plot_performance_heatmap(acc_proportions_event, labels)
        plot_performance_heatmap(acc_proportions_state_change, labels)
      
    
        # average first over frames (axis 1) and then over movies (axis 0)
        norm_avg_eval_loss = np.mean([np.mean(frames) for frames in losses])
        print(f"Avg test loss: {norm_avg_eval_loss}")

        
        ########################################################### Train
        # loss during training on train set
        plot_train_loss(train_losses)

    return
    

def combined_grid_plot(results_dir, timing_idx, all_repetitions,control_condition_list = []):
    
    # predictions of pixels 
    collect_depths = []
    prediction_per_layer = {}
    
    # loop over included conditions or layer depths
    for layer_id, repetitions_per_layer in enumerate(all_repetitions):
        prediction_list = []
        if len(control_condition_list)==0:
            control_condition = None
        else:
            control_condition = control_condition_list[layer_id]
    
        for repetition in repetitions_per_layer:
            
            if len(repetition) == 1:
                repetition = repetition[0]
                
            if control_condition == "special_condition":
                evaluate_special_conditions(results_dir, repetition, assessed_per="event",x0="duration",y0="period", do_plotting = False)
            else:
                evaluate_net(results_dir, repetition,do_plotting = False,control_condition=control_condition)
        
            _, test_results_path, _, _, _, _, _, _ = get_result_paths(results_dir, repetition,control_condition=control_condition)
    
            with open(test_results_path, 'rb') as f:
                # get concatenated losses and fit_labels
                _, _, _, _, _, prediction, target, labels, _,movies, acc_proportions_event, acc_proportions_state_change = pickle.load(f)
                
            if len(prediction_list)==0:
                input_list = movies[timing_idx]
                prediction_list = prediction[timing_idx]
                target_list = target[timing_idx]
              
            else:
                prediction_list = np.hstack((prediction_list, prediction[timing_idx]))
                
        # turn predictions of each repetition into full black or full white 
        prediction_list[prediction_list > 0.5] = 1
        prediction_list[prediction_list <= 0.5] = 0 
    
        if control_condition == "special_condition":
            amount_layers = repetition['special_condition']
            collect_depths.append(None)
        elif control_condition == None:
            amount_layers = str(repetition['num_layers']) + ' layers:'
            collect_depths.append(repetition['num_layers'])
        else:
            amount_layers = control_condition + ":"
            collect_depths.append(repetition['num_layers'])
            
        # % of repetitions predicting black
        prediction_per_layer[amount_layers] = np.mean(prediction_list,axis=1)[:,None,:]
        
    
   
    # save info
    pickle_info = [repetitions_per_layer, prediction_per_layer,target_list]
    overwrite = False     
    grid_path, fig_file_name = save_stats(pickle_info, {'results_section':'gridPlot','depths':collect_depths, 'conditions': control_condition_list, 'timingID': timing_idx},overwrite=overwrite)
        
    # plot actual grid
    show_prediction_grid_rows(target_list, prediction_per_layer, os.path.join(grid_path,fig_file_name),binary_output = True)
        
    

              
def plot_loss_functions_all_depths(results_dir, all_repetitions, control_condition_list = []):
     
     stats = {}
     stats["shapiro"]={}
     
     fig = plt.figure(figsize=(len(all_repetitions)*5, len(all_repetitions)))
     all_losses = []
     all_medians = []
     all_IQRs = []
     yticks = True
     
     all_layers = []
     for rep_id, repetitions in enumerate(all_repetitions):
        if control_condition_list[rep_id] == "special_condition":
            all_layers.append(None)
            continue
        
        first_repetition = repetitions[0]
        if len(first_repetition) == 1:
            first_repetition = first_repetition[0]
        all_layers.append(first_repetition['num_layers'])
        
        
     per_depth = gridspec.GridSpec(1, len(all_repetitions),hspace=0.0)

     for depth_id,repetitions in enumerate(all_repetitions):       
        if len(control_condition_list)==0:
            control_condition = None
        else:
            control_condition = control_condition_list[depth_id]
        
        if control_condition == "special_condition":
            continue
        
        num_layers = all_layers[depth_id]

        ## training loss
        train_losses_array = []
        for repetition in repetitions:
            if len(repetition) == 1:
                repetition = repetition[0]
                
            evaluate_net(results_dir, repetition, do_plotting = False,control_condition=control_condition)
            
            _, test_results_path, _, _, _, _, _, _ = get_result_paths(results_dir, repetition, control_condition=control_condition)
    
            
            with open(test_results_path, 'rb') as f:
                # get concatenated losses and fit_labels
                train_losses, _, _, _, _, _, _,_, _, _, _, _ = pickle.load(f)
            
            if len(train_losses_array)==0:
                train_losses_array = train_losses
            else:
                train_losses_array = np.vstack((train_losses_array, train_losses))
    
        if control_condition != "init":
            
            ax = fig.add_subplot(per_depth[depth_id])
            ax.title.set_text(str(num_layers) + ' layers')
            if depth_id != 0:
                yticks = False
            
            median = np.median(train_losses_array,axis=0) 
            IQR = np.vstack((np.percentile(train_losses_array,25,axis=0), np.percentile(train_losses_array,75,axis=0)))
            plot_train_loss(IQR, close_plots = False,show_fig = False,ylabels=yticks)
            plot_train_loss(median, close_plots = False,show_fig = False,color = "red",ylabels=yticks)
            all_medians.append(median)
            all_IQRs.append(IQR)
            print(median)
            print(IQR)
            all_losses.append(train_losses_array)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            all_losses.append([])
            
     if (not any(control_condition_list) or all(np.asarray(control_condition_list) == "16nodes")) and len(control_condition_list) > 1:         
         eval_string = "kruskal(" 
         for depth_id,repetitions in enumerate(all_repetitions):
            list_losses = [all_losses[depth_id][rep_id][-1] for rep_id,_ in enumerate(repetitions)]
            stats["shapiro"] = shapiro(list_losses)
            eval_string = eval_string + str(list_losses) + ","
         eval_string = eval_string[:-1] + ")"
         stats["kruskal"]= eval(eval_string)
         print(stats["kruskal"])
        
         # post hocs    
         # event_accuracies
         # posthoc dunn test, with correction for multiple testing
         eval_string = "posthoc_dunn([" 
         for depth_id,repetitions in enumerate(all_repetitions):
            list_losses = [all_losses[depth_id][rep_id][-1] for rep_id,_ in enumerate(repetitions)]
            eval_string = eval_string + str(list_losses) + ","
         eval_string = eval_string[:-1] + "],p_adjust='holm-sidak')"
        
         stats["post_hoc"] = eval(eval_string)
         print(stats["post_hoc"])
     elif len(control_condition_list) == 1:
         pass
     else:
         stats["MannWU"] = {}
         stats["Wilcoxon"] = {}
         
         try:
             id_5 = np.where(np.array(control_condition_list) == None)[0][0]
             regular_loss = [all_losses[id_5][rep][-1] for rep in range(len(all_losses[id_5]))]
             stats["shapiro"]["regular"] = shapiro(regular_loss)
         except:
             print('no regular')
        
         # compare 5 layer network with not recurrent
         try:
             id_NN = np.where(np.array(control_condition_list) == "no_recurrency")[0][0]
             stats["MannWU"]["NN"] = mannwhitneyu(regular_loss,[all_losses[id_NN][rep][-1] for rep in range(len(all_losses[id_NN]))])
             stats["shapiro"]["NN"] = shapiro([all_losses[id_NN][rep][-1] for rep in range(len(all_losses[id_NN]))])
             print(stats["MannWU"]["NN"])
         except:
            print('no no recurrency')

         # compare 5 layer network with 16 nodes per layer
         try:
             id_16 = np.where(np.array(control_condition_list) == "16nodes")[0][0]
             stats["MannWU"]["16nodes"] = mannwhitneyu(regular_loss,[all_losses[id_16][rep][-1] for rep in range(len(all_losses[id_16]))])
             stats["shapiro"]["16nodes"] = shapiro([all_losses[id_16][rep][-1] for rep in range(len(all_losses[id_16]))])
             print(stats["MannWU"]["16nodes"])
         except:
             print('no 16 nodes')
         
         # compare 5 layer network with network trained on shuffled data
         try:
             id_shuffled = np.where(np.array(control_condition_list) == "shuffled")[0][0]
             stats["Wilcoxon"]["shuffled"] = wilcoxon(regular_loss,[all_losses[id_shuffled][rep][-1] for rep in range(len(all_losses[id_shuffled]))])
             stats["shapiro"]["shuffled_diff"] = shapiro(np.array(regular_loss) - np.array([all_losses[id_shuffled][rep][-1] for rep in range(len(all_losses[id_shuffled]))]))
             print(stats["Wilcoxon"]["shuffled"])
         except:
            print('no shuffled')
    
     # save info
     matplotlib.rcParams['pdf.fonttype'] = 42
     pickle_info = [all_repetitions, all_losses,all_medians,all_IQRs, stats]
     overwrite = False     
     loss_path, fig_file_name = save_stats(pickle_info, {'results_section':'loss_plots','depths':all_layers, 'conditions': control_condition_list},overwrite=overwrite)
     if os.path.exists(os.path.join(loss_path, fig_file_name)) and overwrite == False:
         print('file ', fig_file_name, ' already exists. Not overwriting')
     else:    
         plt.savefig(os.path.join(loss_path,fig_file_name))
     
     plt.show()
     return pickle_info
     
         
 
     
       
              
