import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader
from generate_dataset import EventTimingDataset
from plotting import plot_performance_heatmap
from utility import to_numpy, get_result_paths,to_tensor
import pickle
from utility import get_avg_res
import config_file as c


def predict_movie_no_net(batch_movies, batch_targets, loss_fn,special_condition):
    '''
        Makes predictions for the movies in the special conditions, where there 
        is no actual network outputting predictions

        made to perform similar to predict_movie in pipeline
    '''

    batch_movies = batch_movies.to(c.DEVICE)
    batch_targets = batch_targets.to(c.DEVICE)

    with torch.no_grad():
        # fake network outputs
        if special_condition == "Only_black":
            net_out = to_tensor(np.ones((len(batch_targets),len(batch_targets[0]),c.IMAGE_WIDTH*c.IMAGE_HEIGHT)))
        elif special_condition == "Only_white":
            net_out = to_tensor(np.zeros((len(batch_targets),len(batch_targets[0]),c.IMAGE_WIDTH*c.IMAGE_HEIGHT)))
        elif special_condition == "Random":
            net_out = to_tensor(1/(1+np.exp(-(np.random.uniform(-1,1,(len(batch_targets),len(batch_targets[0]),c.IMAGE_WIDTH*c.IMAGE_HEIGHT))))))            
        elif special_condition == "Input":
            net_out = batch_movies
        elif special_condition == "Occupancy":
            cumsum = np.cumsum(to_numpy(batch_movies),axis=1)
            len_so_far = np.tile(np.arange(batch_movies.shape[1])+1,(cumsum.shape[0],1))
            len_so_far = len_so_far[:,:,None]
            net_out = to_tensor(cumsum/ len_so_far)
            
        # there is no real network so there is no state
        state = net_out[None,:]
        unsqueezed_state = state.unsqueeze(1)
    
        loss = loss_fn(net_out, batch_targets)
        
        #average over movie length
        batch_losses = torch.mean(loss, dim=2)
        batch_predictions = net_out.reshape(net_out.shape[0], net_out.shape[1], c.IMAGE_WIDTH, c.IMAGE_HEIGHT)

    return to_numpy(batch_losses), to_numpy(batch_predictions), to_numpy(batch_targets), to_numpy(state), to_numpy(unsqueezed_state)



def predict_dataset_no_net(dataloader, loss_fn, special_condition, return_movies=True):
    '''
        Makes predictions for the whole dataset in the special conditions, where there 
        is no actual network outputting predictions

        made to perform similar to predict_dataset in pipeline
    '''
    
    losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, avg_event_responses, avg_movie_responses, avg_state_change_responses, acc_proportions_event,acc_proportions_state_change = [], [], [], [], [], [], [], [], [], [], [], [],[],[]
    dataset_len = len(dataloader.dataset)
    
    # loop over all batches
    for batch_idx, (batch_movies, batch_targets, batch_labels, batch_events) in enumerate(dataloader):
        # init
        bs, seq_len, _ = batch_movies.shape

        batch_losses, batch_predictions, batch_targets, net_state, responses = predict_movie_no_net(batch_movies, batch_targets, loss_fn,special_condition)
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
        
        # responses
        avg_event_responses.append(avg_event_res)
        avg_state_change_responses.append(avg_state_change_res)
        avg_movie_responses.append(avg_movie_res)

        # accuracy proportions
        acc_proportions_event.append(acc_prop_event)
        acc_proportions_state_change.append(acc_prop_state_change)
        
        # loss
        event_losses.append(avg_event_loss)
        state_change_losses.append(avg_state_change_loss)
        movie_losses.append(avg_movie_loss)

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


def evaluate_special_conditions(results_dir, repetition, assessed_per="event", do_plotting = True):
    '''
        Evaluates the "responses" that the special conditions, would produce

        made to perform similar to evaluate_net in pipeline
    '''
    
    net_path, test_results_path, event_response_path, movie_response_path, response_labels_path, mon_fits_path, tun_fits_path, state_change_response_path = get_result_paths(results_dir, repetition,None)
    special_condition = repetition["special_condition"]
    
    if special_condition == "Random":
        do_plotting = False
    ########################################################### whole dataset

    # evaluations on the whole dataset
    # these are the activations I'll use for model fitting (not done for special conditions because there is no actual network)
    if not os.path.exists(event_response_path) or not os.path.exists(state_change_response_path) or not os.path.exists(movie_response_path) or not os.path.exists(response_labels_path):
        event_responses, res_labels = None, None

        # analyse activations on full dataset
        generator = torch.Generator().manual_seed(c.SEED)
        dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                      events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                      cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control = False)
        
        full_dataloader = DataLoader(dataset, batch_size=repetition["batch_size"], shuffle=False, pin_memory=False,
                                      num_workers=0)
        _, _, _, _, _, _, res_labels, _, _, event_responses, state_change_responses, movie_responses,_,_ = predict_dataset_no_net(full_dataloader, c.LOSS_FN, special_condition,
                                                                      return_movies=True)
        
        np.save(event_response_path, event_responses)
        np.save(state_change_response_path, state_change_responses)
        np.save(movie_response_path, movie_responses)
        np.save(response_labels_path, res_labels)
             
    
    ########################################################### Test
        
    # evaluations on the test set
    if not os.path.exists(test_results_path):
                    
        # can be done on unscrambled data set since the split has the same indexing (we set the seed)
        generator = torch.Generator().manual_seed(c.SEED)
        dataset = EventTimingDataset(c.DATASET_DIR, movie_length=c.MOVIE_LENGTH,
                                     events_per_set=c.EVENTS_PER_SET, random_phase=c.RANDOM_PHASE,
                                     cache_datasets=c.CACHE_DATASETS, force_length=c.FORCE_LENGTH, scrambled_control=False)
        trainset_size = int(len(dataset) * c.TRAIN_SPLIT_RATIO)
        train_set, test_set = torch.utils.data.random_split(dataset, [trainset_size, len(dataset) - trainset_size],
                                                            generator)
        eval_dataloader = DataLoader(test_set, batch_size=repetition["batch_size"], shuffle=False, num_workers=0)
    
        
        losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, avg_event_responses, avg_state_change_responses, avg_movie_responses, acc_proportions_event, acc_proportions_state_change = predict_dataset_no_net(eval_dataloader, c.LOSS_FN, special_condition, return_movies=True)
    
        train_losses = []
        table = [train_losses, losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, acc_proportions_event, acc_proportions_state_change]
        with open(test_results_path, "wb") as f:
            pickle.dump(table, f)

    if do_plotting == True:
        with open(test_results_path, 'rb') as f:
            # get concatenated losses and fit_labels
            train_losses, losses, event_losses, state_change_losses, movie_losses, predictions, targets, labels, events, movies, acc_proportions_event, acc_proportions_state_change = pickle.load(f)
            
        plot_performance_heatmap(acc_proportions_event, labels)
        plot_performance_heatmap(acc_proportions_state_change, labels)
      

    return
    
    
