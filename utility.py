import os
import numpy as np
import torch
import pandas as pd
import config_file as c
from itertools import cycle, islice
import pickle
import shutil


class Clipper(object):
    '''
    to limit the hidden weights. This was recommended in the IndRNN paper, however
    their code said it was not necessary for short sequences (<100 time steps is short)
    '''

    def __init__(self,  min_value, max_value):
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight_hh'):
            w = module.weight_hh.data
            torch.clamp(w, self.min_value, self.max_value, out=w)


def flatten_movie(movie):
    """
    Convert movie into vector representation.
    :param movie: 3d movie (tensor or numpy array)
    :return: flat 2d movie
    """
    return movie.reshape((movie.shape[0], -1)) if len(movie.shape) > 2 else movie


def unravel_movie(flat_movie):
    """
    Convert flat, vector representation to movie representation
    :param flat_movie: 2d flat movie (tensor or np array)
    :return: 3d unraveled movie
    """
    return flat_movie.reshape((-1, int(np.sqrt(flat_movie.shape[1])), int(np.sqrt(flat_movie.shape[1])))) if len(
        flat_movie.shape) < 3 else flat_movie


def to_tensor(arr):
    """
    Convert flat vector representation of movie into float32 pytorch tensor.
    :param arr: numpy array of any shape
    :return: float32 tensor
    """

    return torch.from_numpy(arr.astype(np.float32))


def to_numpy(tensor):
    """
    Convert flat tensor representation of movie into numpy array on cpu.
    :param tensor: tensor of any shape
    :return: np array on cpu
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


def transform_movie(movie):
    """
    Convert 3d movie into flat tensor representation.
    :param movie: 3d np array (e.g. shape: (40, 75, 75))
    :return: 2d tensor (e.g. shape: (40, 75 * 75))
    """
    return to_tensor(flatten_movie(movie))


def transform_batch(movies, batch_size):
    """
    Convert 4d series of np array movies into 4d batched flat tensor.
    Number of movies must be dividable by batch size
    :param movies: 4d np array (e.g. shape (10, 40, 75, 75))
    :param batch_size: batch size of resulting tensor
    :return: 4d tensor (e.g. shape (2, 5, 40, 75*75))
    """
    assert (movies.shape[0] % batch_size == 0)
    flat_np = movies.reshape(movies.shape[0], movies.shape[1], -1)
    flat_tensor = to_tensor(flat_np)
    return flat_tensor.reshape(batch_size, -1, flat_tensor.shape[1], flat_tensor.shape[2]).squeeze()


def restore_movie(transformed_movie):
    """
    Converts single flattened tensor movie into original numpy movie on cpu.
    :param transformed_movie: flattened tensor (e.g. shape (40, 75 * 75))
    :return: numpy movie (e.g. shape (40, 75, 75))
    """
    return unravel_movie(to_numpy(transformed_movie))


def restore_batch(transformed_batch):
    """
    Converts flat batch of tensors into numpy batch of movies.
    :param transformed_batch: flat batch of tensors (e.g. shape (5, 40, 75 * 75))
    :return: numpy movies (e.g. shape (5, 40, 75, 75))
    """
    return np.array([unravel_movie(movie) for movie in to_numpy(transformed_batch)])


def get_result_dirs(base_dir, args, control_condition=None):
    """
    Returns all used results directories
    :param base_dir: base results directory
    :param args: arguments manin was called with
    :return: exp_dir, sub_results_dir
    """

    # setup directory for this experimental configuration
    exp_dir = c.NET_TYPE + "_"
    if control_condition == "no_recurrency":
        exp_dir = "NN_"

    if "special_condition" in list(args.keys()):
        exp_dir = ""

    regular_rnn = False
    for key, value in args.items():
        if key in args.keys():  # c.CONFIGURATIONS[0].keys():
            if key == "assessed_per" or key == "cross_v":
                continue
            if key == "ind_rnn" and value == False:
                regular_rnn = True

            if regular_rnn == True and (key == "weight_constraint" or key == "DanielsRNN"):
                continue

            exp_dir += key
            exp_dir += "_"
            exp_dir += str(value)
            exp_dir += "_"

    exp_dir = exp_dir.replace('counter', 'state_training_adult_counter')

    if c.SCRAMBLE_SHIFTED == True:
        exp_dir = exp_dir.replace(
            'scrambled_control_True_', 'scrambled_control_AFTERSHIFT_')

    exp_dir = os.path.join(base_dir, exp_dir)
    if control_condition == "init":
        exp_dir = exp_dir + "init"
    sub_results_dir = os.path.join(exp_dir, "results")
    if control_condition == "init" and not os.path.exists(sub_results_dir):
        os.makedirs(sub_results_dir)

    if "special_condition" in list(args.keys()) and not os.path.exists(sub_results_dir):
        os.makedirs(sub_results_dir)

    return exp_dir, sub_results_dir


def get_result_paths(base_dir, args, control_condition=None, x0="duration", y0="period"):
    """
    Returns all used results paths :
        base_dir: base results directory 
        repetitions: arguments main was called
        control_condition: condition of the repetition you are looking for 
            (e.g. shuffled, init)
        x0: x factor for timing space
        y0: y factor for timing space

    with :return: 
       net_path, test_results_path, event_response_path, movie_response_path, 
       response_labels_path, mon_fits_path, tun_fits_path, state_change_response_path
    """
    exp_dir, sub_results_dir = get_result_dirs(
        base_dir, args, control_condition)

    # dirs where responses are stored
    net_path = os.path.join(exp_dir, "net")
    test_results_path = os.path.join(
        sub_results_dir, "new_testset_results.npy")
    response_labels_path = os.path.join(
        sub_results_dir, "new_response_labels.npy")
    event_response_path = os.path.join(
        sub_results_dir, "new_event_responses.npy")
    movie_response_path = os.path.join(
        sub_results_dir, "new_movie_responses.npy")
    state_change_response_path = os.path.join(
        sub_results_dir, "new_state_change_responses.npy")

    # result dirs where model fitting (tuned/ mono) are stored
    mono_string = 'new_event_definition_normalized_new_curve_fit_'
    tuned_string = 'new_event_definition_normalized_new_curve_fit_limited_sigmas_'
    if c.NEGATIVE == True:
        mono_string = mono_string + 'negative_'
        tuned_string = tuned_string + 'negative_'        
    if "cross_v" in args and args["cross_v"]:
        mono_string += 'cv_'
        tuned_string += 'cv_'
    if "assessed_per" in args and args["assessed_per"] == "event":
        mono_string += 'event_'
        tuned_string += 'event_'
    if x0 != "duration" or y0 != "period":
        mono_string += "x0_" + x0 + "_y0_" + y0 + "_"
        tuned_string += "x0_" + x0 + "_y0_" + y0 + "_"
    mono_string += 'mono_fits.csv'
    tuned_string += 'tuned_fits.csv'

    mon_fits_path = os.path.join(sub_results_dir, mono_string)
    tun_fits_path = os.path.join(sub_results_dir, tuned_string)

    return net_path, test_results_path, event_response_path, movie_response_path, response_labels_path, mon_fits_path, tun_fits_path, state_change_response_path


def save_stats(pickle_info, name_info, overwrite=False):

    save_dir = os.path.join(c.STATS_DIR, name_info['results_section'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pickle_save_name = 'pkl_info'
    keys_and_values = [
        item
        for k, v in name_info.items()
        if k != "results_section"
        for item in (k, v)
    ]
    for item in keys_and_values:
        if type(item) is list:
            for i in item:
                pickle_save_name = pickle_save_name + '_' + str(i)
        else:
            pickle_save_name = pickle_save_name + '_' + str(item)
    pickle_save_name += '.pkl'

    if len(pickle_info) > 0:
        if os.path.exists(os.path.join(save_dir, pickle_save_name)) and overwrite == False:
            print('file ', pickle_save_name, ' already exists. Not overwriting')
        else:
            with open(os.path.join(save_dir, pickle_save_name), 'wb') as f:
                pickle.dump(pickle_info, f)

        config_save_name = pickle_save_name.replace(
            'pkl_', 'config_').replace('.pkl', '.py')
        if os.path.exists(os.path.join(save_dir, config_save_name)) and overwrite == False:
            print('file ', config_save_name, ' already exists. Not overwriting')
        else:
            config_info = os.path.join(c.DIR_PATH, 'config_file.py')
            shutil.copy(config_info, os.path.join(save_dir, config_save_name))

    fig_save_name = pickle_save_name.replace(
        'pkl_info_', 'fig_').replace('.pkl', '.pdf')
    return save_dir, fig_save_name


def get_dataset_paths(base_dir, random_phase, events_per_set, scrambled_control):
    """
    Returns dataset path
    :param base_dir: base dataset directory
    :param dataset_idx: dataset index (int for perfect information else None)
    :param random_phase: random phase
    :param events_per_set: equal events per set
    :return: str dataset path
    """
    if scrambled_control == False:
        dataset_dir = os.path.join(
            base_dir, f"random_phase_{random_phase}_events_per_set_{events_per_set}.npy")
    else:
        dataset_dir = os.path.join(
            base_dir, f"random_phase_{random_phase}_events_per_set_{events_per_set}_scrambled_control.npy")
        if c.SCRAMBLE_SHIFTED == True:
            dataset_dir = os.path.join(
                base_dir, f"random_phase_{random_phase}_events_per_set_{events_per_set}_scrambled_after_shift.npy")
    data_dir = os.path.join(dataset_dir, "data")
    annotations_path = os.path.join(dataset_dir, "annotations.csv")

    return dataset_dir, data_dir, annotations_path


def get_timing_responses(base_dir, repetitions, assessed_per="event", gate_idx=0, x0="duration", y0="period", control_condition=None):
    """
    Returns responses of given net to given dataset
    :param base_dir: base results directory
    :param repetitions: arguments main was called with
    :return: tuple average response shape: (num_splits, num_timings, num_layers, num_hidden), timing_labels shape: (num_timings, 2)
    """

    # responses of the whole dataset saved in NOT test_results
    _, _, event_res_path, movie_res_path, res_labels_path, _, _, state_change_res_path = get_result_paths(
        base_dir, repetitions, control_condition)

    if assessed_per == "event":
        res_path = event_res_path
    elif assessed_per == "movie":
        res_path = movie_res_path
    elif assessed_per == "state_change":
        res_path = state_change_res_path

    responses = np.load(res_path)
    response_labels = np.load(res_labels_path)
    responses = responses[:, :, gate_idx, :]

    num_dataset, num_layers, num_hidden = responses.shape
    labels_rep = np.tile(response_labels, (num_layers, 1))
    layer_rep = np.tile(np.arange(num_layers, dtype=int),
                        (num_dataset, 1)).T.reshape(-1, 1)

    # long
    responses = responses.transpose(1, 0, 2).reshape(-1, num_hidden)
    data = np.concatenate((responses, labels_rep), axis=1)
    data = np.concatenate((data, layer_rep), axis=1)

    ISI = labels_rep[:, 1]-labels_rep[:, 0]
    ISI = ISI.reshape(len(ISI), 1)
    data = np.concatenate((data, ISI), axis=1)

    occupancy = labels_rep[:, 0]/labels_rep[:, 1]
    occupancy = occupancy.reshape(len(occupancy), 1)
    data = np.concatenate((data, occupancy), axis=1)
    
    occupancyISI = (labels_rep[:, 1]-labels_rep[:, 0])/labels_rep[:, 1]
    occupancyISI = occupancyISI.reshape(len(occupancyISI), 1)
    data = np.concatenate((data, occupancyISI), axis=1)

    neuron_names = list(np.arange(num_hidden).astype(str))
    col_names = neuron_names + ["duration",
                                "period", "layer", "ISI", "occupancy","occupancyISI"]
    frame = pd.DataFrame(data, columns=col_names)

    # drop what you don't need from dataframe
    if x0 == "duration" and y0 == "period":
        frame = frame.loc[:, frame.columns != 'ISI']
        frame = frame.loc[:, frame.columns != 'occupancy']
        frame = frame.loc[:, frame.columns != 'occupancyISI']
    elif x0 == "duration" and y0 == "ISI":
        frame = frame.loc[:, frame.columns != 'period']
        frame = frame.loc[:, frame.columns != 'occupancy']
        frame = frame.loc[:, frame.columns != 'occupancyISI']
    elif x0 == "occupancy" and y0 == "period":
        frame = frame.loc[:, frame.columns != 'ISI']
        frame = frame.loc[:, frame.columns != 'duration']
        frame = frame.loc[:, frame.columns != 'occupancyISI']
    elif x0 == "occupancyISI" and y0 == "period":
        frame = frame.loc[:, frame.columns != 'ISI']
        frame = frame.loc[:, frame.columns != 'duration']
        frame = frame.loc[:, frame.columns != 'occupancy']
    elif x0 == "ISI" and y0 == "period":
        frame = frame.loc[:, frame.columns != 'duration']
        frame = frame.loc[:, frame.columns != 'occupancy']
        frame = frame.loc[:, frame.columns != 'occupancyISI']

    frame = frame.sort_values(by=[x0, y0, "layer"])
    # mean because we have several stimuli with the same timing
    grouped = frame.groupby([x0, y0, "layer"])

    average = grouped.mean()
    response_labels = np.array(list(average.index))[:, :2]
    response_labels = response_labels.reshape(-1, num_layers, 2)[:, 0, :]

    if "cross_v" in repetitions.keys() and repetitions["cross_v"]:

        # how many phases of this timing exist
        count_np = grouped.count()["0"].to_numpy()

        # only for one layer
        count_np_layer_0 = count_np[::num_layers]
        even_or_odd = np.asarray([0, 1])

        split_according_groups = []
        for rep_nr in count_np_layer_0:
            for layer in range(num_layers):
                split_according_groups.append(
                    list(islice(cycle(even_or_odd), rep_nr)))

        split_according_groups = np.hstack(
            np.array(split_according_groups, dtype=object))
        frame["split"] = split_according_groups

        grouped_split = frame.groupby(["split", x0, y0, "layer"])
        average_split = grouped_split.mean()
        responses = average_split.to_numpy().reshape(2, -1, num_layers, num_hidden)

    else:

        responses = average.to_numpy().reshape(1, -1, num_layers, num_hidden)

    return responses, response_labels


def get_raw_responses(net, test_movie, mode=0):
    """
    Returns raw responses obtained from forward_scratch
    :param net: network
    :param test_movie: optional test_movie (first frame on)
    :param mode: 0 = first frame on, 1 = all frames on, 2 = all frames off
    :return: raw responses (num_layers, num_gates, num_batches, num_seq, num_hidden)
     e.g. shape (2, 6, 30, 60, 64)
    """

    if isinstance(test_movie, np.ndarray):
        test_movie = transform_movie(test_movie).unsqueeze(0).to(c.DEVICE)

    test_movie = test_movie.to(c.DEVICE)

    net.eval()
    with torch.no_grad():
        out, gates = net(test_movie, None)
        gates = gates.unsqueeze(1)

        res = to_numpy(gates)
        return res
    


def get_avg_res(batch_response, batch_labels, batch_events, batch_losses, batch_movies, batch_targets, batch_predictions=None, assessed_per="event", per_location=False,return_mask=False):
    """
    Returns average event response and loss for a batch. Not complete event will be excluded and result in both.
    :param batch_response: raw activation
    :param batch_events: batch of events
    :param batch_losses: batch of losses
    :return: tuple (avg_event_response, avg_losses)
    """

    avg_losses = []
    avg_responses = []
    acc_props = []

    if return_mask==False:
        assert batch_predictions is not None
        batch_predictions_bw = (batch_predictions > 0.5).astype(int)
        batch_predictions_bw = batch_predictions_bw.squeeze(axis=3)
        accuracy = (batch_predictions_bw == batch_targets).astype(int).squeeze()

    if assessed_per == "event" or assessed_per == "state_change":

        np_movies = to_numpy(batch_movies)

        # NOTE THAT THIS REDUCES THE LENGTH OF THE ARRAY TO 79 INSTEAD OF 80
        # A DIFFERENCE INDUCED AT TIME STEP 3 WILL THUS CREATE A DIFFERENCE IN
        # THE ARRAY AT LOCATION 2
        difference_values = np.diff(np_movies, axis=1)
        state_changes = np.where(difference_values != 0)
        state_change_cut = np.unique(state_changes[0], return_index=True)[1]
        split_state_changes = np.split(state_changes[1], state_change_cut)[1:]

        # time step for which the third state change is inputted
        # we need to add 1 because we computed the difference (which reduced the length
        # of the array).
        input_third_state_change = [
            batch_movie[2] + 1 for batch_movie in split_state_changes]

        # from the third state change input, we can predict events accurately
        # we then want to start counting at the first PREDICTION state change
        # from that frame on
        # Since the difference array is basically shifted over the movie (like I mean the reduced length)
        # it indicates state changes at the frames of the predictions
        # so no + 1 or -1 is needed
        first_frame_prediction_whole_event = np.array(
            [min(val for val in sublist if val >= target) for sublist, target in zip(split_state_changes, input_third_state_change)])

        # A STATE CHANGE ON THE LAST PREDICTION TIME STEP IS NOT CAPTURED
        # no problem, because if this is the start of an event, the whole event is never in
        # the prediction anymore: events are at least 2 frames
        # accounted for with state_change_type_2

        np_batch_labels = to_numpy(batch_labels)
        amount_frames_duration = (
            np_batch_labels[:, 0]/(1000/c.FPS)).astype(int)
        amount_frames_period = (np_batch_labels[:, 1]/(1000/c.FPS)).astype(int)
        amount_frames_ISI = amount_frames_period - amount_frames_duration

        mask = np.zeros(np.squeeze(np_movies, axis=2).shape, dtype=bool)

        num_events = (c.MOVIE_LENGTH -
                      first_frame_prediction_whole_event)//amount_frames_period
        if any(len_elem < 1 for len_elem in num_events):
            ValueError('Warning: no entire events left')

        last_frame_prediction_whole_event = first_frame_prediction_whole_event - \
            1 + amount_frames_period * num_events

        # for computing state change values
        state_change_type_1 = [list(range(first_frame_prediction_whole_event[frame_id], last_frame_prediction_whole_event[frame_id],
                                    amount_frames_period[frame_id])) for frame_id in range(len(first_frame_prediction_whole_event))]

        # if first predicted state change to on: difference 0 --> 1 = 1
        # add frames duration to arrive at next state change
        # if first predicted state change to off: difference 1 --> 0 = -1
        # add frames ISI to arrive at next state change
        first_pred_change_diff_value = [int(difference_values[id_movie, id_frame, :])
                                        for id_movie, id_frame in enumerate(first_frame_prediction_whole_event)]
        first_state = [amount_frames_duration[id_movie] if first_pred_change_diff_value[id_movie] ==
                       1 else amount_frames_ISI[id_movie] for id_movie in range(len(first_pred_change_diff_value))]
        state_change_type_2 = [list(range(first_frame_prediction_whole_event[id_movie] + first_state[id_movie], last_frame_prediction_whole_event[id_movie] +
                                    1, amount_frames_period[id_movie])) for id_movie in range(len(first_frame_prediction_whole_event))]
        state_changes_locations = [[x for id_frame in range(len(state_change_type_1[movie_id])) for x in [
            state_change_type_1[movie_id][id_frame], state_change_type_2[movie_id][id_frame]]] for movie_id in range(len(state_change_type_1))]

        # loop necessary because number of relevant images differ for timings
        for idx, elem_events in enumerate(np.squeeze(np_movies)):

            if assessed_per == "event":
                mask[idx][first_frame_prediction_whole_event[idx]:last_frame_prediction_whole_event[idx]] = 1
                if return_mask == True:
                    continue

                elem_activation = batch_response[:, :, idx, mask[idx], :]
                # average over relevant whole events
                avg_event_response = np.sum(elem_activation, axis=2) / num_events[idx]

                avg_event_loss = np.sum(batch_losses[idx][mask[idx]]) / num_events[idx]
                avg_responses.append(avg_event_response)
                avg_losses.append(avg_event_loss)

                masked_accuracy = accuracy[idx, mask[idx]]
                avg_acc_prop = np.sum(masked_accuracy/(num_events[idx]*amount_frames_period[idx]))

                acc_props.append(avg_acc_prop)

            elif assessed_per == "state_change":
                mask[idx][state_changes_locations[idx]] = 1
                if return_mask == True:
                    continue
                elem_activation = batch_response[:, :, idx, mask[idx], :]

                # average over relevant images
                avg_state_change_response = np.sum(elem_activation, axis=2) / num_events[idx]
                avg_state_change_loss = np.sum(batch_losses[idx, mask[idx]]) / num_events[idx]
                avg_acc_prop = np.sum(accuracy[idx, mask[idx]]/(num_events[idx]*2))

                avg_responses.append(avg_state_change_response)
                avg_losses.append(avg_state_change_loss)
                acc_props.append(avg_acc_prop)

    # MOVIE INFORMATION IS NOT FROM ONSET FULL INFO AND ONLY WHOLE EVENTS CURRENTLY
    # PROBABLY CHANGE IF YOU WANT TO USE THIS
    # right now we are only using per-event evaluations
    elif assessed_per == "movie":
        # loop over each element of batch and mean movie
        try:
            for idx in range(batch_response.shape[2]):
                avg_responses.append(
                    np.mean(batch_response[:, :, idx, ...], axis=2))
                avg_losses.append(
                    np.mean(batch_response[:, :, idx, ...], axis=2))
        except:
            pass
    
    if return_mask == True:
        return mask

    avg_responses = np.array(avg_responses)
    avg_losses = np.array(avg_losses)

    if assessed_per == "event" or assessed_per == "state_change":

        return avg_responses, avg_losses, acc_props
    else:
        return avg_responses, avg_losses
