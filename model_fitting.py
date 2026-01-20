import numpy as np
from scipy.optimize import curve_fit,minimize
import pandas as pd
import config_file as c
from plotting import plot_response_heatmaps
import os
from utility import get_timing_responses, get_result_paths
from pipeline import evaluate_net
import warnings
import sympy as sp

def find_peak_tuning(guess_x, guess_y, x0, y0, sigma_x0, sigma_y0, theta, exponent, slope, intercept):
    """
    Finds peak of the tuned function

    guess_x: guess x-coordinate peak
    guess_y: guess y-coordinate peak
    x0: preferred duration
    y0: preferred period
    sigma_x0, sigma_y0, theta, exponent, slope, intercept: other parameters of the best-fitting tuned function

    returns:
    x and y coordinates of peak
    """

    def tuning_function_to_find_max(coords):
        durations, periods = coords
        
        X = ((durations - x0) * np.cos(np.deg2rad(theta))) - ((periods-y0)*np.sin(np.deg2rad(theta)))
        Y = ((durations - x0) * np.sin(np.deg2rad(theta))) + ((periods-y0)*np.cos(np.deg2rad(theta)))
        
        power = -0.5 * (((Y / sigma_y0) ** 2) + ((X / sigma_x0) ** 2))
        
        # periods are usually not negative,
        # fractional powers of negative numbers always nan, even if the power would not result in a complex number
        # still output for exponent == 1 and exponent == 0
        r =  ((sp.exp(power) * (periods / (periods ** exponent))) * slope) + intercept
        return -r
    
    initial_guess = [guess_x, guess_y]

    if exponent != 1 and exponent != 0:
        bounds = [(None,None),(1e-8,None)]
    else:
        bounds = [(None,None),(None,None)]
        
    result = minimize(tuning_function_to_find_max,initial_guess,method='Nelder-Mead',bounds =bounds)

    max_x, max_y = result.x
    
    return max_x, max_y 

def tuned_function(response_labels,  x0, y0, sigma_x0, sigma_y0, theta, exponent, slope, intercept):
    """
    Gives outcomes tuned function for each timing

    reponse_labels: timings
    x0,y0,sigma_x0, sigma_y0, theta, exponent, slope, intercept: parameters of the tuned function
    """

    warnings.simplefilter('ignore', lineno=73)
    
    durations, periods = response_labels
    
    X = ((durations - x0) * np.cos(np.deg2rad(theta))) - ((periods-y0)*np.sin(np.deg2rad(theta)))
    Y = ((durations - x0) * np.sin(np.deg2rad(theta))) + ((periods-y0)*np.cos(np.deg2rad(theta)))
    
    power = -0.5 * (((Y / sigma_y0) ** 2) + ((X / sigma_x0) ** 2))
        
    # periods are usually not negative,
    # fractional powers of negative numbers always nan, even if the power would not result in a complex number
    # still output for exponent == 1 and exponent == 0
    r =  ((np.exp(power) * (periods / (periods ** exponent))) * slope) + intercept

    return r

def mono_function(response_labels,  x0, y0, ratio_x0, ratio_y0,slope,intercept):
    """
    Gives outcomes of the monotonic function for each timing

    reponse_labels: timings
    x0, y0, ratio_x0, ratio_y0,slope,intercept: parameters of the monotonic function
    """

    warnings.simplefilter('ignore', lineno=92)

    durations, periods = response_labels
    
    # periods are usually not negative,
    # fractional powers of negative numbers always nan, even if the power would not result in a complex number
    # still output for exponent == 1 and exponent == 0
    r = ((ratio_x0 * (durations**x0) + ratio_y0 * (periods/(periods ** y0))) * slope) + intercept
    
    #NOTE IF YOU WANT TO DO THIS FOR NOT DURATION PERIOD--> OTHER MONO FUNCTION!?

    return r


def compute_variance_explained(function_type, node_activity, response_labels, params):
    """
    computes variance explained of the requested function type
    """
    
    if function_type == "tuned":
        function_to_use = tuned_function
    elif function_type == "mono":
        function_to_use = mono_function
        
    pred_activity = function_to_use(response_labels, *params) 
    
    ss_res = sum((pred_activity - node_activity)**2)
    node_mean = np.mean(node_activity) 
    ss_tot = sum((node_activity - node_mean)**2)
    variance_explained = 1 - ss_res / ss_tot

    return variance_explained


    
def fit_responses(responses, response_labels, function_type, outcomes_file=None, cross_v = True,x0="duration",y0="period",save = True):
    """
    Fits response functions to the responses of the nodes

    responses: per-event activations for each timing
    reponse_labels: each timing
    function_type: monotonic or tuned
    outcomes_file: location to save the results
    cross_v: cross-validation boolean (True: evaluate fit on other half of the data) 
    x0: timing label to use for x dimension of the response function space
    y0: timing label to use for y dimension of the response function space
    save: boolean
    """

    # parameters, start dataframe
    if function_type == "tuned":
        columns = ["node","layer","split","pref_x0", "pref_y0", "sigma_x0", "sigma_y0", "theta", "exponent","tuned_slope","tuned_intercept","tuned_VE"]    
        if cross_v == True:
            columns.append("tuned_cv_VE")
    elif function_type == "tuned_no_exp":
        columns = ["node","layer","split","pref_x0", "pref_y0", "sigma_x0", "sigma_y0", "theta","tuned_slope","tuned_VE"]    
        cross_v=False
    elif function_type == "mono":
        columns = ["node","layer","split","x0_exp", "y0_exp", "ratio_x0", "ratio_y0","mono_slope","mono_intercept","mono_VE"]
        if cross_v == True:
            columns.append("mono_cv_VE")
    
    outcomes = pd.DataFrame(columns = columns)
    
    num_splits, num_timings, num_layers, num_nodes = responses.shape
    response_labels = response_labels/1000
    response_labels = np.asarray(response_labels.T)
    
    index = 0
    for split in np.arange(num_splits):
        for layer_id in np.arange(num_layers):
            for node_id in np.arange(num_nodes):
                
                node_activity = np.asarray(responses[split,:,layer_id,node_id])
                
                if all(node_act == node_activity[0] for node_act in node_activity):
                    outcomes.loc[index] = [node_id, layer_id, split, *np.zeros(len(columns)-3) + np.nan]
                    index += 1
                    continue
                
                # because we want ftol to have the same tolerance for each of the nodes, we normalize
                # each node's activation
                node_activity = (node_activity - min(node_activity)) / (max(node_activity) - min(node_activity))
                
                attempt = 0
                solved = False
                node_fit = None
            
                
                while attempt <= 100:
                    
                    # first search with ftol (stepsize in cost function) quite large
                    # use outputted parameters as initial guess in smaller steps 
                    if solved == False:
                        ftol = 1e-4
                        initial_guess = (c.FIT_SETTINGS["initial"][function_type][1] - c.FIT_SETTINGS["initial"][function_type][0]) * np.random.random(c.FIT_SETTINGS["initial"][function_type][0].shape) + c.FIT_SETTINGS["initial"][function_type][0]
                        attempt += 1
                    else:
                        ftol = ftol/10
                        initial_guess = node_fit[0]
                
                    try:
                        
                        if function_type == "tuned":
                            function_to_fit = tuned_function
                        elif function_type == "mono":
                            function_to_fit = mono_function
                    
                        node_fit = curve_fit(
                            function_to_fit, 
                            response_labels, 
                            node_activity, 
                            p0 = initial_guess, 
                            bounds = c.FIT_SETTINGS["bounds"][function_type],
                            method = c.FIT_SETTINGS["method"],
                            ftol=ftol,
                            full_output=True)
                        
                        if round(ftol,5) == round(1e-4,5):
                            solved = True
                        
                        # smallest stepsize is 1e-10, then accept these parameters
                        if round(ftol,15) == round(1e-14,15):
                            # plot findings
                            if node_id % 200 == 0:
                                pred_activity = function_to_fit(response_labels, *node_fit[0])
                                compare_actual_pred = np.stack((node_activity[None,...,None], pred_activity[None,...,None]),axis=2)
                                plot_response_heatmaps(compare_actual_pred, response_labels.T, 0,x0 =x0,y0 =y0,normalize_per_plot=False)
                                
                            # we have to compute it again because scipy.optimize.curve_fit has a bug that prevents us from using full_output when passing a 'bounds' parameter)
                            variance_explained = compute_variance_explained(function_type, node_activity, response_labels, node_fit[0]) 
                            if cross_v == True:
                                node_activity_other_split = np.asarray(responses[abs(split - 1),:,layer_id,node_id])
                                node_activity_other_split = (node_activity_other_split - min(node_activity_other_split)) / (max(node_activity_other_split) - min(node_activity_other_split))
                                cv_variance_explained = compute_variance_explained(function_type, node_activity_other_split, response_labels, node_fit[0]) 
                                variance_explained = [variance_explained, cv_variance_explained]
                            else:
                                variance_explained = [variance_explained]
                                
                            outcomes.loc[index] = [node_id, layer_id, split, *node_fit[0], *variance_explained]
                            index += 1
                            attempt = 101
                            
                    except RuntimeError:
                        # if it doesn't work for smaller stepsize ==> keep values from previous larger stepsize
                        if solved == True:
                            print('oh no runtimeerror even though it worked under previous ftol')
                            print(ftol)
                            
                            # we have to compute it again because scipy.optimize.curve_fit has a bug that prevents us from using full_output when passing a 'bounds' parameter)
                            variance_explained = compute_variance_explained(function_type, node_activity, response_labels, initial_guess) 
                            if cross_v == True:
                                node_activity_other_split = np.asarray(responses[abs(split - 1),:,layer_id,node_id])
                                node_activity_other_split = (node_activity_other_split - min(node_activity_other_split)) / (max(node_activity_other_split) - min(node_activity_other_split))
                                cv_variance_explained = compute_variance_explained(function_type, node_activity_other_split, response_labels, initial_guess) 
                                variance_explained = [variance_explained, cv_variance_explained]
                            else:
                                variance_explained = [variance_explained]
                            
                            
                            outcomes.loc[index] = [node_id, layer_id, split, *initial_guess, *variance_explained]
                            index += 1
                            attempt = 101
                        
                        # if it could not be solved with large stepsize after 100 random initializations: give up
                        else:
                            print(f'\t\tnode {node_id} in layer {layer_id}: optimal parameters not found after max evaluations. {attempt}/100 attempts')
                            if  attempt == 100:
                                outcomes.loc[index] = [node_id, layer_id, split, *np.zeros(len(columns)-3) + np.nan]
                                index += 1

                    
    if save == True:
        outcomes.to_csv(outcomes_file, index=False)

    return outcomes


def run_fit(repetitions, assessed_per="event",x0="duration",y0="period",control_condition=None):
    """
    load in responses of all nodes in multiple replicas and start the fitting procedure

    repetitions: information about the network replicas
    assessed_per: how to sum activations: we always use "event" here
    x0: timing label to use for x dimension of the response function space
    y0: timing label to use for y dimension of the response function space
    control_condition: type of control condition, if any

    returns: fits of the monotonic and tuned response functions 
    """
        
    if  "cross_v" in repetitions and repetitions["cross_v"] == True:
        cross_v = True
    else:
        cross_v = False
      
    # get file if it exist
    results_dir = c.RESULTS_DIR
    net_path, _, event_response_path, movie_response_path, response_labels_path, mono_fits_path, tuned_fits_path, state_change_response_path = get_result_paths(results_dir, repetitions,control_condition=control_condition,x0=x0,y0=y0)

    if os.path.exists(mono_fits_path) and os.path.exists(tuned_fits_path):
        mono_fits = pd.read_csv(mono_fits_path)
        tuned_fits = pd.read_csv(tuned_fits_path)
        return mono_fits, tuned_fits
    
    # evaluations on the whole dataset
    # these are the activations I'll use for model fitting
    if not os.path.exists(event_response_path) or not os.path.exists(state_change_response_path) or not os.path.exists(movie_response_path) or not os.path.exists(response_labels_path):
        evaluate_net(results_dir, repetitions,do_plotting = False,control_condition=control_condition)
        
    timing_responses, response_labels = get_timing_responses(results_dir, repetitions,assessed_per=assessed_per,gate_idx=0,x0=x0,y0=y0,control_condition=control_condition) 
    if not os.path.exists(mono_fits_path):
        mono_fits = fit_responses(timing_responses, response_labels, "mono", mono_fits_path, cross_v = cross_v,x0=x0,y0=y0)
    else:
        mono_fits = pd.read_csv(mono_fits_path)

    if not os.path.exists(tuned_fits_path):
        tuned_fits = fit_responses(timing_responses, response_labels, "tuned", tuned_fits_path, cross_v = cross_v,x0=x0,y0=y0)
    else:
        tuned_fits = pd.read_csv(tuned_fits_path)

    
    return mono_fits, tuned_fits