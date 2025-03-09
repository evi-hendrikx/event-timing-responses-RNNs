import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import config_file as c
import os
import matplotlib as mpl

# Constants
COLOR_MAP = "Greys"
IMAGE_NORM = plt.Normalize(0.0, 1.0)


def show_prediction_grid_rows(target_list, prediction_per_layer, save_path,binary_output=False):
    """
    Shows images of a movie in a grid
    :param target_list: actual wanted predictions
    :param prediction_per_layer: like target_list, but mulitple stacked outputs of the models
    save_path: where you want the figure to be saved
    binary_output (bool): if yes, >0.5 is given as black and else white. If no: gray shades for values
    """
    
    text_space = 1.5
    amount_frames_per_row = 40
    h_size_fig = 18
    white_space = 0.1
    size_frames = (h_size_fig - text_space -
                   amount_frames_per_row * white_space)/amount_frames_per_row

    space_rows = 3
    target_rows = 1
    network_rows = len(prediction_per_layer)
    info_rows = target_rows  + network_rows
    amount_rows = info_rows * 2 + space_rows
    v_space_rows = size_frames / 2
    v_size_fig = amount_rows * size_frames + (amount_rows - 1) * v_space_rows

    fig = plt.figure(figsize=(h_size_fig, v_size_fig))

    fig.suptitle("Time -->", ha="left", va="center", fontsize=20)
    rows_conditions = gridspec.GridSpec(
        amount_rows, 2, wspace=0, hspace=v_space_rows, width_ratios=np.array([text_space, h_size_fig-text_space]))

    # create_inside_grid
    for row in range(amount_rows):
        text_col = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=rows_conditions[row, 0])
        ax_text = plt.Subplot(fig, text_col[0])
        ax_text.set_axis_off()

        frames = gridspec.GridSpecFromSubplotSpec(
            1, amount_frames_per_row, subplot_spec=rows_conditions[row, 1], wspace=white_space)

        if row == 0 or row == amount_rows - info_rows + 0:
            ax_text.text(0, 0.4, "Target:")
            ax_text.set_xticks([])
            ax_text.set_yticks([])
            fig.add_subplot(ax_text)

            for frame in range(amount_frames_per_row):
                ax = plt.Subplot(fig, frames[frame])
                if row == 0:
                    frame_to_show = frame
                else:
                    frame_to_show = frame + amount_frames_per_row
                ax.imshow(target_list[frame_to_show],
                          norm=IMAGE_NORM, cmap=COLOR_MAP)
                ax.title.set_text(f"{frame_to_show + 1}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)


        if (row > 0 and row <= network_rows) or row > amount_rows - info_rows:
            if row > 0 and row <= network_rows:
                prediction_id = row - 1
            else:
                prediction_id = row - 1 - (amount_rows - info_rows)

            prediction_key = list(prediction_per_layer.keys())[prediction_id]
            ax_text.text(0, 0.4, prediction_key)
            ax_text.set_xticks([])
            ax_text.set_yticks([])
            fig.add_subplot(ax_text)

            assert len(prediction_per_layer[prediction_key]) == len(target_list)

            for frame in range(amount_frames_per_row):
                ax = plt.Subplot(fig, frames[frame])
                if row > 0 and row <= network_rows:
                    frame_to_show = frame
                else:
                    frame_to_show = frame + amount_frames_per_row

                ax.imshow(prediction_per_layer[prediction_key]
                          [frame_to_show], norm=IMAGE_NORM, cmap=COLOR_MAP)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

    plt.savefig(save_path)
    plt.show()


def plot_train_loss(loss_array, close_plots=True, show_fig=True,color="black",ylabels = True):
    """
    Plots average train loss per epoch
    :param loss_array: 1d np array (e.g. shape (100,))
    :return:
    """
    if close_plots == True:
        plt.close()

    if loss_array.shape[0] == 2:
        la_0 = loss_array[0]
        la_1 = loss_array[1]
        x = np.arange(len(la_0))
        plt.fill_between(x, la_0, la_1, color=color, alpha=0.2)
    else:

        x = np.arange(len(loss_array))

        if show_fig == True:
            plt.plot(x, loss_array, color=color)
            plt.ylim(0, 1)
            plt.show()
        elif show_fig == False:
            plt.plot(x, loss_array, color=color, alpha=0.2)
            
            
    plt.ylim(0, 1)
   
    plt.yticks([])

    return



def plot_response_heatmaps(timing_res, timing_labels, neuron_idx, x0="duration", y0="period", normalize_per_plot=True,layer_id = None,split = 0, ax = []):
    """
    Plots response heatmaps for one neuron, all num_layers and all timings.
    :param timing_res: timing response of shape (num_layers, num_splits, num_timings, num_hidden)
    :param timing_labels: fit_labels for timing response of shape (num_timings, 2)
    :param neuron_idx: neuron index to plot
    :return: None
    """
    if layer_id !=None:
        num_layers = 1
    else:
        num_layers = timing_res.shape[2]
    
    if isinstance(ax,list):
        fig, axes = plt.subplots(num_layers, 1, figsize=(15, num_layers * 10))
    
        if num_layers == 1:
            axes = [axes]

    timing_res = timing_res[split, ...]
    
    share_color_bar = True

    if isinstance(ax,list):
        
        for idx, ax in enumerate(axes):
            
            if layer_id != None:
                idx = layer_id
                
            t_res = timing_res[:, idx, neuron_idx].reshape(-1, 1)
            data = np.concatenate((t_res, timing_labels), axis=1)
    
            frame = pd.DataFrame(data, columns=["value", x0, y0])
            unpivot = frame
            frame = pd.pivot_table(frame, index=y0, columns=x0, values="value")
            frame = frame.reindex(frame.index.sort_values(ascending=False))
    
            if normalize_per_plot:
                min = np.nanmin(frame.values)
                max = np.nanmax(frame.values)
    
                # normalize to values between 0 and 1
                frame = (frame - min) / (max - min) if max != min else frame
    
                sns.heatmap(frame, cmap="coolwarm", ax=ax, vmin=-0.1, vmax=1.1,annot=False)#, annot=np.around(
                    #frame.to_numpy(), 2))
    
            else:
                if share_color_bar == True:
                    vmin = 0#np.min(np.asarray(timing_res))
                    vmax = 1#np.max(np.asarray(timing_res))
                else:
                    vmin = np.min(np.asarray(t_res))
                    vmax = np.max(np.asarray(t_res))
                    
                sns.heatmap(frame, cmap="coolwarm", ax=ax,vmin = vmin, vmax = vmax,
                            annot=False)#np.around(frame.to_numpy(), 2))
                
    else:    
            
        if layer_id != None:
            idx = layer_id
            
        t_res = timing_res[:, idx, neuron_idx].reshape(-1, 1)
        data = np.concatenate((t_res, timing_labels), axis=1)

        frame = pd.DataFrame(data, columns=["value", x0, y0])
        unpivot = frame
        frame = pd.pivot_table(frame, index=y0, columns=x0, values="value")
        frame = frame.reindex(frame.index.sort_values(ascending=False))

        if normalize_per_plot:
            min = np.nanmin(frame.values)
            max = np.nanmax(frame.values)

            # normalize to values between 0 and 1
            frame = (frame - min) / (max - min) if max != min else frame

            sns.heatmap(frame, cmap="coolwarm", ax=ax, annot=np.around(
                frame.to_numpy(), 2), vmin=-0.1, vmax=1.1,cbar = False)

        else:

            if share_color_bar == True:
                vmin = 0#np.min(np.asarray(timing_res))
                vmax = 1#np.max(np.asarray(timing_res))
            else:
                vmin = np.min(np.asarray(t_res))
                vmax = np.max(np.asarray(t_res))
                
            sns.heatmap(frame, cmap="coolwarm", ax=ax,vmin = vmin, vmax = vmax,
                        annot=np.around(frame.to_numpy(), 2),cbar = False)

    # plt.show()
    return frame


def plot_inside_response_heatmaps(timing_res, timing_labels, node_id, layer_id, x0="duration", y0="period", normalize=True):
    """
    Plots response heatmaps for one neuron, all num_layers and all timings.
    :return: None
    """
    t_res = timing_res[:, layer_id, node_id].reshape(-1, 1)
    if sum(np.isnan(t_res)) == len(t_res):
        return False

    data = np.concatenate((t_res, timing_labels), axis=1)
    frame = pd.DataFrame(data, columns=["value", x0, y0])
    frame = pd.pivot_table(frame, index=y0, columns=x0, values="value")
    frame = frame.reindex(frame.index.sort_values(ascending=False))

    min = np.nanmin(frame.values)
    max = np.nanmax(frame.values)

    # normalize to values between 0 and 1
    if normalize == True:
        frame = (frame - min) / (max - min) if max != min else frame
        sns.heatmap(frame, cmap="coolwarm", vmin=0, vmax=1,
                    cbar=False, xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(frame, cmap="coolwarm", vmin=0, vmax=2,
                    cbar=False, xticklabels=False, yticklabels=False)


    plt.savefig(c.RESULTS_DIR + '/node_activation.png')

    return True


def plot_performance_heatmap(performance_measure, labels, do_plotting = True,stds = []):
    """
    Shows average performance of a net for all timings
    :param losses: losses of test dataset shape = (dataset_len,)
    :param labels: timing fit_labels of losses shape = (dataset_len, 2)
    :return:
    """

    data = np.concatenate((labels, performance_measure.reshape(-1, 1)), axis=1)
    frame = pd.DataFrame(data, columns=["duration", "period", "value"])
    frame = frame.astype({"duration": int, "period": int, "value": float})

    frame = pd.pivot_table(frame, index="period",
                           columns="duration", values="value")
    frame = frame.reindex(frame.index.sort_values(ascending=False))
    
    if len(stds) != 0:
        data_stds = np.concatenate((labels, stds.reshape(-1, 1)), axis=1)
        frame_stds = pd.DataFrame(data_stds, columns=["duration", "period", "value"])
        frame_stds = frame_stds.astype({"duration": int, "period": int, "value": float})
        frame_stds = pd.pivot_table(frame_stds, index="period",
                               columns="duration", values="value")
        frame_stds = frame_stds.reindex(frame_stds.index.sort_values(ascending=False))
        

        
    annotations = np.asarray(["{:.2f}".format(sd) for mean,sd in zip(frame.to_numpy().flatten(),frame_stds.to_numpy().flatten())]).reshape(frame.shape)
        
        
    if do_plotting == True:

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sns.heatmap(frame, cmap="coolwarm", annot=annotations, vmin=0.0, vmax=1, xticklabels=True,
            yticklabels=True,fmt='')


        # .set_xlim(50, 1000)
        # ax.set_ylim(50, 1000)
        plt.show()
    return frame

