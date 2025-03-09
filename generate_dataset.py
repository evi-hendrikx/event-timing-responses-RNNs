import numpy as np
from torch.utils.data import Dataset
from utility import transform_movie, to_tensor, get_dataset_paths
import os
import pandas as pd
import config_file as c

class EventTimingDataset(Dataset):

    def __init__(self, base_dir, movie_length, events_per_set=None, random_phase=False,
                 transform=True, cache_datasets=True, force_length=True, scrambled_control=False):

        # size of pickle buffer 
        self.chunk_size = 500

        # from 50 to 1000 in steps of 50 including 50 and 1000
        self.time_steps = 20

        # Frames per second
        self.FPS = c.FPS

        # Movie properties
        self.dot_size = c.DOT_SIZE
        self.canvas_hor = c.IMAGE_WIDTH
        self.canvas_ver = c.IMAGE_HEIGHT

        # this corresponds to 50 / movie_length = 60 frames in each movie if mvoie length = 60
        self.fixed_set_duration = movie_length * 50

        # approximate movie length - this will vary based on event length
        self.movie_length = movie_length

        # if true the movie length will always be fixed as specified
        self.force_length = force_length

        # if true all possible onsets of movie will be generated
        self.random_phase = random_phase

        # specifies the number of events per movie - this will overwrite movie length
        self.events_per_set = events_per_set

        # if true load dataset from saved location (if existent)
        self.cache_datasets = cache_datasets

        # transforms movies to torch.tensor
        self.transform = transform

        self.possible_durations = np.arange(50, 1050, 50)
        self.possible_periods = np.arange(50, 1050, 50)
        
        self.dot_position = [0,0]

        self.dataset_dir, self.data_dir, self.annotations_path = get_dataset_paths(base_dir, self.random_phase, self.events_per_set, scrambled_control)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # there must be at least some movies in movies_dir
        if self.cache_datasets and len(os.listdir(self.data_dir)) > 0:
            self.movie_labels = pd.read_csv(self.annotations_path)

        else:
            # generate dataset
            print("Generating dataset...")
            self.generate_dataset(scrambled_control)
            print("...Finished generating")
            self.movie_labels = pd.read_csv(self.annotations_path)

    def generate_dataset(self,scrambled_control):
        # events is a list of dicts. The entry "movie" is the final generated images. The rest of the entries are
        # helpers to generate "movies".
        movie_path_list = []
        target_path_list = []
        event_path_list = []
        durations_list = []
        periods_list = []
        events = []
        event_durations = []
        event_periods = []
        
        # set seed so all different computers running this use the same shuffled dataset
        if scrambled_control == True:
            random_shuffle_seed = np.random.RandomState(c.SEED)

        # loop over all possible combinations of duration and period
        for duration in self.possible_durations:
            for period in self.possible_periods:
                if c.CHANGE_LOCATION == False and duration < period:
                    event_durations.append(duration)
                    event_periods.append(period)
                elif c.CHANGE_LOCATION == True and duration <= period: 
                    event_durations.append(duration)
                    event_periods.append(period)

        # cast to np array
        event_durations = np.array(event_durations)
        event_periods = np.array(event_periods)

        # Inter-stimulus interval, the time from one event ending to the next event starting
        event_ISI = event_periods - event_durations
        
        if np.count_nonzero(event_ISI < 0) > 0:
            print('Event dataset_durations cannot be longer than dataset_periods, check timing of event %d'.format(event_ISI[event_ISI > 0]))

        # For a square image, make the dimensions equal. This is a good choice because we want the image to fall into a
        # central circle, but a rectangular image may be a better input to a network.

        # The maximum distance between the center of the image and the centre of the dot. This fills the whole canvas's short
        # axis, any larger and the dot can fall off the screen.

        # Number of dot diameters that must fall between consecutive dots. Under 1 means consecutive dots can overlap.
        min_dot_to_dot = 1
        self.min_dot_to_dot = min_dot_to_dot * self.dot_size * 2

        # The duration of one frame, in milliseconds
        stim_frame = 1000 / self.FPS

        # If this is zero, a fixed number of events will occur. Otherwise, this gives the duration of the set (of repeating
        # events) in milliseconds
        if self.events_per_set is not None:
            # Repeated for every event timing
            events_per_set = np.tile(self.events_per_set, len(event_durations))
        else:
            # A whole number of events does not always fit perfectly into the set duration, so we take the nearest whole
            # number. You can also use 'ceil' or floor' to always round up or down
            events_per_set = (self.fixed_set_duration // event_periods).astype(int)
            #print(f"events_per_set: {events_per_set}")

        # Generate lists to give the state of each frame
        for eventCounter in range(len(event_durations)):
            event_on = np.ones([int(round(event_durations[eventCounter] / stim_frame))], dtype="int")
            event_off = np.zeros([int(round(event_ISI[eventCounter] / stim_frame))], dtype="int")
            single_event = np.append(event_on, event_off)
            # Now generate a list of whether each frame has the event on or off, for this event timing
            tmp = {"seq": np.tile(single_event, [events_per_set[eventCounter]])}

            # And also a list of whether each frame start a new event, in case one event stops and the next starts on the
            # next frame. This also states the duration of that event.
            new_set = np.append(event_durations[eventCounter],
                                np.zeros([round(event_periods[eventCounter] / stim_frame - 1)], dtype="int"))

            tmp["new_set"] = np.tile(new_set, [events_per_set[eventCounter]])

            events.append(tmp)

        # create movie
        for eventCounter in range(len(events)):
            # initialize empty movie, using integer data type (or even logical) to keep it small
            events[eventCounter]["movie"] = np.zeros((len(events[eventCounter]["seq"]), self.canvas_hor, self.canvas_ver), dtype=np.int8)
            # Initialise empty dot position at the start of each movie
            
            self.dot_position = [0,0]
            
            for frameCounter in range(len(events[eventCounter]["seq"])):
                # If this frame has an event
                if events[eventCounter]["seq"][frameCounter] > 0:
                    # If we are starting a new event
                    if events[eventCounter]["new_set"][frameCounter] > 0:
                        events[eventCounter]["movie"][frameCounter, :, :] = self.generate_image()
                    elif frameCounter >= 1:
                        # Otherwise just copy the previous frame
                        events[eventCounter]["movie"][frameCounter, :, :] = events[eventCounter]["movie"][frameCounter - 1, :, :]

            total_old_dur = sum(events[eventCounter]["movie"])
            if scrambled_control == True:
                if c.SCRAMBLE_SHIFTED == False:
                    random_shuffle_seed.shuffle(events[eventCounter]["movie"])
                
            if sum(events[eventCounter]["movie"]) != total_old_dur:
                ValueError('shuffled movies are not created properly')
                
            event_arr = np.array(events[eventCounter]["new_set"] > 0, dtype=bool)
            movie = events[eventCounter]["movie"]

            # save at least not rolled movie and events once
            ext = np.concatenate((movie, movie), axis=0)
            default_movie = ext[:self.movie_length]
            default_target = ext[1:self.movie_length + 1]
            default_event = np.concatenate((event_arr, event_arr), axis=0)[:self.movie_length]

            d_m_path = os.path.join(self.data_dir,
                                  f"movie_{event_durations[eventCounter]}_{event_periods[eventCounter]}.npy")
            d_t_path = os.path.join(self.data_dir,
                                    f"target_{event_durations[eventCounter]}_{event_periods[eventCounter]}.npy")
            d_e_path = os.path.join(self.data_dir,
                                  f"event_{event_durations[eventCounter]}_{event_periods[eventCounter]}.npy")
            np.save(d_m_path, default_movie)
            np.save(d_t_path, default_target)
            np.save(d_e_path, default_event)

            # Next part is going to shift events over its own phase
            # (shorter events have more --> this is to compensate for that)
            # first event_arr entry is always true, so we have to look at the second split
            split_events = np.split(event_arr, np.where(event_arr == True)[0])[1]
            # phase_len for constant luminance is 1 
            # Currently doesn't happen: we excluded constant luminance because it's a movie full
            # of black pixels: doesn't have any event timing info available anymore
            phase_len = len(split_events) if c.CHANGE_LOCATION == False else len(split_events) *2

            for phase in range(phase_len):
                if self.random_phase:
                    m = np.roll(movie, phase + 1, axis=0)
                    t = np.roll(m, -1, axis=0) 
                    e = np.roll(event_arr, phase + 1, axis=0)
                    if self.force_length:
                        ext = np.concatenate((m, m), axis=0)
                        if scrambled_control == True and c.SCRAMBLE_SHIFTED == True:
                            shuffled_indices = random_shuffle_seed.permutation(self.movie_length)
                            m = ext[shuffled_indices]
                            t = ext[np.append(shuffled_indices[1:], np.random.randint(self.movie_length, len(ext)))]
                        else:
                            m = ext[:self.movie_length]
                            t = ext[1: self.movie_length + 1]
                        
                        e = np.concatenate((e, e), axis=0)[:self.movie_length]
                            
                    m_path = os.path.join(self.data_dir, f"movie_{event_durations[eventCounter]}_{event_periods[eventCounter]}_{phase}.npy")
                    t_path = os.path.join(self.data_dir,
                                          f"target_{event_durations[eventCounter]}_{event_periods[eventCounter]}_{phase}.npy")
                    e_path = os.path.join(self.data_dir, f"event_{event_durations[eventCounter]}_{event_periods[eventCounter]}_{phase}.npy")
                    

                    
                    np.save(m_path, m)
                    np.save(t_path, t)
                    np.save(e_path, e)

                    movie_path_list.append(m_path)
                    target_path_list.append(t_path)
                    event_path_list.append(e_path)
                    durations_list.append(event_durations[eventCounter])
                    periods_list.append(event_periods[eventCounter])

                else:
                    # append same movie several times to match event number
                    movie_path_list.append(d_m_path)
                    target_path_list.append(d_t_path)
                    event_path_list.append(d_e_path)
                    durations_list.append(event_durations[eventCounter])
                    periods_list.append(event_periods[eventCounter])

        annotations_frame = pd.DataFrame({"movie_path": movie_path_list,
                                          "target_path": target_path_list,
                                          "event_path": event_path_list,
                                          "duration": durations_list,
                                          "period": periods_list})
        annotations_frame.to_csv(self.annotations_path, index=False)
        return

    def __len__(self):
        return len(self.movie_labels)

    def __getitem__(self, idx):
        movie_path = os.path.join(self.data_dir, self.movie_labels.iloc[idx, 0])
        target_path = os.path.join(self.data_dir, self.movie_labels.iloc[idx, 1])
        event_path = os.path.join(self.data_dir, self.movie_labels.iloc[idx, 2])
        movie = np.load(movie_path)
        target = np.load(target_path)
        events = np.load(event_path)
        label = self.movie_labels.iloc[idx, 3:5].to_numpy()

        if self.transform:
            movie = transform_movie(movie)
            target = transform_movie(target)
            events = to_tensor(events)
            label = to_tensor(label)

        return movie, target, label, events
    
    
    def generate_image(self):
        """
        Generates an image with dots in the centre
        :return: generated image
        """
        
        # dot indicates whether there will be a dot in the picture
        # Get the x and y position for each pixel.
        xpos, ypos = np.meshgrid(np.arange(self.canvas_hor), np.arange(self.canvas_ver))
      
        if c.CHANGE_LOCATION:
            # TODO: add random position generator if we do this with more than 1 pixel
            random_x = np.random.randint(0,self.canvas_hor)
            random_y = np.random.randint(0,self.canvas_ver)
            new_position = [random_x, random_y]
            while np.sqrt(((new_position[0]-self.dot_position[0]) **2) + ((new_position[1]-self.dot_position[1]) **2)) < self.min_dot_to_dot :
                random_x = np.random.randint(0,self.canvas_hor)
                random_y = np.random.randint(0,self.canvas_ver)
                new_position = [random_x, random_y]
            self.dot_position = new_position
            
        pixel_distances = np.sqrt((xpos - self.dot_position[0]) ** 2 + (ypos - self.dot_position[1]) ** 2)
        pixel_distances = pixel_distances.T


        # Set pixels within the dot to value 1, others in this frame to zero
        image = np.array(pixel_distances <= self.dot_size).astype(np.int8)

        return image
