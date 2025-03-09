import torch
import numpy as np
from network_models_new import RNNetwork, NNetwork
import os

# paths (will be initialized in main)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(DIR_PATH, "datasets")
RESULTS_DIR = os.path.join(DIR_PATH, "results")
STATS_DIR = os.path.join(DIR_PATH, "stats_and_figs")

# Creating movies
IMAGE_WIDTH = 1
IMAGE_HEIGHT = 1
CHANGE_LOCATION = False
MOVIE_LENGTH = 80 #  80 * 50 ms = 4000 ms
DOT_SIZE = 0.5 # radius
FPS = 20
RANDOM_PHASE = True
CONTROL_MOVIES = False
SCRAMBLE_SHIFTED = True # True means that NOT all movies of the same timing are shifted similarly (first scramble, then shift over phases), RATHER these are scambled after shift
DATASET_IDX = None
FORCE_LENGTH = True
# None means that there is a fixed movie length for each condition, 10 means that there will be 10 events in each movie
EVENTS_PER_SET = None

# Trainig model parameters
RESPONSE_MASK = False
BINARY_OUTPUT = True
NET_CLASS = RNNetwork # Change if you run without recurrency
NET_TYPE = "RNN" # Change if you run without recurrency
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
SHUFFLE = True # To have the data reshuffled at every epoch
DROPOUT_PROB = 0
NONLINEARITY = "relu"
BIAS = True
BATCH_FIRST = True
RWEIGHT_CONSTRAINT = 2 ** (1/MOVIE_LENGTH)
RWEIGHT_CONSTRAINT_MIN = -(2 ** (1/MOVIE_LENGTH))

NUM_EPOCHS = 10000
EARLY_STOPPING = False
TRAIN_SPLIT_RATIO = 0.8
USE_CACHED_NETWORKS = True
CACHE_DATASETS = True

# torch constants
if BINARY_OUTPUT== True:
    LOSS_FN = torch.nn.BCELoss(reduction="none") 
else:
    LOSS_FN = torch.nn.MSELoss(reduction="none")
MAX_LOSS = None
SEED = 42

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
USE_AMP = True if DEVICE == "cuda" else False
NUM_WORKERS = 4

# fitting monotonic & tuned functions
NEGATIVE = False
if NEGATIVE == True:
    FIT_SETTINGS = {'random': True, 
                 'initial': {"tuned":(np.array([-0.05, -0.05, 0.001, 0.001, 0, 0, 0.1,-2]), 
                             np.array([1.1, 1.1,1.5, 3, 180, 1, 10,2])), 
                             "mono":(np.array([0, 0, 0,0,0.1,-2]), 
                                         np.array([1, 1, 10,10,10,2]))},
                'bounds': {"tuned":(np.array([-0.05, -0.05, 0.001, 0.001, 0, 0,-np.inf,-np.inf]),
                            np.array([1.1, 1.1, 10, 10, 180, 1, np.inf,np.inf])),
                            "mono":(np.array([0,0, -np.inf, -np.inf,1e-08,-np.inf]),
                            np.array([1, 1, np.inf, np.inf,np.inf,np.inf]))
                             },
                 'loss': 'linear',
                 'method': 'trf'
                }
else:
    FIT_SETTINGS = {'random': True, 
                 'initial': {"tuned":(np.array([-0.05, -0.05, 0.001, 0.001, 0, 0, 0.1,-2]), 
                             np.array([1.1, 1.1,1.5, 3, 180, 1, 10,2])), 
                             "mono":(np.array([0, 0, 0,0,0.1,-2]), 
                                         np.array([1, 1, 10,10,10,2])),
                             "tuned_no_exp":(np.array([-0.05, -0.05, 0.001, 0.001, 0, 0.1]), 
                                         np.array([1.1, 1.1,1.5, 3, 180, 10]))},
                   'bounds': {"tuned":(np.array([-0.05, -0.05, 0.001, 0.001, 0, 0,1e-08,-np.inf]),
                              np.array([1.1, 1.1, 10, 10, 180, 1, np.inf,np.inf])),
                              "mono":(np.array([0,0, 1e-08, 1e-08,1e-08,-np.inf]),
                              np.array([1, 1, np.inf, np.inf,np.inf,np.inf])),
                              "tuned_no_exp":(np.array([-0.05, -0.05, 0.001, 0.001, 0,1e-08]),
                                         np.array([1.1, 1.1, 10, 10, 180, np.inf]))
                             },
                 'loss': 'linear',
                 'method': 'trf'
                }
