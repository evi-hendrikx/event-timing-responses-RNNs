
# event-timing-responses-RNNs

Currently submitted: _Transitions in event-timing responses in recurrent neural network models mirror those in the human brain_ 
(Evi Hendrikx, Daniel Manns, Nathan van der Stoep, Alberto Testolin, Marco Zorzi, Ben M. Harvey)
This pipeline was started by Daniel Manns and further adapted and developed by Evi Hendrikx

Previous preprint: https://doi.org/10.1101/2024.08.29.610320

Independent recurrent neural networks (indRNN, Li et al., 2018) are trained on a generative timing task: predict the next frame in a movie with a repeating event. This frame only exists of a single pixel. In an event pixels can be on (1) and off (0). An example movie could look like: 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0

## Running the pipeline
In order to run everything: run "main.py". 
Movies with regular temporal intervals are created using "generate_dataset.py". 
Networks are built using "network_models_new.py" and trained and evaluated using "pipeline.py".
The used datasets, and (trained) network architectures can also be found in the corresponding OSF database: https://osf.io/q3t75/ (10.17605/OSF.IO/Q3T75) 

## Evaluating accuracy
Accuracy is compared between different network depths (1-5 layers) in "accuracy_stats.py"

## Evaluating response functions in the network nodes
Monotonic and tuned functions are fit on the per-event activations of the network nodes in "model_fitting.py".
Fits are compared between different network depths and different model layers within the same network in "model_fitting_stats.py"

## Evaluating response function properties
Properties of the response functions are started in "parameter_stats.py" 

## Evaluating importance of tuned nodes
Comparing the performance of networks with disabled tuned nodes in the last layer versus networks with other nodes disabled in the last layer happens in "take_out_tuned.py"

## Requirements to run
provided are the required packages for Linux ("environment.yml")

## Data from previous publications
Most scripts are linked to each other and are called starting in main. Exceptions are the scripts that load in the data from other studies: importHarvey2020 and importHendrikx2022

## License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Contact
For questions or comments reach out to Evi Hendrikx: e.h.h.hendrikx [at] gmail [dot] com

To cite this code base:
Hendrikx, E., Manns, D., van der Stoep, N., Testolin, A., Zorzi, M., Harvey, B.M. (2025) event-timing-responses-RNNs. https://doi.org/10.5281/zenodo.14995621
