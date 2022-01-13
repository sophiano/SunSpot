# nn_models


This folder contains the trained neural networks (files with **nn**) and recurrent neural networks (files with **rnn**) for monitoring the composites 'Nc'. 

The prediction performances of the models are also saved (in PNG images). 

Files with **365** (resp. **27**) in their extension indicate that the networks are trained to detect deviations in the data smoothed on 365 days (resp. 27 days). 

Those models are saved for reproducibility purpose.

## Instructions

These models can be loaded with the function ``model_from_json()`` from the package [keras](https://keras.io/) : 

````
json_file = open('nn_models/nn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("nn_models/nn_regression_27.h5")
     
````



