# MSG-SEVIRI-CNN-wet-dry

Supplementary material to the master thesis "Improved Rainfall Measurements from Microwave Link Attenuation Data via MSG SEVIRI Cloud Information", published in DGPF Jahrestagung 2024 (...).

MSG-SEVIRI-CNN-wet-dry is a CNN model to process the satellite imagery of MSG SEVIRI to a wet-dry classification. 
The input consists of a set of channels, two infrared (IR 3.9 & IR 10.8), one water vapor (WV 6.2) and one visible channel (VIS 0.6).
The CNN model seperately convolves and downsamples the channels to learn information of each channel individually.
Further information on the model can be found in the figure below and Wiegels et al. (2024).

![image](https://github.com/RebWiegels/MSG-SEVIRI-CNN-wet-dry/assets/62548605/2e04b413-6506-4ea0-ac96-0b530da4dafa)

# Usage

### Requirements

### Data

A test sub set of samples to run the model with prepared patches is provided in ```/data```.
- patches.nc to run the MSG-SEVIRI-CNN-wet-dry model
- input.nc and target.nc are the given test sets to evaluate the trained model

### Model

The model trained for the master thesis results is given in ```/model/sev_cnn```

### Code for Preprocessing 

The notebooks for preprocessing the data are given in the notebooks 01_... - 04_... . 
- Preparing SEVIRI data for patches and testing of the model in 01_rw_seviri_preparation.ipynb
- Preparing DWDs radar data RADKLIM YW to same temporal and spatial resolution of SEVIRI data in 02_rw_target_preparation.ipynb
- Preparing the baseline data exemplary in 03_rw_pcph_preparation.ipynb
- Preparing the patches of input and target dataset required by the model in 04_rw_patch_preparation.ipynb

### Code for Training and Evaluation

The notebook required to train the model is 05_rw_CNN_training.ipynb in which the model, normalization and furhter processing steps required for this model can be found.
The notebook 06_rw_CNN_evaluation.ipynb shows how the model can be evaluated by using the test sub set and the pre-trained model.

__Note: all helper modules are in ```/src```__

# Citation
