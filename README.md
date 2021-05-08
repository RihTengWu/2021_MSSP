# 2021_MSSP
This repository shares the established deep learning models presented in the paper: 

[1] Rih-Teng Wu et al. (2021) "A Physics-constrained Deep Learning Based Approach for Acoustic Inverse Scattering Problems," *Mechanical Systems and Signal Processing*

Given a downstream user-defined pressure fields, the objective is to find the corresponding geometry of the scatterers that lead to the downstream pressure responses. In this study, three scenarios of acoustic scattering experiments are designed:

**Scenario 1**: Single Frequency - Single Scatterer, where the number of scatterers is one, and the incident wave is generated with one wave frequency.

**Scenario 2**: Single Frequency - Cluster of Scatterers, where the number of scatterers is four, and the incident wave is generated with one wave frequency.

**Scenario 3**: Multiple Frequencies - Cluster of Scatterers, where the number of scatterers is four, and the incident wave is allowed to be generated with four wave frequencies, i.e., 200(Hz), 796(Hz), 5000(Hz), 12500(Hz).

These models are implemented in Python 2 using PyTorch version 0.2 with CUDA 8.0, cuDNN 6.0.21, and Ubuntu 16.04. In this repository, the shared three trained models are referring to:

**Scenario 1**: model_3_150_5

**Scenario 2**: model_3_150_46

**Scenario 3**: model_3_300_62

All the three models are implemented with a geometry estimator and a deep-autoencoder. The geometry estimator and the decoder shares the features learned from the encoder. For the model in Scenario 1, given an input real and imaginary part of the pressure fields (with size 270x210x2), the geometry estimator will output a weight factor that controls the shape of the scatterer, and the decoder will output the reconstructed real and imaginary part of the pressure fields. For the model in Scenario 2, given an input real and imaginary part of the pressure fields (with size 270x210x2), the geometry estimator will output a scatterer configuration (defined in [1]) that controls the shape of the four scatterers, and the decoder will output the reconstructed real and imaginary part of the pressure fields. For the model in Scenario 3, given an input real and imaginary part of the pressure fields (with size 270x210x8), the geometry estimator will output a scatterer configuration (defined in [1]) that controls the shape of the four scatterers, and the decoder will output the reconstructed real and imaginary part of the pressure fields.

A step by step description about how to use the models are provided as follows:

**Step 1**: Scale the real part of the pressure fields to [0,1] within each frequency.

**Step 2**: Scale the imaginary part of the pressure fields to [0,1] within each frequency.

**Step 3**: Stack the real and imaginary parts of the pressure fields.

**Step 4**: Feed the inputs into the network. The geometry estimator will estimate the geometry of the scatterers.

**Step 5**: The decoder will reconstruct the inputs.





