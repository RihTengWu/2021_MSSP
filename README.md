# 2021_MSSP
This repository shares the established deep learning models presented in the paper: 

Rih-Teng Wu et al. (2021) "A Physics-constrained Deep Learning Based Approach for Acoustic Inverse Scattering Problems," Mechanical Systems and Signal Processing

In this study, three scenarios of acoustic scattering experiments are designed:

Scenario 1: Single Frequency - Single Scatterer, where the number of scatterers is one, and the incident wave is generated with one wave frequency.

Scenario 2: Single Frequency - Cluster of Scatterers, where the number of scatterers is four, and the incident wave is generated with one wave frequency.

Scenario 3: Multiple Frequencies - Cluster of Scatterers, where the number of scatterers is four, and the incident wave is allowed to be generated with four wave frequencies, i.e., 200(Hz), 796(Hz), 5000(Hz), 12500(Hz).

These models are implemented in Python 2 using PyTorch version 0.2 with CUDA 8.0, cuDNN 6.0.21, and Ubuntu 16.04.
