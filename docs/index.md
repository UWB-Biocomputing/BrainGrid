[![DOI](https://zenodo.org/badge/6034062.svg)](https://zenodo.org/badge/latestdoi/6034062)
## The BrainGrid Project

The original idea behind the BrainGrid Project is to develop a toolkit/software architecture to ease creating high-performance neural network simulators. It is now composed of two parts BrainGrid (the simulator) and Workbench (a simulation and data/software provenance tool). So, we now say that BrainGrid+Workbench focuses on enabling **high-performance/high-quality** neural simulation. Its key features:
- Its target audience are researchers who need higher performance than that made available by general-purpose simulators, those that use Python, etc. and who are therefore considering writing their own simulation code in C/C++.
- Its target application is large scale and long duration neural simulations.
- It provides a C++ programming interface that focuses new coding just on the core implementation of neuron, synapse, etc. state updates.
- It assumes that an investigator will first want to implement small and/or short duration simulations that run as a single thread on a CPU, to provide validation data to increase confidence in result validity (this is part of the "high quality" theme).
- It provides a programming interface to minimize code changes needed to port the simulation to single or multiple GPUs. In effect, we've already optimized the simulator for GPUs (this addresses the "high performance" aspect).
- We provide a (Java based) Workbench that allows the investigator to download, build, and run the simulator; the Workbench captures all software and data provenance information, so that later the investigator can compare different results generated at different times and see how they differ in terms of not only their input parameters but also their simulator versions. This helps one to determine whether two simulations are comparable, whether one or more are invalid due to simulator bugs, etc. (This also connects to the idea of "high quality".)

## Table of Contents

1. [Introduction](http://uwb-biocomputing.github.io/BrainGrid/1_introduction)

   1.1 What is BrainGrid?
   
   1.2 What is BrainGrid for?
   
   1.3 Why do we need BrainGrid?

2. [Installation](http://uwb-biocomputing.github.io/BrainGrid/2_installation)

   2.1 Necessary Hardware/Software
   
   2.2 Download BrainGrid
   
   2.3 Install BrainGrid

3. [Quickstart](http://uwb-biocomputing.github.io/BrainGrid/3_quickstart)

   3.1 Quick Sanity Test
   
   3.2 Use of Screen

4. [Configuration](http://uwb-biocomputing.github.io/BrainGrid/4_configuration)

   4.1 Use built-in models
   
   4.2 Configuring the model

5. Examples [under construction]

6. [Lab Publication](http://uwb-biocomputing.github.io/BrainGrid/6_lab-publication) 

7. [Acknowledgements](http://uwb-biocomputing.github.io/BrainGrid/7_acknowledgements)

---------
### Multiple simulation architectures:

- Single-threaded on general-purpose CPU
- GPU-accelerated using NVIDIA's CUDA libraries
- Multi-cluster (using multiple CPU cores and/or multiple GPUs) — *under development*
- GPU-accelerated using OpenCL — *planned*
- Multi-threading using OpenMP — *planned*

### Supported operating systems:

- GNU Linux

### BrainGrid Resources:

- [BrainGrid Forum]([https://groups.google.com/forum/#!forum/uwb-braingrid](https://groups.google.com/forum/#!forum/uwb-braingrid)): A place where BrainGridders can communicate and collaborate. Click the button "Apply to join this group" to be a BrainGridder.
- [Git Crash Course](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Git-Crash-Course)
- [Linux Crash Course](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Linux-Crash-Course)

### Latest News about BrainGrid:

Our recently published conference paper for IJCNN 2017:

> Michael Stiber, Fumitaka Kawasaki, Delmar Davis, Hazeline Asuncion, Jewel Lee, and Destiny Boyer. *BrainGrid+Workbench: High-Performance/High-Quality Neural Simulation*. International Joint Conference on Neural Networks (IJCNN), 14 – 19 May 2017, Anchorage, AK.

### Support or Contact:

BrainGrid documentation is under construction. If you cannot find what you are looking for or have trouble using BrainGrid, please contact us. 
