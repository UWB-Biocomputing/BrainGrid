# 2. Installation

## 2.1 Necessary Hardware/Software

BrainGrid is designed to be easy to use and fast to simulate with, but given its scope and flexibility, there are some tradeoffs. 

First, and perhaps most importantly, for the speedups that we desire, we found that **CUDA** was the most reasonable way to go. Hence,  if you want to use BrainGrid for migrating your model to GPUs, you will need the following: 

- **Linux**: Currently, BrainGrid only works on Linux. Any distro that supports **GNU-Make** and your chosen NVIDIA graphics card (if going the GPU route) should work. Make sure you have these packages:
- **NVIDIA GPU**: If you want your simulator to run on GPUs, you must use an NVIDIA GPU that is CUDA capable. Check NVIDIA's website for an up-to-date [list](https://developer.nvidia.com/cuda-gpus) of CUDA-compliant devices. 
- [**CUDA**](https://developer.nvidia.com/cuda-downloads): if you intend to use the GPU functionality for high performance. BrainGrid has been tested running on CUDA Version 8.0.44. 
- [HDF5](https://support.hdfgroup.org/HDF5/): HDF5 is a data model, library, and file format for storing and managing data. For example, Matlab has built-in functions that can easily manage, view, and analyze data in HDF5 format. To install HDF5, simply follow the website instructions. If you don't wish to use HDF5, you can use the XML format which is also supported.  

To become a BrainGrid user or collaborator, you might also need:

- **[Git](http://git-scm.com/)** & **[GitHub](https://github.com/)**: If you wish to use or contribute to the most up-to-date BrainGrid that is currently under development, you will need to get it from GitHub and keep it in sync. 
- **Matlab** or **Octave**: If you want to view the output results using our scripts

Of course, BrainGrid is totally open source. If you wanted, you could modify BrainGrid and make an OpenCL version. 

## 2.2 Download BrainGrid

In order to get started with BrainGrid, you will need to build it from scratch, which means getting its source codes. You can either download BrainGrid source codes as a zip file of a stable release (See [2.2.1 Download a release](#221-download-a-release)) or fork the development version from BrainGrid GitHub repository (See [2.2.2 Fork and clone BrainGrid](#222-fork-and-clone-braingrid)).  

### 2.2.1 Download a release

Point your browser to *https://github.com/UWB-Biocomputing/BrainGrid/releases* and download a release from the list by clicking the relevant source code button. 

- [Source code (zip)](https://github.com/UWB-Biocomputing/BrainGrid/archive/v0.9-alpha.zip)
- [Source code (tar.gz)](https://github.com/UWB-Biocomputing/BrainGrid/archive/v0.9-alpha.tar.gz)

After downloading the source code, unpack it in a convenient location and continue to install BrainGrid as described in the [Installing BrainGrid](#2.2-installing-braingrid) section below. 

### 2.2.2 Fork and clone BrainGrid

If you are a Github user, you can simply fork and clone BrainGrid. If you are new to Github, follow our Wiki page on [Contribute to BrainGrid open source project](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Contribute-to-BrainGrid-open-source-project). You can also go over our [Git Crash Course](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Git-Crash-Course) for some useful tips.

## 2.3 Install BrainGrid

In order to compile and run BrainGrid, you will need to set up a couple things in the **Makefile** first. 

1. Change to BrainGrid directory in your terminal

   ```shell
   $ cd BrainGrid
   ```

2. Open **Makefile** and change the following parameters:

	If you are using **CUDA**, 
   	
	- change the CUDA library directory: ```CUDALIBDIR = YOUR_CUDA_LIBRARY_PATH``` 
   	- you might also need to add your CUDA home directory into the ```PATH``` environment variable 

	If you are using **HDF5**, 
   	
	-  change HDF5 home directory: ```H5INCDIR = YOUR_HDF5_HOME_PATH``` 
   	-  change HDF5 library directory: ```H5LIBDIR = YOUR_HDF5_LIBRARY_PATH```
   	-  make ```CUSEHDF5 = yes``` in line 17 to use HDF5 file format instead of XML


3. BrainGrid is written in C++ and CUDA C/C++. Make sure you have all these dependencies in order to compile BrainGrid:
   - [make](https://www.gnu.org/software/make/)
   - [g++](https://gcc.gnu.org/)
   - [h5c++](https://support.hdfgroup.org/HDF5/Tutor/compile.html): compile script for HDF5 C++ programs
   - [nvcc](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4ftSRZe00): if you are using GPU for high performance, nvcc is the compiler by Nvidia for use with CUDA

---------
[>> Next: 3. Quickstart](http://uwb-biocomputing.github.io/BrainGrid/3_quickstart)

---------
[<< Go back to BrainGrid Home page](http://uwb-biocomputing.github.io/BrainGrid/)
