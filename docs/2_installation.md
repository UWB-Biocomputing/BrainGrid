# 2. Installation

## 2.1 Necessary Hardware/Software

BrainGrid is designed to be easy to use and fast to simulate with, but given its scope and flexibility, there are some tradeoffs. 

First, and perhaps most importantly, for the speedups that we desire, we found that **CUDA** was the most reasonable way to go. Hence,  if you want to use BrainGrid for migrating your model to GPUs, you will need the following: 

- **NVIDIA GPU**: If you want your simulator to run on GPUs, you must use an NVIDIA GPU that is CUDA capable. Check NVIDIA's website for an up-to-date [list](https://developer.nvidia.com/cuda-gpus) of CUDA-compliant devices. 

Of course, BrainGrid is totally open source. If you wanted, you could modify BrainGrid and make an OpenCL version. 

In order to run BrainGrid, you will also need the following software:

- **Linux**: Currently, BrainGrid only works on Linux. Any distro that supports **GNU-Make** and your chosen NVIDIA graphics card (if going the GPU route) should work. Make sure you have these packages:
- [**CUDA**](https://developer.nvidia.com/cuda-downloads): Only if you intend to use the GPU functionality for high performance
- **[Git](http://git-scm.com/)** & **[GitHub](https://github.com/)**: If you wish to use or contribute to the most up-to-date BrainGrid that is currently under development, you will need to get it from GitHub and keep it in sync. 
- **Matlab** or **Octave**: If you want to view the output results using our scripts

## 2.2 Download BrainGrid

In order to get started with BrainGrid, you will need to build it from scratch, which means getting its source codes. You can either download BrainGrid source codes as a zip file of a stable release (See [2.2.1 Download a release](#221-download-a-release)) or fork the development version from BrainGrid GitHub repository (See [2.2.2 Fork and clone BrainGrid](#222-fork-and-clone-braingrid)).  

### 2.2.1 Download a release

Point your browser to *https://github.com/UWB-Biocomputing/BrainGrid/releases* and download a release from the list by clicking the relevant source code button. 

- [Source code (zip)](https://github.com/UWB-Biocomputing/BrainGrid/archive/v0.9-alpha.zip)
- [Source code (tar.gz)](https://github.com/UWB-Biocomputing/BrainGrid/archive/v0.9-alpha.tar.gz)

After downloading the source code, unpack it in a convenient location and continue to install BrainGrid as described in the [Installing BrainGrid](#2.2-installing-braingrid) section below. 

### 2.2.2 Fork and clone BrainGrid

First, if you are new to Github, get your [Github](https://github.com/) account ready and follow the below steps to download BrianGrid. You can also go over our [Git Crash Course](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Git-Crash-Course) for some useful tips.

1. To fork a repo, navigate to the [BrianGrid Github](https://github.com/UWB-Biocomputing/BrainGrid) page and click on the **Fork** button

2. Open terminal and go to the directory you plan to have BrainGrid placed. For example, you can place it under your home directory  `$ cd ~`

   ```shell
   $ cd YOUR_PREFERED_PATH
   ```

3. To clone the forked repo to your local machine: [How](https://help.github.com/articles/fork-a-repo/)

   ```shell
   $ git clone https://github.com/YOUR_USERNAME/BrainGrid.git
   ```

Now that you have the entire BrainGrid repository, including all open branches. By default you will be on the `master` branch which is the most recent version (might be unstable). If you are a user, this should be the right branch for you. If you are a collaborator and wish to work on specific part of the BrainGrid, check out to your desired branch or create a new branch. 

### 2.2.3 Keep your forked BrainGrid in sync

If you are forking the BrainGrid that is currently under development, you will want to sync your forked repo with our original repo from time to time to keep it up-to-date ([Why](https://help.github.com/articles/syncing-a-fork/)). Here is how to set up origin & upstream to keep your forked repo in sync: 

- List the current configured remote repository for your fork. 

  When a repo is cloned, the default remote **origin** is your fork on Github, not BrainGrid repo it was forked from.
  
  ```shell
  $ git remote -v 
  origin  https://github.com/YOUR_USERNAME/public_html.git (fetch)
  origin  https://github.com/YOUR_USERNAME/public_html.git (push)
  ```

- Set BrainGrid as your new remote **upstream** in order to keep your local copy in sync with the BrainGrid.

  ```shell
  $ git remote add upstream https://github.com/UWB-Biocomputing/BrainGrid.git
  ```

- Verify the new remote repository you've specified for your fork. 

  ```shell
  $ git remote -v
  origin  https://github.com/YOUR_USERNAME/BrainGrid.git (fetch)
  origin  https://github.com/YOUR_USERNAME/BrainGrid.git (push)
  upstream        https://github.com/UWB-Biocomputing/BrainGrid.git (fetch)
  upstream        https://github.com/UWB-Biocomputing/BrainGrid.git (push)
  ```

  if you want to see more detail, do:

  ```shell
  $ git remote show origin
  $ git remote show upstream
  ```

- Syncing your fork by fetching from upstream BrainGrid repo: [How](https://help.github.com/articles/syncing-a-fork/) 

  ```shell
  $ git fetch upstream
  ```

   Merge the changes from `upstream/master` into your local `master` branch

  ```shell
  $ git checkout master
  $ git merge upstream/master
  ```

Now your fork's `master` branch is in sync with the latest BrainGrid repo without losing your local changes. You are now all set to use BrainGrid.

## 2.3 Install BrainGrid

In order to compile and run BrainGrid, you will need to set up a couple things in the **Makefile** first. 

1. Change to BrainGrid directory in your terminal

   ```shell
   $ cd BrainGrid
   ```

2. Open **Makefile** and change the following parameters:

-  If you are using **CUDA**, 

   - change the CUDA library directory: ```CUDALIBDIR = YOUR_CUDA_LIBRARY_PATH``` 
   - you might also need to add your CUDA home directory into the ```PATH``` environment variable 

-  If you are using **HDF5**, 

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

--------

[<< Previous: 1. Introduction](http://uwb-biocomputing.github.io/BrainGrid/1_introduction)
