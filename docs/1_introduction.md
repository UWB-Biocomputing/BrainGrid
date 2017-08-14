# 1. Introduction

## 1.1 What is BrainGrid?

BrainGrid is an **open-source spiking neural network simulator** that is intended to aid scientists and researchers by providing pre-built code that can be easily modified to fit different models. The program also offers the ability to easily transition to a GPU-centered model with little additional work, which can provide a potential speedup of up to a twentieth of the original runtime.

Our current focus is on single-threaded simulators and simulators running on GPUs using NVIDIA's CUDA libraries. We expect to shortly thereafter roll out support for multi-cluster, and potentially multi-threading using OpenMP in the future. 

## 1.2 What is BrainGrid for?

BrainGrid is a framework for quickly implementing GPU-speed-up spiking neural networks. If you have a mathematical model for a spiking neural network, or if you have an idea for one, BrainGrid will provide you with a quick way of getting that network implemented in code and running on CPUs or GPUs.

In sum, BrainGrid is designed to do the following:

1. Simulate spiking neural networks with built-in models (such as leaky-integrate-and-fire neuron, izhikevich neuron)
2. Provide a framework to make it easier to code models of your own
3. Make these models FAST - provide code to make it easy to migrate models to single or multiple GPUs, providing a huge speedup from single or multi-threaded host-side models.

## 1.3 Why do we need BrainGrid?

The initial principles that we are basing BrainGrid on are as follows:

- Provide support for common algorithms and data structures needed for biological neural simulation. This includes concepts such as:
  - *neurons* and *synapses*, which can be dynamical and have internal state that must be initialized, can be updated, and can be serialized and deserialized,
  - neuron outputs can fan out to multiple synapses,
  - multiple synapses fan in to single neurons via *summation points*,
  - neurons have (x, y) spatial coordinates,
  - synapses can have individualized transmission delays, and
  - **noise generation** is available to be added to state variables.
- Be constructred with a design that provides useful metaphors for thinking about simulator implementation in the context of different maching architectures. In other words, if we want to have high performance, we need to expose enough of the underlying machine architecture so that we can take advantage of this. We are not shooting for a high-level tool or language that will give a 2x speedup when one moves one's code to a GPU; we are looking for at least a 20x speedup.
- We're assuming that a researcher/simulator developer is starting with an implementation that runs as a single thread of execution on a generic host computer. This may be an already-existing simulator or it may be a desire to develop a new one. Thus, the entry point for BrainGrid use is the classes and data structures associated with single-threaded execution.
- We're also assuming that the user wants to move that simulation to a parallel implementation. So, we have an architecture that acknowledges that there are two orthogonal implementation axes: the model being simulated and the platform being delivered on. So, the user is asked to decompose and structure the simulation code to separate out components that are common to any platform. This means that platform-dependent code is segregated out, easing the changes necessary for porting, and core data structures are organized to accommodate the different platforms. Users should be able to implement a single-threaded simulator, verify its correct operation, and then move their code to a parallel platform and validate against the single-threaded version. 


-------------
[>> Next: 2. Installation](http://uwb-biocomputing.github.io/BrainGrid/2_installation)

-------------
[<< Previous:  Table of Contents](http://uwb-biocomputing.github.io/BrainGrid/)
