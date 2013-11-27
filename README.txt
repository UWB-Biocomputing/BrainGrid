# What is The BrainGrid Project? #

The idea behind The BrainGrid Project is to develop a toolkit/software architecture to ease creating high-performance neural network simulators. It is particularly focused on facilitating biologically realistic modeling. BrainGrid is currently under development, and not quite ready for prime time, but we expect it to be available for beta test during the first half of 2014. Our current focus is on single-threaded simulators and simulators running on GPUs using NVIDIA's CUDA libraries. We expect to shortly thereafter roll out support for multi-threading using OpenMP.

# What are the basic ideas behind BrainGrid? #

The initial principles that we are basing BrainGrid on are as follows:

* Provide support for common algorithms and data structures needed for biological neural simulation. This includes concepts such as:
  + _neurons_ and _synapses_, which can be dynamical and have internal state that must be initialized, can be updated, and can be serialized and deserialized,
  + neuron outputs can fan out to multiple synapses,
  + multiple synapses fan in to single neurons via _summation points_,
  + neurons have (x, y) spatial coordinates,
  + synapses can have individualized transmission delays, and
  + _noise generation_ is available to be added to state variables.
* Be constructred with a design that provides useful metaphors for thinking about simulator implementation in the context of different maching architectures. In other words, if we want to have high performance, we need to expose enough of the underlying machine architecture so that we can take advantage of this. We are not shooting for a high-level tool or language that will give a 2X speedup when one moves one's code to a GPU; we're looking for at least a 20X speedup.
* We're assuming that a researcher/simulator developer is starting with an implementation that runs as a single thread of execution on a generic host computer. This may be an already-existing simulator or it may be a desire to develop a new one. Thus, the entry point for BrainGrid use is the classes and data structures associated with single-threaded execution.
* We're also assuming that the user wants to move that simulation to a parallel implementation. So, we have an architecture that acknowledges that there are two orthogonal implementaiton axes: the model being simulated and the platform being delivered on. So, the user is asked to decompose and structure the simulation code to separate out components that are common to any platform. This means that platform-dependent code is segregated out, easing the changes necessary for porting, and core data structures are organized to accommodate the different platforms. Users should be able to implement a single-threaded simulator, verify its correct operation, and then move their code to a parallel platform and validate against the single-threaded version.
