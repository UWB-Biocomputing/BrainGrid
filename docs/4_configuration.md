# 4.  Configuring the model

Now that you have run through a quick test and made sure you have a working BrainGrid repository, it is time to learn how to use it!

We will be going through this in a few steps:

1. First, we will look at how to implement a quick and dirty model and simulation parameters, which will involve putting together all the files that BrainGrid uses as inputs.

2. Second, we will configure BrainGrid to use a GPU (you've already seen how to do it with a single thread). And then we will run the simulation.

3. Lastly, we will collect BrainGrid's output and examine a few ways one might actually visualize the data.

Ready? Okay.

## 4.1 Inside the Config files

There are two config files needed to run a simulation on BrainGrid:

1. The input (or "stimulation") file - **SimInfoParams**
2. The model configuration - **ModelParams**

First, we are going to go through using a built-in model. This is by far the easiest route - if you have a quick idea you want to play with that uses a grid of **Izhikivich** or **LIF** neurons, go for it! As long as you only want to use excitatory and inhibitory neurons, this is the way to go. I'll show you how to specify the parameters you want and then run the simulation.

If on the other hand, you have a more complicated model in mind - such as using different types of neurotransmitters, then you will have to get your hands dirty writing some C++ code. Don't worry though, I'll walk you through that too.

## 4.2.2 Use built-in models

Let's go through the steps required to use a built-in model.

Take a look at **test-tiny.xml** file that is under  `BrainGrid/validation`  directory: 

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Parameter file for the DCT growth modeling -->
<!-- This file holds constants, not state information -->
<SimInfoParams>
   <!-- size of pool of neurons [x y z] -->
   <PoolSize x="2" y="2" z="1"/>
   <!-- Simulation Parameters -->
   <SimParams Tsim="1.0" numSims="1"/>
   <!-- Simulation Configuration Parameters -->
   <SimConfig maxFiringRate="200" maxSynapsesPerNeuron="200"/>
   <!-- Random seed - set to zero to use /dev/random -->
   <!-- TODO: implement support for this -->
   <Seed value="1"/>
   <!-- State output file name, this name will be overwritten if -o option is specified -->
   <OutputParams stateOutputFileName="test-tiny-out.xml"/>
</SimInfoParams>

<ModelParams>
   <NeuronsParams class="AllLIFNeurons">
      <!-- Interval of constant injected current -->
      <Iinject min="13.5e-09" max="13.5e-09"/>
      <!-- Interval of STD of (gaussian) noise current -->
      <Inoise min="1.0e-09" max="1.5e-09"/>
      <!-- Interval of firing threshold -->
      <Vthresh min="15.0e-03" max="15.0e-03"/>
      <!-- Interval of asymptotic voltage -->
      <Vresting min="0.0" max="0.0"/>
      <!-- Interval of reset voltage -->
      <Vreset min="13.5e-03" max="13.5e-03"/>
      <!-- Interval of initial membrance voltage -->
      <Vinit min="13.0e-03" max="13.0e-03"/>
      <!-- Starter firing threshold -->
      <starter_vthresh min="13.565e-3" max="13.655e-3"/>
      <!-- Starter reset voltage -->
      <starter_vreset min="13.0e-3" max="13.0e-3"/>
   </NeuronsParams>
   
   <SynapsesParams class="AllDSSynapses">
   </SynapsesParams>
   
   <ConnectionsParams class="ConnGrowth">
      <!-- Growth parameters -->
      <GrowthParams epsilon="0.60" beta="0.10" rho="0.0001" targetRate="1.9" minRadius="0.1" startRadius="0.4"/>
   </ConnectionsParams>

   <LayoutParams class="FixedLayout">

      <!-- If FixedLayout is present, the grid will be laid out according to the positions below, rather than randomly based on LsmParams -->
      <FixedLayoutParams>
         <!-- 0-indexed positions of endogenously active neurons in the list -->
         <A>0</A>

         <!-- 0-indexed positions of inhibitory neurons in the list -->
         <I>1</I>
        
         <!--
         original 10x10 (not updated for 30x30)
            9 . . . . . . . A . .
            8 . . . . . . . . . .
            7 . A . . A . . . . .
            6 . . . . . . I A . .
            5 . . . . . . . . . .
            4 . A . . A . . . . .
            3 . . . I . . . A . .
            2 . . . . . . . . . .
            1 . A . . A . . . . .
            0 . . . . . . . A . .
              0 1 2 3 4 5 6 7 8 9
         -->
      </FixedLayoutParams>
   </LayoutParams>
</ModelParams>
```

This is a typical example of a model configuration file that you must give to use BrainGrid. This type of file is mandatory - BrainGrid won't run without specifying model parameters. Even if you plan on writing your own model from "scratch", you may want to read this section anyway.

You can see that this file is a pretty standard XML file. It has tags that specify what each section is, like `<SimInfoParams>` and end tags that end said section, like `</SimInfoParams>`. Within each section, you can have sub-sections ad infinitum (in fact, XML files follow a tree structure, with a root node which branches into a top level of nodes, which branch into their own nodes, which branch, etc.)

Notice the `<!-- Parameter file for the DCT growth modeling -->` Anything that follows that pattern (i.e., `<!-- blah blah blah -->` is a comment, and won't have any effect on anything. It is good practice to comment stuff in helpful, far-seeing ways.

On to the actual parameters.

#### SimInfoParams

The first set of parameters that BrainGrid expects out of this file is stored in the SimInfoParams node. These parameters are required no matter what your model is. Here you must specify the:

* **PoolSize**: the three dimensional grid of neurons' parameters - expects an x (how many neurons are on the x axis), a y (how many neurons are on the y axis) and a z (not currently used). These three numbers together form a network of neurons that is x by y by z neurons (though in reality, the z dimension is not currently implemented).
* **SimParams**: the time configurations - expects a Tsim, which is how much time the simulation is simulating (in seconds) and a numSims, which is how many times to run the simulation (each simulation cycle picks up where the previous one left off)
* **SimConfig**: the maxFiringRate of a neuron and the maxSynapsesPerNeuron (the limitations of the simulation). Note the rate is in Hz.
* **Seed**: a random seed for the random generator.
* **OutputParams**: requires stateOutputFileName, which is where the simulator will store the output file.

#### ModelParams

The next set of parameters is the ModelParams. These parameters are specific to your model. Later, when we go through the "from scratch" example (where you will code up your own model using C++ to provide utmost flexibility), you will specify what goes here. But for now, we are using a built in model - specifically LIF (leaky integrate and fire), just to see what's expected. You must specify the:

* **NeuronsParams**: This is an XML node in and of itself, which requires several items. Each of these items is presented as a range, with the idea that each neuron will be chosen with random values from each of these intervals.
    + **Iinject**: The interval of constant injected current. Each neuron will be randomly assigned a value from this interval on start (with a uniform distribution).
    + **Inoise**: Describes the background (noise) current, if you want some in your experiment (simulates realistic settings). Each neuron will have a background noise current chosen from this range.
    + **Vthresh**: The threshold membrane voltage that must be reached before a neuron fires; again, specified as a range of values from which each neuron will be chosen randomly.
    + **Vresting**: The resting membrane potential of a neuron.
    + **Vreset**: The voltage that a neuron gets reset to after firing.
    + **Vinit**: The starting voltage of a neuron.
    + **starter_vthresh**: In this particular model, there are endogenously active neurons called 'starter neurons', whose threshold voltage is drawn from this range. This range is set low enough that their noise can actually drive them to fire, so that there need not be any input into the neural net. You can of course, configure these neurons to be exactly the same as the other ones, but without then coding an input to the net, your net won't do anything.
    + **starter_vreset**: The voltage to which a starter neuron gets reset after firing.

* **SynapsesParams**: Another node that should be populated - though you'll note in this particular example, we aren't specifying anything about the synapses.

* **ConnectionsParams**: Another node to populate. Its parameters are as follows:
    + **GrowthParams**: The growth parameters for this simulation. The mathematics behind epsilon, beta, and rho can be found [TODO]. The targetRate is TODO, and the minRadius, and startRadius should be self-explanatory.

* **Layout Params**: Another node to populate. Its only parameter is:
    + **FixedLayoutParams**: As you can see from the helpful comment, the simulator will use this if specified, rather than randomly placing the neurons.

