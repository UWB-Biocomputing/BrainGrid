#!/home/NETID/lundvm/anaconda3/bin/python3

#########################################################################
#
# @file         py_input_regular_single.py
# @author       Fumitaka Kawasaki
# @date         11/02/2020
#
# @brief        Run one second simulation with a single LIF neuron, 
#               pour reghular input spike train, and record the psr of the synapse
#               and the membrane voltage (vm) of the neuron.
#
#########################################################################

import sys
import growth
import h5py

# create simulation info object
simInfo = growth.SimulationInfo()

# Set simulation parameters
simInfo.epochDuration = 1
simInfo.width = 1
simInfo.height = 1
simInfo.totalNeurons = simInfo.width * simInfo.height
simInfo.maxSteps = 1
simInfo.maxFiringRate = 200
simInfo.maxSynapsesPerNeuron = 200
simInfo.seed = 777
simInfo.stateOutputFileName = "../results/single_historyDump.h5"
simInfo.numClusters = 1
#simInfo.minSynapticTransDelay = 1

# Create an instance of the neurons class
neurons = growth.AllLIFNeurons()
neurons.createNeuronsProps()
neuronsProps = neurons.neuronsProps

# Set neurons parameters
neuronsProps.Iinject[0] = 13.5e-09
neuronsProps.Iinject[1] = 13.5e-09
neuronsProps.Inoise[0] = 1.0e-09
neuronsProps.Inoise[1] = 1.5e-09
neuronsProps.Vthresh[0] = 15.0e-03
neuronsProps.Vthresh[1] = 15.0e-03
neuronsProps.Vresting[0] = 0.0
neuronsProps.Vresting[1] = 0.0
neuronsProps.Vreset[0] = 13.5e-03
neuronsProps.Vreset[1] = 13.5e-03
neuronsProps.Vinit[0] = 13.0e-03
neuronsProps.Vinit[1] = 13.0e-03
neuronsProps.starter_Vthresh[0] = 13.565e-3
neuronsProps.starter_Vthresh[1] = 13.655e-3
neuronsProps.starter_Vreset[0] = 13.0e-3
neuronsProps.starter_Vreset[1] = 13.0e-3

# Create an instance of the synapses class
synapses = growth.AllSpikingSynapses()
synapses.createSynapsesProps()
synapsesProps = synapses.synapsesProps

# Create an instance of the connections class
conns = growth.ConnStatic()

# Set connections parameters
conns.nConnsPerNeuron = 0
conns.threshConnsRadius = 0
conns.pRewiring = 0
conns.excWeight[0] = 0
conns.excWeight[1] = 0
conns.inhWeight[0] = 0
conns.inhWeight[1] = 0

# Create an instance of the layout class
layout = growth.FixedLayout()
layout.set_endogenously_active_neuron_list(list()) # no active neurons
layout.num_endogenously_active_neurons = len(layout.get_endogenously_active_neuron_list())
layout.set_inhibitory_neuron_layout(list()) # no inhibitory neurons
layout.set_probed_neuron_list( [0] )

# Create clustersInfo
clusterInfo = growth.ClusterInfo()
clusterInfo.clusterID = 0
clusterInfo.clusterNeuronsBegin = 0
clusterInfo.totalClusterNeurons = simInfo.totalNeurons
clusterInfo.seed = simInfo.seed

# Create clsuters
cluster = growth.SingleThreadedCluster(neurons, synapses)

vtClr = [cluster]
vtClrInfo = [clusterInfo]

# Create a model
# To keep the C++ model object alive, we need to save the C++ model object pointer
# in the python variable so that python can manage the C++ model object.
# When the Python variable (model) is destroyed, the C++ model object is also deleted.
model = growth.Model(conns, layout, vtClr, vtClrInfo)
simInfo.model = model

# create & init simulation recorder
# To keep the C++ recorder object alive we need to save the C++ recorder object pointer
# in the python variable so that python can manage the C++ recorder object.
# When the Python variable (recorder) is destroyed, the C++ recorder object is also deleted.
recorder = growth.createRecorder(simInfo)
if recorder == None:
    print("! ERROR: invalid state output file name extension.", file=sys.stderr)
    sys.exit(-1)

# Set the C++ recorder object in the C++ internal simInfo structure.
simInfo.simRecorder = recorder

# Create a stimulus input object
maskIndex = []  # masks for the input
duration = 0.1  # 100 ms duration
interval = 0.4  # 400 ms interval
sync = 'yes'
weight = 50.0   # synapse weight
firingRate = 500 # firing rate (500Hz == 2ms interval)
sinput = growth.HostSInputRegular(simInfo, firingRate, duration, interval, sync, weight, maskIndex)
simInfo.pInput = sinput

# create the simulator
simulator = growth.Simulator()

# setup simulation
simulator.setup(simInfo)

# get the synapses properties for the input layer
synapsesSInput = clusterInfo.synapsesSInput
synapsesSInputProps = synapsesSInput.synapsesProps 

logTime = list()
logVm = list()
logPsr = list()

# Run simulation

# Main simulation loop - execute maxSteps
for currentStep in range(1, simInfo.maxSteps+1):
    simInfo.currentStep = currentStep

    # advanceUntilGrowth
    g_simulationStep = growth.getSimulationStep()
    # Compute step number at end of this simulation epoch
    endStep = int(g_simulationStep + simInfo.epochDuration / simInfo.deltaT)

    # Advance simulation to next growth cycle
    # This should simulate all neuron and synapse activity for one epoch.
    while g_simulationStep < endStep:
        # incremental step
        iStep = endStep - g_simulationStep
        iStep = iStep if (iStep < simInfo.minSynapticTransDelay) else simInfo.minSynapticTransDelay
        # Advance the Network iStep time step
        model.advance(simInfo, int(iStep))
        g_simulationStep += iStep 
        growth.setSimulationStep(g_simulationStep)

        # Get the neuron's membrane voltge of neuron 0
        Vm = growth.get_Vm(neuronsProps, 0)

        # Get the synapse's psr of the input layer synapse 0
        psr = growth.get_psr(synapsesSInputProps, 0)

        # log the voltage
        logTime.append(g_simulationStep)
        logVm.append(Vm)
        logPsr.append(psr)

    model.updateConnections(simInfo)
    model.updateHistory(simInfo)

# Terminate the stimulus input
if sinput != None:
    sinput.term(simInfo, vtClrInfo)

# Writes simulation results to an output destination
simulator.saveData(simInfo)

# Tell simulation to clean-up and run any post-simulation logic.
simulator.finish(simInfo)

# terminates the simulation recorder
recorder.term()

# All C++ objects that are managed by Python will be deleted here.
#     - Cluster
#     - ClusterInfo
#     - SimulationInfo
#     - Simulator
#     - Recorder
#     - Model
#     - SInput

# Save the neuron's membrane voltage record
with h5py.File('../results/vmLog.h5', 'w') as f:
    f.create_dataset('logTime', data=logTime)
    f.create_dataset('logVm', data=logVm)
    f.create_dataset('logPsr', data=logPsr)

