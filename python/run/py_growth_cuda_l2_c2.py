import sys
import growth_cuda
import xml.etree.ElementTree as ET

# create simulation info object
simInfo = growth_cuda.SimulationInfo()

# Set simulation parameters
simInfo.epochDuration = 100
simInfo.width = 100
simInfo.height = 100
simInfo.totalNeurons = simInfo.width * simInfo.height
simInfo.maxSteps = 600
simInfo.maxFiringRate = 200
simInfo.maxSynapsesPerNeuron = 200
simInfo.seed = 777
simInfo.stateOutputFileName = "../results/tR_1.0--fE_0.98_10000_historyDump.h5"
simInfo.numClusters = 2

# Create an instance of the neurons class
neurons = growth_cuda.AllLIFNeurons()
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
synapses = growth_cuda.AllDSSynapses()
synapses.createSynapsesProps()
synapsesProps = synapses.synapsesProps

# Create a second neurons class object and copy neurons parameters
# from the first one
neurons_2 = growth_cuda.AllLIFNeurons()
neurons_2.createNeuronsProps()
neuronsProps_2 = neurons_2.neuronsProps
growth_cuda.copyNeurons(neurons_2, neurons)

# Create a second synapses class object and copy synapses parameters
# from the first one
synapses_2 = growth_cuda.AllDSSynapses()
synapses_2.createSynapsesProps()

# Create an instance of the connections class
conns = growth_cuda.ConnGrowth()

# Set connections parameters
conns.epsilon = 0.6
conns.beta = 0.1
conns.rho = 0.0001
conns.targetRate = 1.0
conns.minRadius = 0.1
conns.startRadius = 0.4
conns.maxRate = conns.targetRate / conns.epsilon

# Create an instance of the layout class
layout = growth_cuda.FixedLayout()
# Read endogenously active neurons list and set it
endogenously_active_neuron_tree = ET.parse('configfiles/NList/activeNList_0.10_10000.xml')
endogenously_active_neuron_list = endogenously_active_neuron_tree.getroot().text.split()
endogenously_active_neuron_list_i = [int(s) for s in endogenously_active_neuron_list]
layout.set_endogenously_active_neuron_list(endogenously_active_neuron_list_i)
layout.num_endogenously_active_neurons = len(layout.get_endogenously_active_neuron_list())
# Read inhibitory neurons list and set it
inhibitory_neuron_tree = ET.parse('configfiles/NList/inhNList_0.98_10000.xml')
inhibitory_neuron_list = inhibitory_neuron_tree.getroot().text.split()
inhibitory_neuron_list_i = [int(s) for s in inhibitory_neuron_list]
layout.set_inhibitory_neuron_layout(inhibitory_neuron_list_i)

# Create clustersInfo
clusterInfo = growth_cuda.ClusterInfo()
clusterInfo.clusterID = 0
clusterInfo.clusterNeuronsBegin = 0
clusterInfo.totalClusterNeurons = int(simInfo.totalNeurons / 2)
clusterInfo.seed = simInfo.seed
clusterInfo.deviceId = 0

clusterInfo_2 = growth_cuda.ClusterInfo()
clusterInfo_2.clusterID = 1
clusterInfo_2.clusterNeuronsBegin = clusterInfo.totalClusterNeurons
clusterInfo_2.totalClusterNeurons = int(simInfo.totalNeurons / 2)
clusterInfo_2.seed = simInfo.seed + 1
clusterInfo.deviceId = 1

# Create clsuters
cluster = growth_cuda.GPUSpikingCluster(neurons, synapses)
cluster_2 = growth_cuda.GPUSpikingCluster(neurons_2, synapses_2)

vtClr = [cluster, cluster_2]
vtClrInfo = [clusterInfo, clusterInfo_2]

# Create a model
# To keep the C++ model object alive, we need to save the C++ model object pointer
# in the python variable so that python can manage the C++ model object.
# When the Python variable (model) is destroyed, the C++ model object is also deleted.
model = growth_cuda.Model(conns, layout, vtClr, vtClrInfo)
simInfo.model = model

# create & init simulation recorder
# To keep the C++ recorder object alive we need to save the C++ recorder object pointer 
# in the python variable so that python can manage the C++ recorder object.
# When the Python variable (recorder) is destroyed, the C++ recorder object is also deleted.
recorder = growth_cuda.createRecorder(simInfo)
if recorder == None:
    print("! ERROR: invalid state output file name extension.", file=sys.stderr)
    sys.exit(-1)

# Set the C++ recorder object in the C++ internal simInfo structure.
simInfo.simRecorder = recorder

# create the simulator
simulator = growth_cuda.Simulator()

# setup simulation
simulator.setup(simInfo)

# Run simulation
simulator.simulate(simInfo)

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
