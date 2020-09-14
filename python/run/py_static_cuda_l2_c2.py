import sys
import growth_cuda

# create simulation info object
simInfo = growth_cuda.SimulationInfo()

# Set simulation parameters
simInfo.epochDuration = 1
simInfo.width = 40
simInfo.height = 25
simInfo.totalNeurons = simInfo.width * simInfo.height
simInfo.maxSteps = 1
simInfo.maxFiringRate = 1000
simInfo.maxSynapsesPerNeuron = 1000
simInfo.seed = 777
simInfo.stateOutputFileName = "../results/static_izh_historyDump.h5"
simInfo.numClusters = 2

# Create an instance of the neurons class
neurons = growth_cuda.AllIZHNeurons()
neurons.createNeuronsProps()
neuronsProps = neurons.neuronsProps

# Set neurons parameters
neuronsProps.Iinject[0] = 13.5e-09
neuronsProps.Iinject[1] = 13.5e-09
neuronsProps.Inoise[0] = 0.5e-06
neuronsProps.Inoise[1] = 0.7329e-06
neuronsProps.Vthresh[0] = 30.0e-03
neuronsProps.Vthresh[1] = 30.0e-03
neuronsProps.Vresting[0] = 0.0
neuronsProps.Vresting[1] = 0.0
neuronsProps.Vreset[0] = -0.065
neuronsProps.Vreset[1] = -0.065
neuronsProps.Vinit[0] = -0.065
neuronsProps.Vinit[1] = -0.065
neuronsProps.starter_Vthresh[0] = 13.565e-3
neuronsProps.starter_Vthresh[1] = 13.655e-3
neuronsProps.starter_Vreset[0] = 13.0e-3
neuronsProps.starter_Vreset[1] = 13.0e-3
neuronsProps.excAconst[0] = 0.02
neuronsProps.excAconst[1] = 0.02
neuronsProps.inhAconst[0] = 0.02
neuronsProps.inhAconst[1] = 0.1
neuronsProps.excBconst[0] = 0.2
neuronsProps.excBconst[1] = 0.2
neuronsProps.inhBconst[0] = 0.2
neuronsProps.inhBconst[1] = 0.25
neuronsProps.excCconst[0] = -65
neuronsProps.excCconst[1] = -50
neuronsProps.inhCconst[0] = -65
neuronsProps.inhCconst[1] = -65
neuronsProps.excDconst[0] = 2
neuronsProps.excDconst[1] = 8
neuronsProps.inhDconst[0] = 2
neuronsProps.inhDconst[1] = 2

# Create an instance of the synapses class
synapses = growth_cuda.AllSpikingSynapses()
synapses.createSynapsesProps()
synapsesProps = synapses.synapsesProps

# Create a second neurons class object and copy neurons parameters
# from the first one
neurons_2 = growth_cuda.AllIZHNeurons()
neurons_2.createNeuronsProps()
neuronsProps_2 = neurons_2.neuronsProps
growth_cuda.copyNeurons(neurons_2, neurons)

# Create a second synapses class object and copy synapses parameters
# from the first one
synapses_2 = growth_cuda.AllSpikingSynapses()
synapses_2.createSynapsesProps()

# Create an instance of the connections class
conns = growth_cuda.ConnStatic()

# Set connections parameters
conns.nConnsPerNeuron = 999
conns.threshConnsRadius = 50
conns.pRewiring = 0
conns.excWeight[0] = 0
conns.excWeight[1] = 0.5e-7
conns.inhWeight[0] = -0.5e-7
conns.inhWeight[1] = 0

# Create an instance of the layout class
layout = growth_cuda.FixedLayout()
layout.num_endogenously_active_neurons = 0
layout.set_inhibitory_neuron_layout(list(range(800, 1000)))
layout.set_probed_neuron_list(list(range(0, 1000)))

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
clusterInfo_2.deviceId = 1

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
