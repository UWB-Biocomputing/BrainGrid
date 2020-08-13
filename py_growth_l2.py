import sys
import growth

# create simulation info object
simInfo = growth.SimulationInfo()

# Set simulation parameters
simInfo.epochDuration = 100
simInfo.width = 10
simInfo.height = 10
simInfo.totalNeurons = simInfo.width * simInfo.height
simInfo.maxSteps = 10
simInfo.maxFiringRate = 200
simInfo.maxSynapsesPerNeuron = 200
simInfo.seed = 777
simInfo.stateOutputFileName = "results/tR_1.0--fE_0.98_historyDump.h5"
simInfo.numClusters = 1

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
synapses = growth.AllDSSynapses()
synapses.createSynapsesProps()
synapsesProps = synapses.synapsesProps

# Create an instance of the connections class
conns = growth.ConnGrowth()

# Set connections parameters
conns.epsilon = 0.6
conns.beta = 0.1
conns.rho = 0.0001
conns.targetRate = 1.0
conns.minRadius = 0.1
conns.startRadius = 0.4
conns.maxRate = conns.targetRate / conns.epsilon

# Create an instance of the layout class
layout = growth.FixedLayout()
layout.set_endogenously_active_neuron_list( [7, 11, 14, 37, 41, 44, 67, 71, 74, 97] )
layout.num_endogenously_active_neurons = len(layout.get_endogenously_active_neuron_list())
layout.set_inhibitory_neuron_layout( [33, 66] )

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

# create the simulator
simulator = growth.Simulator()

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
