import sys
import growth_cuda

argv = sys.argv

# create simulation info object
simInfo = growth_cuda.SimulationInfo()

# Handles parsing of the command line
if growth_cuda.parseCommandLine(argv, simInfo) == False:
    print("! ERROR: failed during command line parse", file=sys.stderr)
    sys.exit(-1)

# Create all model instances and load parameters from a file.
vtClr = []
vtClrInfo = []
if growth_cuda.LoadAllParameters(simInfo, vtClr, vtClrInfo) == False:
    print("! ERROR: failed while parsing simulation parameters.", file=sys.stderr)
    sys.exit(-1)

# Save the C++ object in Python variable and transfer the ownership of the object
# to Python.
model = simInfo.model

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
