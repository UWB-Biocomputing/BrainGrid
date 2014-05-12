################################################################################
# Default Target
################################################################################
all: growth growth_cuda

################################################################################
# Source Directories
################################################################################
MAIN = .
CUDADIR = $(MAIN)/cuda
COMMDIR = $(MAIN)/common
MATRIXDIR = $(MAIN)/matrix
PARAMDIR = $(MAIN)/paramcontainer
RNGDIR = $(MAIN)/rng
XMLDIR = $(MAIN)/tinyxml

################################################################################
# Build tools
################################################################################
CXX = g++
LD = g++
OPT = g++ 

################################################################################
# Flags
################################################################################
CXXFLAGS = -O2 -s -I$(COMMDIR) -I$(MATRIXDIR) -I$(PARAMDIR) -I$(RNGDIR) -I$(XMLDIR) -Wall -g -pg -c -DTIXML_USE_STL -DDEBUG_OUT 
CGPUFLAGS = -DUSE_GPU
LDFLAGS = -lstdc++ 
LGPUFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart
LH5FLAGS =  -L/opt/hdf5/latest/lib -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5

################################################################################
# Objects
################################################################################

#CUDAOBJS =  $(CUDADIR)/CUDA_LIFModel.o \

CUDAOBJS =   \
	    $(COMMDIR)/LIFGPUModel.o \
	    $(COMMDIR)/GPUSimulator.o \
            $(CUDADIR)/LifNeuron_struct_d.o \
            $(CUDADIR)/DynamicSpikingSynapse_struct_d.o \
            $(CUDADIR)/MersenneTwister_kernel.o \
            $(CUDADIR)/BGDriver_cuda.o \
            $(CUDADIR)/Global_cuda.o

LIBOBJS = $(COMMDIR)/AllNeurons.o \
			$(COMMDIR)/AllSynapses.o \
			$(COMMDIR)/Simulator.o \
			$(COMMDIR)/SingleThreadedSim.o \
			$(COMMDIR)/LIFModel.o \
			$(COMMDIR)/LIFSingleThreadedModel.o \
			$(COMMDIR)/Network.o \
			$(COMMDIR)/ParseParamError.o \
			$(COMMDIR)/Timer.o \
			$(COMMDIR)/Util.o \
			$(COMMDIR)/XmlRecorder.o 

#			$(COMMDIR)/Hdf5Recorder.o \
 
		
MATRIXOBJS = $(MATRIXDIR)/CompleteMatrix.o \
				$(MATRIXDIR)/Matrix.o \
				$(MATRIXDIR)/SparseMatrix.o \
				$(MATRIXDIR)/VectorMatrix.o \

PARAMOBJS = $(PARAMDIR)/ParamContainer.o
		
RNGOBJS = $(RNGDIR)/Norm.o

SINGLEOBJS = $(MAIN)/BGDriver.o  \
			$(COMMDIR)/Global.o 

XMLOBJS = $(XMLDIR)/tinyxml.o \
			$(XMLDIR)/tinyxmlparser.o \
			$(XMLDIR)/tinyxmlerror.o \
			$(XMLDIR)/tinystr.o

################################################################################
# Targets
################################################################################
growth: $(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) 
	$(LD) -o growth -g $(LDFLAGS) $(LH5FLAGS) $(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) 

growth_cuda:$(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(XMLOBJS) $(OTHEROBJS) $(CUDAOBJS)
	nvcc -o growth_cuda -g -G $(LDFLAGS) $(LH5FLAGS) $(LGPUFLAGS) $(LIBOBJS) $(CUDAOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(XMLOBJS) $(OTHEROBJS)

clean:
	rm -f $(MAIN)/*.o $(COMMDIR)/*.o $(MATRIXDIR)/*.o $(PARAMDIR)/*.o $(RNGDIR)/*.o $(XMLDIR)/*.o ./growth
	

################################################################################
# Build Source Files
################################################################################

# CUDA
# ------------------------------------------------------------------------------
$(CUDADIR)/MersenneTwister_kernel.o: $(CUDADIR)/MersenneTwister_kernel.cu $(COMMDIR)/Global.h $(CUDADIR)/MersenneTwister.h

	nvcc -c -g -arch=sm_20 $(CUDADIR)/MersenneTwister_kernel.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -o $(CUDADIR)/MersenneTwister_kernel.o


$(COMMDIR)/LIFGPUModel.o: $(COMMDIR)/LIFGPUModel.cu $(COMMDIR)/Global.h $(COMMDIR)/LIFGPUModel.h $(COMMDIR)/AllNeurons.h $(COMMDIR)/AllSynapses.h $(COMMDIR)/Model.h 

	nvcc -c -g -G -arch=sm_20 $(COMMDIR)/LIFGPUModel.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -o $(COMMDIR)/LIFGPUModel.o

$(CUDADIR)/LifNeuron_struct_d.o: $(CUDADIR)/LifNeuron_struct_d.cu $(COMMDIR)/Global.h $(COMMDIR)/LIFGPUModel.h
	nvcc -c -g -arch=sm_20 $(CUDADIR)/LifNeuron_struct_d.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -o $(CUDADIR)/LifNeuron_struct_d.o

$(CUDADIR)/DynamicSpikingSynapse_struct_d.o: $(CUDADIR)/DynamicSpikingSynapse_struct_d.cu $(COMMDIR)/Global.h $(COMMDIR)/LIFGPUModel.h
	nvcc -c -g -arch=sm_20 $(CUDADIR)/DynamicSpikingSynapse_struct_d.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -o $(CUDADIR)/DynamicSpikingSynapse_struct_d.o

$(CUDADIR)/BGDriver_cuda.o: $(MAIN)/BGDriver.cpp $(COMMDIR)/Global.h $(COMMDIR)/Model.h $(COMMDIR)/AllNeurons.h $(COMMDIR)/AllSynapses.h $(COMMDIR)/Model.h $(COMMDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -I$(CUDADIR) -c $(MAIN)/BGDriver.cpp -o $(CUDADIR)/BGDriver_cuda.o

$(CUDADIR)/Global_cuda.o: $(COMMDIR)/Global.cpp $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(COMMDIR)/Global.cpp -o $(CUDADIR)/Global_cuda.o

$(CUDADIR)/GPUSimulator.o: $(COMMDIR)/GPUSimulator.cpp $(COMMDIR)/GPUSimulator.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(COMMDIR)/GPUSimulator.cpp -o $(CUDADIR)/GPUSimulator.o


# Library
# ------------------------------------------------------------------------------

$(COMMDIR)/AllNeurons.o: $(COMMDIR)/AllNeurons.cpp $(COMMDIR)/AllNeurons.h $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/AllNeurons.cpp -o $(COMMDIR)/AllNeurons.o
	
$(COMMDIR)/AllSynapses.o: $(COMMDIR)/AllSynapses.cpp $(COMMDIR)/AllSynapses.h $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/AllSynapses.cpp -o $(COMMDIR)/AllSynapses.o

$(COMMDIR)/Global.o: $(COMMDIR)/Global.cpp $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Global.cpp -o $(COMMDIR)/Global.o

$(COMMDIR)/Simulator.o: $(COMMDIR)/Simulator.cpp $(COMMDIR)/Simulator.h $(COMMDIR)/Global.h $(COMMDIR)/SimulationInfo.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Simulator.cpp -o $(COMMDIR)/Simulator.o

$(COMMDIR)/SingleThreadedSim.o: $(COMMDIR)/SingleThreadedSim.cpp $(COMMDIR)/SingleThreadedSim.h $(COMMDIR)/Simulator.h $(COMMDIR)/Global.h $(COMMDIR)/SimulationInfo.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/SingleThreadedSim.cpp -o $(COMMDIR)/SingleThreadedSim.o

$(COMMDIR)/LIFModel.o: $(COMMDIR)/LIFModel.cpp $(COMMDIR)/LIFModel.h $(COMMDIR)/Model.h $(COMMDIR)/ParseParamError.h $(COMMDIR)/Util.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/LIFModel.cpp -o $(COMMDIR)/LIFModel.o

$(COMMDIR)/LIFSingleThreadedModel.o: $(COMMDIR)/LIFSingleThreadedModel.cpp $(COMMDIR)/LIFSingleThreadedModel.h $(COMMDIR)/LIFModel.h 
	$(CXX) $(CXXFLAGS) $(COMMDIR)/LIFSingleThreadedModel.cpp -o $(COMMDIR)/LIFSingleThreadedModel.o

$(COMMDIR)/Network.o: $(COMMDIR)/Network.cpp $(COMMDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Network.cpp -o $(COMMDIR)/Network.o

$(COMMDIR)/ParseParamError.o: $(COMMDIR)/ParseParamError.cpp $(COMMDIR)/ParseParamError.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/ParseParamError.cpp -o $(COMMDIR)/ParseParamError.o

$(COMMDIR)/Timer.o: $(COMMDIR)/Timer.cpp $(COMMDIR)/Timer.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Timer.cpp -o $(COMMDIR)/Timer.o

$(COMMDIR)/Util.o: $(COMMDIR)/Util.cpp $(COMMDIR)/Util.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Util.cpp -o $(COMMDIR)/Util.o

$(COMMDIR)/XmlRecorder.o: $(COMMDIR)/XmlRecorder.cpp $(COMMDIR)/XmlRecorder.h $(COMMDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/XmlRecorder.cpp -o $(COMMDIR)/XmlRecorder.o

$(COMMDIR)/Hdf5Recorder.o: $(COMMDIR)/Hdf5Recorder.cpp $(COMMDIR)/Hdf5Recorder.h $(COMMDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Hdf5Recorder.cpp -o $(COMMDIR)/Hdf5Recorder.o


# Matrix
# ------------------------------------------------------------------------------

$(MATRIXDIR)/CompleteMatrix.o: $(MATRIXDIR)/CompleteMatrix.cpp  $(MATRIXDIR)/CompleteMatrix.h $(MATRIXDIR)/KIIexceptions.h $(MATRIXDIR)/Matrix.h $(MATRIXDIR)/VectorMatrix.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/CompleteMatrix.cpp -o $(MATRIXDIR)/CompleteMatrix.o

$(MATRIXDIR)/Matrix.o: $(MATRIXDIR)/Matrix.cpp $(MATRIXDIR)/Matrix.h  $(MATRIXDIR)/KIIexceptions.h  $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/Matrix.cpp -o $(MATRIXDIR)/Matrix.o

$(MATRIXDIR)/SparseMatrix.o: $(MATRIXDIR)/SparseMatrix.cpp $(MATRIXDIR)/SparseMatrix.h  $(MATRIXDIR)/KIIexceptions.h $(MATRIXDIR)/Matrix.h $(MATRIXDIR)/VectorMatrix.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/SparseMatrix.cpp -o $(MATRIXDIR)/SparseMatrix.o

$(MATRIXDIR)/VectorMatrix.o: $(MATRIXDIR)/VectorMatrix.cpp $(MATRIXDIR)/VectorMatrix.h $(MATRIXDIR)/CompleteMatrix.h $(MATRIXDIR)/SparseMatrix.h $(MATRIXDIR)/
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/VectorMatrix.cpp -o $(MATRIXDIR)/VectorMatrix.o


# ParamContainer
# ------------------------------------------------------------------------------

$(PARAMDIR)/ParamContainer.o: $(PARAMDIR)/ParamContainer.cpp $(PARAMDIR)/ParamContainer.h
	$(CXX) $(CXXFLAGS) $(PARAMDIR)/ParamContainer.cpp -o $(PARAMDIR)/ParamContainer.o

# RNG
# ------------------------------------------------------------------------------

$(RNGDIR)/Norm.o: $(RNGDIR)/Norm.cpp $(RNGDIR)/Norm.h $(RNGDIR)/MersenneTwister.h $(COMMDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(RNGDIR)/Norm.cpp -o $(RNGDIR)/Norm.o


# XML
# ------------------------------------------------------------------------------

$(XMLDIR)/tinyxml.o: $(XMLDIR)/tinyxml.cpp $(XMLDIR)/tinyxml.h $(XMLDIR)/tinystr.h $(COMMDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxml.cpp -o $(XMLDIR)/tinyxml.o

$(XMLDIR)/tinyxmlparser.o: $(XMLDIR)/tinyxmlparser.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlparser.cpp -o $(XMLDIR)/tinyxmlparser.o

$(XMLDIR)/tinyxmlerror.o: $(XMLDIR)/tinyxmlerror.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlerror.cpp -o $(XMLDIR)/tinyxmlerror.o

$(XMLDIR)/tinystr.o: $(XMLDIR)/tinystr.cpp $(XMLDIR)/tinystr.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinystr.cpp -o $(XMLDIR)/tinystr.o

# Single Threaded
# ------------------------------------------------------------------------------

$(MAIN)/BGDriver.o: $(MAIN)/BGDriver.cpp $(COMMDIR)/Global.h $(COMMDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(MAIN)/BGDriver.cpp -o $(MAIN)/BGDriver.o



