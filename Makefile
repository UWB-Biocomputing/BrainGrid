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
CXXFLAGS = -O2 -s -I$(COMMDIR) -I$(MATRIXDIR) -I$(PARAMDIR) -I$(RNGDIR) -I$(XMLDIR) -Wall -pg -c -DTIXML_USE_STL -DDEBUG_OUT -DSTORE_SPIKEHISTORY
CGPUFLAGS = -DUSE_GPU -I$(CUDADIR)
LDFLAGS = -lstdc++ 
LGPUFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

################################################################################
# Objects
################################################################################

CUDAOBJS =  $(CUDADIR)/CUDA_LIFModel.o \
            $(CUDADIR)/LifNeuron_struct.o \
            $(CUDADIR)/LifSynapse_struct.o \
            $(CUDADIR)/MersenneTwister_kernel.o \
            $(CUDADIR)/BGDriver_cuda.o


LIBOBJS = $(COMMDIR)/AllNeurons.o \
			$(COMMDIR)/AllSynapses.o \
			$(COMMDIR)/Global.o \
			$(COMMDIR)/HostSimulator.o \
			$(COMMDIR)/LIFModel.o \
			$(COMMDIR)/Network.o \
			$(COMMDIR)/ParseParamError.o \
			$(COMMDIR)/Timer.o \
			$(COMMDIR)/Util.o
		
MATRIXOBJS = $(MATRIXDIR)/CompleteMatrix.o \
				$(MATRIXDIR)/Matrix.o \
				$(MATRIXDIR)/SparseMatrix.o \
				$(MATRIXDIR)/VectorMatrix.o \

PARAMOBJS = $(PARAMDIR)/ParamContainer.o
		
RNGOBJS = $(RNGDIR)/Norm.o

SINGLEOBJS = $(MAIN)/BGDriver.o 

XMLOBJS = $(XMLDIR)/tinyxml.o \
			$(XMLDIR)/tinyxmlparser.o \
			$(XMLDIR)/tinyxmlerror.o \
			$(XMLDIR)/tinystr.o

################################################################################
# Targets
################################################################################
growth: $(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) 
	$(LD) -o growth -g $(LDFLAGS) $(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) 

growth_cuda:$(LIBOBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS) $(CUDAOBJS)
	nvcc -o growth_cuda -g -G $(LDFLAGS) $(LGPUFLAGS) $(LIBOBJS) $(CUDAOBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS)

clean:
	rm -f $(MAIN)/*.o $(COMMDIR)/*.o $(MATRIXDIR)/*.o $(PARAMDIR)/*.o $(RNGDIR)/*.o $(XMLDIR)/*.o $(CUDADIR)/*.o  ./growth ./growth_cuda
	

################################################################################
# Build Source Files
################################################################################

# CUDA
# ------------------------------------------------------------------------------
$(CUDADIR)/MersenneTwister_kernel.o: $(CUDADIR)/MersenneTwister_kernel.cu $(COMMDIR)/Global.h $(CUDADIR)/MersenneTwisterGPU.h
	nvcc -c -g -G -arch=sm_13 $(CUDADIR)/CUDA_LIFModel.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -DSTORE_SPIKEHISTORY -o $(CUDADIR)/MersenneTwister_kernel.o

$(CUDADIR)/CUDA_LIFModel.o: $(CUDADIR)/CUDA_LIFModel.cu $(COMMDIR)/Global.h $(CUDADIR)/MersenneTwisterGPU.h $(COMMDIR)/LifNeuron_struct.h $(CUDADIR)/DelayIdx.h $(CUDADIR)/LifSynapse_struct.h  $(COMMDIR)/AllNeurons.h $(COMMDIR)/AllSynapses.h $(COMMDIR)/Model.h 
	nvcc -c -g -G -arch=sm_13 $(CUDADIR)/CUDA_LIFModel.cu $(CGPUFLAGS) -I$(CUDADIR) -I$(COMMDIR) -I$(MATRIXDIR) -DSTORE_SPIKEHISTORY -o $(CUDADIR)/CUDA_LIFModel.o

$(CUDADIR)/LifNeuron_struct.o: $(CUDADIR)/LifNeuron_struct.cpp $(COMMDIR)/Global.h $(COMMDIR)/LifNeuron_struct.h $(CUDADIR)/LifSynapse_struct.h $(CUDADIR)/DelayIdx.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(CUDADIR)/LifNeuron_struct.cpp -o $(CUDADIR)/LifNeuron_struct.o

$(CUDADIR)/LifSynapse_struct.o: $(CUDADIR)/LifSynapse_struct.cpp $(COMMDIR)/Global.h $(COMMDIR)/LifNeuron_struct.h $(CUDADIR)/LifSynapse_struct.h $(CUDADIR)/DelayIdx.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(CUDADIR)/LifSynapse_struct.cpp -o $(CUDADIR)/LifSynapse_struct.o

$(CUDADIR)/BGDriver_cuda.o: $(MAIN)/BGDriver.cpp $(COMMDIR)/Global.h $(COMMDIR)/Model.h $(COMMDIR)/AllNeurons.h $(COMMDIR)/AllSynapses.h $(COMMDIR)/Model.h $(COMMDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c $(MAIN)/BGDriver.cpp -o $(CUDADIR)/BGDriver_cuda.o


# Library
# ------------------------------------------------------------------------------

$(COMMDIR)/AllNeurons.o: $(COMMDIR)/AllNeurons.cpp $(COMMDIR)/AllNeurons.h $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/AllNeurons.cpp -o $(COMMDIR)/AllNeurons.o
	
$(COMMDIR)/AllSynapses.o: $(COMMDIR)/AllSynapses.cpp $(COMMDIR)/AllSynapses.h $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/AllSynapses.cpp -o $(COMMDIR)/AllSynapses.o

$(COMMDIR)/Global.o: $(COMMDIR)/Global.cpp $(COMMDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Global.cpp -o $(COMMDIR)/Global.o

$(COMMDIR)/HostSimulator.o: $(COMMDIR)/HostSimulator.cpp $(COMMDIR)/HostSimulator.h $(COMMDIR)/Simulator.h $(COMMDIR)/Global.h $(COMMDIR)/SimulationInfo.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/HostSimulator.cpp -o $(COMMDIR)/HostSimulator.o

$(COMMDIR)/LIFModel.o: $(COMMDIR)/LIFModel.cpp $(COMMDIR)/LIFModel.h $(COMMDIR)/Model.h $(COMMDIR)/ParseParamError.h $(COMMDIR)/Util.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/LIFModel.cpp -o $(COMMDIR)/LIFModel.o

$(COMMDIR)/Network.o: $(COMMDIR)/Network.cpp $(COMMDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Network.cpp -o $(COMMDIR)/Network.o

$(COMMDIR)/ParseParamError.o: $(COMMDIR)/ParseParamError.cpp $(COMMDIR)/ParseParamError.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/ParseParamError.cpp -o $(COMMDIR)/ParseParamError.o

$(COMMDIR)/Timer.o: $(COMMDIR)/Timer.cpp $(COMMDIR)/Timer.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Timer.cpp -o $(COMMDIR)/Timer.o

$(COMMDIR)/Util.o: $(COMMDIR)/Util.cpp $(COMMDIR)/Util.h
	$(CXX) $(CXXFLAGS) $(COMMDIR)/Util.cpp -o $(COMMDIR)/Util.o


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



