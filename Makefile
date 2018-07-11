################################################################################
# Default Target
# -----------------------------------------------------------------------------
# growth	 - single threaded
# growth_cuda	 - multithreaded
################################################################################
all: growth growth_cuda

################################################################################
# Conditional Flags
# -----------------------------------------------------------------------------
# CUSEHDF5:	 yes - use hdf5 file format 
#		 no  - use default xml 
# CPMETRICS: 	 yes - see performance of large function calls  
#		 no  - not showing performance results
################################################################################
CUSEHDF5 = yes
CPMETRICS = no

################################################################################
# Source Directories
################################################################################
MAIN = .
COREDIR = $(MAIN)/Core
CONNDIR = $(MAIN)/Connections
INPUTDIR = $(MAIN)/Inputs
LAYOUTDIR = $(MAIN)/Layouts
MATRIXDIR = $(MAIN)/Matrix
NEURONDIR = $(MAIN)/Neurons
PARAMDIR = $(MAIN)/paramcontainer
RECORDERDIR = $(MAIN)/Recorders
RNGDIR = $(MAIN)/RNG
SYNAPSEDIR = $(MAIN)/Synapses
XMLDIR = $(MAIN)/tinyxml
UTILDIR = $(MAIN)/Utils

# cuda
CUDALIBDIR = /usr/local/cuda/lib64

# hdf5
ifeq ($(CUSEHDF5), yes)
	H5INCDIR = /usr/include/hdf5/serial		   # include dir
	H5LIBDIR = /usr/lib/x86_64-linux-gnu/hdf5/serial/  # library dir
	#H5LIBDIR = /usr/lib/x86_64-linux-gnu/	 	   # another library dir
else
	H5INCDIR = .
endif

################################################################################
# Build tools
#
# for single-threaded version, use h5c++ instead of g++ to avoid linking bug
################################################################################
CXX = g++
ifeq ($(CUSEHDF5), yes)
	LD = h5c++	
else
	LD = g++
endif
LD_cuda = nvcc	# linking the compiled cuda with g++ can be troublesome 

################################################################################
# Flags
################################################################################
# BUGGY MAKE!! ---------------- our Make is buggy. Can't use ifeq in this case
# ----------------------------- ifdef and ifneq both works.
ifneq ($(CPMETRICS), no)
#ifdef CPMETRICS
	PMFLAGS = -DPERFORMANCE_METRICS
else
	PMFLAGS = 
endif
	
ifeq ($(CUSEHDF5), yes)
	LH5FLAGS =  -L$(H5LIBDIR) -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5 -lsz
	H5FLAGS = -DUSE_HDF5
else
	LH5FLAGS =
	H5FLAGS = 
endif

INCDIRS = -I$(CONNDIR) -I$(COREDIR) -I$(H5INCDIR) -I$(INPUTDIR) -I$(LAYOUTDIR) \
          -I$(MATRIXDIR) -I$(NEURONDIR) -I$(PARAMDIR) -I$(RECORDERDIR) \
          -I$(RNGDIR) -I$(SYNAPSEDIR) -I$(UTILDIR) -I$(XMLDIR)

CXXFLAGS = -O2 -std=c++98 -s -Wall -g -pg -c -DTIXML_USE_STL -DDEBUG_OUT $(INCDIRS) $(PMFLAGS) $(H5FLAGS) 
CGPUFLAGS = -DUSE_GPU $(PMFLAGS) $(H5FLAGS)
LDFLAGS = -lstdc++ 
LGPUFLAGS = -L$(CUDALIBDIR) -lcuda -lcudart
NVCCFLAGS =  -g -arch=sm_20 -rdc=true $(INCDIRS)

################################################################################
# Objects
################################################################################

CUDAOBJS =   \
		$(COREDIR)/GPUSpikingModel.o \
		$(NEURONDIR)/AllNeuronsDeviceFuncs_d.o \
		$(NEURONDIR)/AllNeurons_cuda.o \
		$(NEURONDIR)/AllSpikingNeurons_cuda.o \
		$(NEURONDIR)/AllSpikingNeurons_d.o \
		$(NEURONDIR)/AllIFNeurons_cuda.o \
		$(NEURONDIR)/AllIFNeurons_d.o \
		$(NEURONDIR)/AllLIFNeurons_cuda.o \
		$(NEURONDIR)/AllLIFNeurons_d.o \
		$(NEURONDIR)/AllIZHNeurons_cuda.o \
		$(NEURONDIR)/AllIZHNeurons_d.o \
		$(SYNAPSEDIR)/AllSynapsesDeviceFuncs_d.o \
		$(SYNAPSEDIR)/AllSynapses_cuda.o \
		$(SYNAPSEDIR)/AllSpikingSynapses_cuda.o \
		$(SYNAPSEDIR)/AllSpikingSynapses_d.o \
		$(SYNAPSEDIR)/AllDSSynapses_cuda.o \
		$(SYNAPSEDIR)/AllDSSynapses_d.o \
		$(SYNAPSEDIR)/AllSTDPSynapses_cuda.o \
		$(SYNAPSEDIR)/AllSTDPSynapses_d.o \
		$(SYNAPSEDIR)/AllDynamicSTDPSynapses_cuda.o \
		$(SYNAPSEDIR)/AllDynamicSTDPSynapses_d.o \
		$(CONNDIR)/Connections_cuda.o \
		$(CONNDIR)/ConnGrowth_cuda.o \
		$(CONNDIR)/ConnStatic_cuda.o \
		$(CONNDIR)/ConnGrowth_d.o \
		$(RNGDIR)/MersenneTwister_d.o \
		$(COREDIR)/BGDriver_cuda.o \
		$(INPUTDIR)/GpuSInputRegular.o \
		$(INPUTDIR)/GpuSInputPoisson.o \
		$(INPUTDIR)/FSInput_cuda.o \
		$(COREDIR)/FClassOfCategory_cuda.o \
		$(UTILDIR)/Global_cuda.o

ifeq ($(CUSEHDF5), yes)
LIBOBJS =   \
		$(COREDIR)/Simulator.o \
		$(COREDIR)/SimulationInfo.o \
		$(COREDIR)/Model.o \
		$(LAYOUTDIR)/Layout.o \
		$(LAYOUTDIR)/FixedLayout.o \
		$(LAYOUTDIR)/DynamicLayout.o \
		$(UTILDIR)/ParseParamError.o \
		$(UTILDIR)/Timer.o \
		$(UTILDIR)/Util.o \
		$(RECORDERDIR)/XmlRecorder.o \
		$(RECORDERDIR)/XmlGrowthRecorder.o \
		$(RECORDERDIR)/Hdf5Recorder.o \
		$(RECORDERDIR)/Hdf5GrowthRecorder.o 
else
LIBOBJS =   \
		$(COREDIR)/Simulator.o \
		$(COREDIR)/SimulationInfo.o \
		$(COREDIR)/Model.o \
		$(LAYOUTDIR)/Layout.o \
		$(LAYOUTDIR)/FixedLayout.o \
		$(LAYOUTDIR)/DynamicLayout.o \
		$(UTILDIR)/ParseParamError.o \
		$(UTILDIR)/Timer.o \
		$(UTILDIR)/Util.o \
		$(RECORDERDIR)/XmlRecorder.o \
		$(RECORDERDIR)/XmlGrowthRecorder.o
endif

MATRIXOBJS =	$(MATRIXDIR)/CompleteMatrix.o \
		$(MATRIXDIR)/Matrix.o \
		$(MATRIXDIR)/SparseMatrix.o \
		$(MATRIXDIR)/VectorMatrix.o 

PARAMOBJS =	$(PARAMDIR)/ParamContainer.o

RNGOBJS =	$(RNGDIR)/Norm.o \
		$(RNGDIR)/MersenneTwister.o

SINGLEOBJS =	$(COREDIR)/BGDriver.o  \
		$(COREDIR)/SingleThreadedSpikingModel.o \
		$(INPUTDIR)/FSInput.o \
		$(COREDIR)/FClassOfCategory.o \
		$(NEURONDIR)/AllNeurons.o \
		$(NEURONDIR)/AllSpikingNeurons.o \
		$(NEURONDIR)/AllIFNeurons.o \
		$(NEURONDIR)/AllLIFNeurons.o \
		$(NEURONDIR)/AllIZHNeurons.o \
		$(SYNAPSEDIR)/AllSynapses.o \
		$(SYNAPSEDIR)/AllSpikingSynapses.o \
		$(SYNAPSEDIR)/AllDSSynapses.o \
		$(SYNAPSEDIR)/AllSTDPSynapses.o \
		$(SYNAPSEDIR)/AllDynamicSTDPSynapses.o \
		$(CONNDIR)/Connections.o \
		$(CONNDIR)/ConnGrowth.o \
		$(CONNDIR)/ConnStatic.o \
		$(UTILDIR)/Global.o 

XMLOBJS =	$(XMLDIR)/tinyxml.o \
		$(XMLDIR)/tinyxmlparser.o \
		$(XMLDIR)/tinyxmlerror.o \
		$(XMLDIR)/tinystr.o

INPUTOBJS =	$(INPUTDIR)/HostSInputRegular.o \
		$(INPUTDIR)/SInputRegular.o \
		$(INPUTDIR)/HostSInputPoisson.o \
		$(INPUTDIR)/SInputPoisson.o 

################################################################################
# Targets
################################################################################
# make growth (single threaded version)
# ------------------------------------------------------------------------------
growth: $(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) $(INPUTOBJS)
	$(LD) -o growth -g $(LDFLAGS) $(LH5FLAGS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(SINGLEOBJS) $(XMLOBJS) $(INPUTOBJS) $(LIBOBJS) 

# make growth_cuda (multi-threaded version)
# ------------------------------------------------------------------------------
growth_cuda: 	$(LIBOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(XMLOBJS) $(OTHEROBJS) $(CUDAOBJS) $(INPUTOBJS)
		$(LD_cuda) -o growth_cuda $(NVCCFLAGS) $(LDFLAGS) $(LH5FLAGS) $(LGPUFLAGS) $(LIBOBJS) $(CUDAOBJS) $(MATRIXOBJS) $(PARAMOBJS) $(RNGOBJS) $(XMLOBJS) $(OTHEROBJS) $(INPUTOBJS)

# make clean
# ------------------------------------------------------------------------------
clean:
	rm -f $(COREDIR)/*.o $(CONNDIR)/*.o $(INPUTDIR)/*.o $(LAYOUTDIR)/*.o $(MATRIXDIR)/*.o $(NEURONDIR)/*.o $(PARAMDIR)/*.o $(RECORDERDIR)/*.o $(RNGDIR)/*.o $(SYNAPSEDIR)/*.o $(XMLDIR)/*.o $(UTILDIR)/*.o ./growth ./growth_cuda

################################################################################
# Build Source Files
################################################################################

# CUDA
# ------------------------------------------------------------------------------

$(RNGDIR)/MersenneTwister_d.o: $(RNGDIR)/MersenneTwister_d.cu $(UTILDIR)/Global.h $(RNGDIR)/MersenneTwister_d.h
	nvcc -c $(NVCCFLAGS) $(RNGDIR)/MersenneTwister_d.cu $(CGPUFLAGS) -o $(RNGDIR)/MersenneTwister_d.o

$(COREDIR)/GPUSpikingModel.o: $(COREDIR)/GPUSpikingModel.cu $(UTILDIR)/Global.h $(COREDIR)/GPUSpikingModel.h $(NEURONDIR)/AllIFNeurons.h $(SYNAPSEDIR)/AllSynapses.h $(COREDIR)/IModel.h  
	nvcc -c $(NVCCFLAGS) $(COREDIR)/GPUSpikingModel.cu $(CGPUFLAGS) -o $(COREDIR)/GPUSpikingModel.o

$(NEURONDIR)/AllNeuronsDeviceFuncs_d.o: $(NEURONDIR)/AllNeuronsDeviceFuncs_d.cu $(UTILDIR)/Global.h $(NEURONDIR)/AllNeuronsDeviceFuncs.h
	nvcc -c $(NVCCFLAGS) $(NEURONDIR)/AllNeuronsDeviceFuncs_d.cu $(CGPUFLAGS) -o $(NEURONDIR)/AllNeuronsDeviceFuncs_d.o

$(NEURONDIR)/AllSpikingNeurons_d.o: $(NEURONDIR)/AllSpikingNeurons_d.cu $(UTILDIR)/Global.h $(NEURONDIR)/AllSpikingNeurons.h
	nvcc -c $(NVCCFLAGS) $(NEURONDIR)/AllSpikingNeurons_d.cu $(CGPUFLAGS) -o $(NEURONDIR)/AllSpikingNeurons_d.o

$(NEURONDIR)/AllIFNeurons_d.o: $(NEURONDIR)/AllIFNeurons_d.cu $(UTILDIR)/Global.h $(NEURONDIR)/AllIFNeurons.h
	nvcc -c $(NVCCFLAGS) $(NEURONDIR)/AllIFNeurons_d.cu $(CGPUFLAGS) -o $(NEURONDIR)/AllIFNeurons_d.o

$(NEURONDIR)/AllLIFNeurons_d.o: $(NEURONDIR)/AllLIFNeurons_d.cu $(UTILDIR)/Global.h $(NEURONDIR)/AllLIFNeurons.h
	nvcc -c $(NVCCFLAGS) $(NEURONDIR)/AllLIFNeurons_d.cu $(CGPUFLAGS) -o $(NEURONDIR)/AllLIFNeurons_d.o

$(NEURONDIR)/AllIZHNeurons_d.o: $(NEURONDIR)/AllIZHNeurons_d.cu $(UTILDIR)/Global.h $(NEURONDIR)/AllIZHNeurons.h
	nvcc -c $(NVCCFLAGS) $(NEURONDIR)/AllIZHNeurons_d.cu $(CGPUFLAGS) -o $(NEURONDIR)/AllIZHNeurons_d.o

$(SYNAPSEDIR)/AllSynapsesDeviceFuncs_d.o: $(SYNAPSEDIR)/AllSynapsesDeviceFuncs_d.cu $(UTILDIR)/Global.h $(SYNAPSEDIR)/AllSynapsesDeviceFuncs.h
	nvcc -c $(NVCCFLAGS) $(SYNAPSEDIR)/AllSynapsesDeviceFuncs_d.cu $(CGPUFLAGS) -o $(SYNAPSEDIR)/AllSynapsesDeviceFuncs_d.o

$(SYNAPSEDIR)/AllSpikingSynapses_d.o: $(SYNAPSEDIR)/AllSpikingSynapses_d.cu $(UTILDIR)/Global.h $(SYNAPSEDIR)/AllSpikingSynapses.h
	nvcc -c $(NVCCFLAGS) $(SYNAPSEDIR)/AllSpikingSynapses_d.cu $(CGPUFLAGS) -o $(SYNAPSEDIR)/AllSpikingSynapses_d.o

$(SYNAPSEDIR)/AllDSSynapses_d.o: $(SYNAPSEDIR)/AllDSSynapses_d.cu $(UTILDIR)/Global.h $(SYNAPSEDIR)/AllDSSynapses.h
	nvcc -c $(NVCCFLAGS) $(SYNAPSEDIR)/AllDSSynapses_d.cu $(CGPUFLAGS) -o $(SYNAPSEDIR)/AllDSSynapses_d.o

$(SYNAPSEDIR)/AllSTDPSynapses_d.o: $(SYNAPSEDIR)/AllSTDPSynapses_d.cu $(UTILDIR)/Global.h $(SYNAPSEDIR)/AllSTDPSynapses.h
	nvcc -c $(NVCCFLAGS) $(SYNAPSEDIR)/AllSTDPSynapses_d.cu $(CGPUFLAGS) -o $(SYNAPSEDIR)/AllSTDPSynapses_d.o

$(SYNAPSEDIR)/AllDynamicSTDPSynapses_d.o: $(SYNAPSEDIR)/AllDynamicSTDPSynapses_d.cu $(UTILDIR)/Global.h $(SYNAPSEDIR)/AllDynamicSTDPSynapses.h
	nvcc -c $(NVCCFLAGS) $(SYNAPSEDIR)/AllDynamicSTDPSynapses_d.cu $(CGPUFLAGS) -o $(SYNAPSEDIR)/AllDynamicSTDPSynapses_d.o

$(CONNDIR)/ConnGrowth_d.o: $(CONNDIR)/ConnGrowth_d.cu $(UTILDIR)/Global.h $(CONNDIR)/ConnGrowth.h
	nvcc -c $(NVCCFLAGS) $(CONNDIR)/ConnGrowth_d.cu $(CGPUFLAGS) -o $(CONNDIR)/ConnGrowth_d.o

$(COREDIR)/BGDriver_cuda.o: $(COREDIR)/BGDriver.cpp $(UTILDIR)/Global.h $(COREDIR)/IModel.h $(NEURONDIR)/AllIFNeurons.h $(SYNAPSEDIR)/AllSynapses.h 
	$(CXX) -c $(CXXFLAGS) $(CGPUFLAGS) $(COREDIR)/BGDriver.cpp -o $(COREDIR)/BGDriver_cuda.o

$(NEURONDIR)/AllNeurons_cuda.o: $(NEURONDIR)/AllNeurons.cpp $(NEURONDIR)/AllNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(NEURONDIR)/AllNeurons.cpp -o $(NEURONDIR)/AllNeurons_cuda.o

$(NEURONDIR)/AllSpikingNeurons_cuda.o: $(NEURONDIR)/AllSpikingNeurons.cpp $(NEURONDIR)/AllSpikingNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(NEURONDIR)/AllSpikingNeurons.cpp -o $(NEURONDIR)/AllSpikingNeurons_cuda.o

$(NEURONDIR)/AllIFNeurons_cuda.o: $(NEURONDIR)/AllIFNeurons.cpp $(NEURONDIR)/AllIFNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(NEURONDIR)/AllIFNeurons.cpp -o $(NEURONDIR)/AllIFNeurons_cuda.o

$(NEURONDIR)/AllLIFNeurons_cuda.o: $(NEURONDIR)/AllLIFNeurons.cpp $(NEURONDIR)/AllLIFNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(NEURONDIR)/AllLIFNeurons.cpp -o $(NEURONDIR)/AllLIFNeurons_cuda.o

$(NEURONDIR)/AllIZHNeurons_cuda.o: $(NEURONDIR)/AllIZHNeurons.cpp $(NEURONDIR)/AllIZHNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(NEURONDIR)/AllIZHNeurons.cpp -o $(NEURONDIR)/AllIZHNeurons_cuda.o

$(SYNAPSEDIR)/AllSynapses_cuda.o: $(SYNAPSEDIR)/AllSynapses.cpp $(SYNAPSEDIR)/AllSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(SYNAPSEDIR)/AllSynapses.cpp -o $(SYNAPSEDIR)/AllSynapses_cuda.o

$(SYNAPSEDIR)/AllSpikingSynapses_cuda.o: $(SYNAPSEDIR)/AllSpikingSynapses.cpp $(SYNAPSEDIR)/AllSpikingSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(SYNAPSEDIR)/AllSpikingSynapses.cpp -o $(SYNAPSEDIR)/AllSpikingSynapses_cuda.o

$(SYNAPSEDIR)/AllDSSynapses_cuda.o: $(SYNAPSEDIR)/AllDSSynapses.cpp $(SYNAPSEDIR)/AllDSSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(SYNAPSEDIR)/AllDSSynapses.cpp -o $(SYNAPSEDIR)/AllDSSynapses_cuda.o

$(SYNAPSEDIR)/AllSTDPSynapses_cuda.o: $(SYNAPSEDIR)/AllSTDPSynapses.cpp $(SYNAPSEDIR)/AllSTDPSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(SYNAPSEDIR)/AllSTDPSynapses.cpp -o $(SYNAPSEDIR)/AllSTDPSynapses_cuda.o

$(SYNAPSEDIR)/AllDynamicSTDPSynapses_cuda.o: $(SYNAPSEDIR)/AllDynamicSTDPSynapses.cpp $(SYNAPSEDIR)/AllDynamicSTDPSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(SYNAPSEDIR)/AllDynamicSTDPSynapses.cpp -o $(SYNAPSEDIR)/AllDynamicSTDPSynapses_cuda.o

$(CONNDIR)/Connections_cuda.o: $(CONNDIR)/Connections.cpp $(CONNDIR)/Connections.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(CONNDIR)/Connections.cpp -o $(CONNDIR)/Connections_cuda.o

$(CONNDIR)/ConnGrowth_cuda.o: $(CONNDIR)/ConnGrowth.cpp $(CONNDIR)/ConnGrowth.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(CONNDIR)/ConnGrowth.cpp -o $(CONNDIR)/ConnGrowth_cuda.o

$(CONNDIR)/ConnStatic_cuda.o: $(CONNDIR)/ConnStatic.cpp $(CONNDIR)/ConnStatic.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(CONNDIR)/ConnStatic.cpp -o $(CONNDIR)/ConnStatic_cuda.o

$(UTILDIR)/Global_cuda.o: $(UTILDIR)/Global.cpp $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(UTILDIR)/Global.cpp -o $(UTILDIR)/Global_cuda.o

$(COREDIR)/FClassOfCategory_cuda.o: $(COREDIR)/FClassOfCategory.cpp $(COREDIR)/FClassOfCategory.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(COREDIR)/FClassOfCategory.cpp -o $(COREDIR)/FClassOfCategory_cuda.o


# Library
# ------------------------------------------------------------------------------

$(NEURONDIR)/AllNeurons.o: $(NEURONDIR)/AllNeurons.cpp $(NEURONDIR)/AllNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(NEURONDIR)/AllNeurons.cpp -o $(NEURONDIR)/AllNeurons.o

$(NEURONDIR)/AllSpikingNeurons.o: $(NEURONDIR)/AllSpikingNeurons.cpp $(NEURONDIR)/AllSpikingNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(NEURONDIR)/AllSpikingNeurons.cpp -o $(NEURONDIR)/AllSpikingNeurons.o

$(NEURONDIR)/AllIFNeurons.o: $(NEURONDIR)/AllIFNeurons.cpp $(NEURONDIR)/AllIFNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(NEURONDIR)/AllIFNeurons.cpp -o $(NEURONDIR)/AllIFNeurons.o

$(NEURONDIR)/AllLIFNeurons.o: $(NEURONDIR)/AllLIFNeurons.cpp $(NEURONDIR)/AllLIFNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(NEURONDIR)/AllLIFNeurons.cpp -o $(NEURONDIR)/AllLIFNeurons.o

$(NEURONDIR)/AllIZHNeurons.o: $(NEURONDIR)/AllIZHNeurons.cpp $(NEURONDIR)/AllIZHNeurons.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(NEURONDIR)/AllIZHNeurons.cpp -o $(NEURONDIR)/AllIZHNeurons.o

$(SYNAPSEDIR)/AllSynapses.o: $(SYNAPSEDIR)/AllSynapses.cpp $(SYNAPSEDIR)/AllSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(SYNAPSEDIR)/AllSynapses.cpp -o $(SYNAPSEDIR)/AllSynapses.o

$(SYNAPSEDIR)/AllSpikingSynapses.o: $(SYNAPSEDIR)/AllSpikingSynapses.cpp $(SYNAPSEDIR)/AllSpikingSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(SYNAPSEDIR)/AllSpikingSynapses.cpp -o $(SYNAPSEDIR)/AllSpikingSynapses.o

$(SYNAPSEDIR)/AllDSSynapses.o: $(SYNAPSEDIR)/AllDSSynapses.cpp $(SYNAPSEDIR)/AllDSSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(SYNAPSEDIR)/AllDSSynapses.cpp -o $(SYNAPSEDIR)/AllDSSynapses.o

$(SYNAPSEDIR)/AllSTDPSynapses.o: $(SYNAPSEDIR)/AllSTDPSynapses.cpp $(SYNAPSEDIR)/AllSTDPSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(SYNAPSEDIR)/AllSTDPSynapses.cpp -o $(SYNAPSEDIR)/AllSTDPSynapses.o

$(SYNAPSEDIR)/AllDynamicSTDPSynapses.o: $(SYNAPSEDIR)/AllDynamicSTDPSynapses.cpp $(SYNAPSEDIR)/AllDynamicSTDPSynapses.h $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(SYNAPSEDIR)/AllDynamicSTDPSynapses.cpp -o $(SYNAPSEDIR)/AllDynamicSTDPSynapses.o

$(UTILDIR)/Global.o: $(UTILDIR)/Global.cpp $(UTILDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(UTILDIR)/Global.cpp -o $(UTILDIR)/Global.o

$(COREDIR)/Simulator.o: $(COREDIR)/Simulator.cpp $(COREDIR)/Simulator.h $(UTILDIR)/Global.h $(COREDIR)/SimulationInfo.h
	$(CXX) $(CXXFLAGS) $(COREDIR)/Simulator.cpp -o $(COREDIR)/Simulator.o

$(COREDIR)/SimulationInfo.o: $(COREDIR)/SimulationInfo.cpp $(COREDIR)/SimulationInfo.h $(UTILDIR)/Global.h 
	$(CXX) $(CXXFLAGS) $(COREDIR)/SimulationInfo.cpp -o $(COREDIR)/SimulationInfo.o

$(COREDIR)/Model.o: $(COREDIR)/Model.cpp $(COREDIR)/Model.h $(COREDIR)/IModel.h $(UTILDIR)/ParseParamError.h $(UTILDIR)/Util.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(COREDIR)/Model.cpp -o $(COREDIR)/Model.o

$(CONNDIR)/Connections.o: $(CONNDIR)/Connections.cpp $(CONNDIR)/Connections.h 
	$(CXX) $(CXXFLAGS) $(CONNDIR)/Connections.cpp -o $(CONNDIR)/Connections.o

$(CONNDIR)/ConnStatic.o: $(CONNDIR)/ConnStatic.cpp $(CONNDIR)/ConnStatic.h 
	$(CXX) $(CXXFLAGS) $(CONNDIR)/ConnStatic.cpp -o $(CONNDIR)/ConnStatic.o

$(CONNDIR)/ConnGrowth.o: $(CONNDIR)/ConnGrowth.cpp $(CONNDIR)/ConnGrowth.h 
	$(CXX) $(CXXFLAGS) $(CONNDIR)/ConnGrowth.cpp -o $(CONNDIR)/ConnGrowth.o

$(LAYOUTDIR)/Layout.o: $(LAYOUTDIR)/Layout.cpp $(LAYOUTDIR)/Layout.h 
	$(CXX) $(CXXFLAGS) $(LAYOUTDIR)/Layout.cpp -o $(LAYOUTDIR)/Layout.o

$(LAYOUTDIR)/FixedLayout.o: $(LAYOUTDIR)/FixedLayout.cpp $(LAYOUTDIR)/FixedLayout.h 
	$(CXX) $(CXXFLAGS) $(LAYOUTDIR)/FixedLayout.cpp -o $(LAYOUTDIR)/FixedLayout.o

$(LAYOUTDIR)/DynamicLayout.o: $(LAYOUTDIR)/DynamicLayout.cpp $(LAYOUTDIR)/DynamicLayout.h 
	$(CXX) $(CXXFLAGS) $(LAYOUTDIR)/DynamicLayout.cpp -o $(LAYOUTDIR)/DynamicLayout.o

$(COREDIR)/SingleThreadedSpikingModel.o: $(COREDIR)/SingleThreadedSpikingModel.cpp $(COREDIR)/SingleThreadedSpikingModel.h $(COREDIR)/Model.h 
	$(CXX) $(CXXFLAGS) $(COREDIR)/SingleThreadedSpikingModel.cpp -o $(COREDIR)/SingleThreadedSpikingModel.o

$(UTILDIR)/ParseParamError.o: $(UTILDIR)/ParseParamError.cpp $(UTILDIR)/ParseParamError.h
	$(CXX) $(CXXFLAGS) $(UTILDIR)/ParseParamError.cpp -o $(UTILDIR)/ParseParamError.o

$(UTILDIR)/Timer.o: $(UTILDIR)/Timer.cpp $(UTILDIR)/Timer.h
	$(CXX) $(CXXFLAGS) $(UTILDIR)/Timer.cpp -o $(UTILDIR)/Timer.o

$(UTILDIR)/Util.o: $(UTILDIR)/Util.cpp $(UTILDIR)/Util.h
	$(CXX) $(CXXFLAGS) $(UTILDIR)/Util.cpp -o $(UTILDIR)/Util.o

$(RECORDERDIR)/XmlRecorder.o: $(RECORDERDIR)/XmlRecorder.cpp $(RECORDERDIR)/XmlRecorder.h $(RECORDERDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(RECORDERDIR)/XmlRecorder.cpp -o $(RECORDERDIR)/XmlRecorder.o

$(RECORDERDIR)/XmlGrowthRecorder.o: $(RECORDERDIR)/XmlGrowthRecorder.cpp $(RECORDERDIR)/XmlGrowthRecorder.h $(RECORDERDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(RECORDERDIR)/XmlGrowthRecorder.cpp -o $(RECORDERDIR)/XmlGrowthRecorder.o

ifeq ($(CUSEHDF5), yes)
$(RECORDERDIR)/Hdf5GrowthRecorder.o: $(RECORDERDIR)/Hdf5GrowthRecorder.cpp $(RECORDERDIR)/Hdf5GrowthRecorder.h $(RECORDERDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(RECORDERDIR)/Hdf5GrowthRecorder.cpp -o $(RECORDERDIR)/Hdf5GrowthRecorder.o


$(RECORDERDIR)/Hdf5Recorder.o: $(RECORDERDIR)/Hdf5Recorder.cpp $(RECORDERDIR)/Hdf5Recorder.h $(RECORDERDIR)/IRecorder.h
	$(CXX) $(CXXFLAGS) $(RECORDERDIR)/Hdf5Recorder.cpp -o $(RECORDERDIR)/Hdf5Recorder.o
endif

$(COREDIR)/FClassOfCategory.o: $(COREDIR)/FClassOfCategory.cpp $(COREDIR)/FClassOfCategory.h
	$(CXX) $(CXXFLAGS) $(COREDIR)/FClassOfCategory.cpp -o $(COREDIR)/FClassOfCategory.o


# Matrix
# ------------------------------------------------------------------------------

$(MATRIXDIR)/CompleteMatrix.o: $(MATRIXDIR)/CompleteMatrix.cpp  $(MATRIXDIR)/CompleteMatrix.h $(MATRIXDIR)/MatrixExceptions.h $(MATRIXDIR)/Matrix.h $(MATRIXDIR)/VectorMatrix.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/CompleteMatrix.cpp -o $(MATRIXDIR)/CompleteMatrix.o

$(MATRIXDIR)/Matrix.o: $(MATRIXDIR)/Matrix.cpp $(MATRIXDIR)/Matrix.h  $(MATRIXDIR)/MatrixExceptions.h  $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/Matrix.cpp -o $(MATRIXDIR)/Matrix.o

$(MATRIXDIR)/SparseMatrix.o: $(MATRIXDIR)/SparseMatrix.cpp $(MATRIXDIR)/SparseMatrix.h  $(MATRIXDIR)/MatrixExceptions.h $(MATRIXDIR)/Matrix.h $(MATRIXDIR)/VectorMatrix.h
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/SparseMatrix.cpp -o $(MATRIXDIR)/SparseMatrix.o

$(MATRIXDIR)/VectorMatrix.o: $(MATRIXDIR)/VectorMatrix.cpp $(MATRIXDIR)/VectorMatrix.h $(MATRIXDIR)/CompleteMatrix.h $(MATRIXDIR)/SparseMatrix.h $(MATRIXDIR)/
	$(CXX) $(CXXFLAGS) $(MATRIXDIR)/VectorMatrix.cpp -o $(MATRIXDIR)/VectorMatrix.o


# ParamContainer
# ------------------------------------------------------------------------------

$(PARAMDIR)/ParamContainer.o: $(PARAMDIR)/ParamContainer.cpp $(PARAMDIR)/ParamContainer.h
	$(CXX) $(CXXFLAGS) $(PARAMDIR)/ParamContainer.cpp -o $(PARAMDIR)/ParamContainer.o

# RNG
# ------------------------------------------------------------------------------

$(RNGDIR)/Norm.o: $(RNGDIR)/Norm.cpp $(RNGDIR)/Norm.h $(RNGDIR)/MersenneTwister.h $(UTILDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(RNGDIR)/Norm.cpp -o $(RNGDIR)/Norm.o

$(RNGDIR)/MersenneTwister.o: $(RNGDIR)/MersenneTwister.cpp $(RNGDIR)/MersenneTwister.h $(UTILDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(RNGDIR)/MersenneTwister.cpp -o $(RNGDIR)/MersenneTwister.o


# XML
# ------------------------------------------------------------------------------

$(XMLDIR)/tinyxml.o: $(XMLDIR)/tinyxml.cpp $(XMLDIR)/tinyxml.h $(XMLDIR)/tinystr.h $(UTILDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxml.cpp -o $(XMLDIR)/tinyxml.o

$(XMLDIR)/tinyxmlparser.o: $(XMLDIR)/tinyxmlparser.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlparser.cpp -o $(XMLDIR)/tinyxmlparser.o

$(XMLDIR)/tinyxmlerror.o: $(XMLDIR)/tinyxmlerror.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlerror.cpp -o $(XMLDIR)/tinyxmlerror.o

$(XMLDIR)/tinystr.o: $(XMLDIR)/tinystr.cpp $(XMLDIR)/tinystr.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinystr.cpp -o $(XMLDIR)/tinystr.o

# Input
# ------------------------------------------------------------------------------
$(INPUTDIR)/FSInput.o: $(INPUTDIR)/FSInput.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/FSInput.h $(INPUTDIR)/HostSInputRegular.h $(INPUTDIR)/GpuSInputRegular.h $(INPUTDIR)/HostSInputPoisson.h $(INPUTDIR)/GpuSInputPoisson.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(INPUTDIR)/FSInput.cpp -o $(INPUTDIR)/FSInput.o

$(INPUTDIR)/FSInput_cuda.o: $(INPUTDIR)/FSInput.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/FSInput.h $(INPUTDIR)/HostSInputRegular.h $(INPUTDIR)/GpuSInputRegular.h $(INPUTDIR)/HostSInputPoisson.h $(INPUTDIR)/GpuSInputPoisson.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) $(INPUTDIR)/FSInput.cpp -o $(INPUTDIR)/FSInput_cuda.o

$(INPUTDIR)/SInputRegular.o: $(INPUTDIR)/SInputRegular.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/SInputRegular.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(INPUTDIR)/SInputRegular.cpp -o $(INPUTDIR)/SInputRegular.o

$(INPUTDIR)/SInputPoisson.o: $(INPUTDIR)/SInputPoisson.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/SInputPoisson.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(INPUTDIR)/SInputPoisson.cpp -o $(INPUTDIR)/SInputPoisson.o

$(INPUTDIR)/HostSInputRegular.o: $(INPUTDIR)/HostSInputRegular.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/HostSInputRegular.h
	$(CXX) $(CXXFLAGS) $(INPUTDIR)/HostSInputRegular.cpp -o $(INPUTDIR)/HostSInputRegular.o

$(INPUTDIR)/HostSInputPoisson.o: $(INPUTDIR)/HostSInputPoisson.cpp $(INPUTDIR)/ISInput.h $(INPUTDIR)/HostSInputPoisson.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(INPUTDIR)/HostSInputPoisson.cpp -o $(INPUTDIR)/HostSInputPoisson.o

$(INPUTDIR)/GpuSInputRegular.o: $(INPUTDIR)/GpuSInputRegular.cu $(INPUTDIR)/ISInput.h $(INPUTDIR)/GpuSInputRegular.h
	nvcc -c $(NVCCFLAGS) $(INPUTDIR)/GpuSInputRegular.cu $(CGPUFLAGS) -o $(INPUTDIR)/GpuSInputRegular.o

$(INPUTDIR)/GpuSInputPoisson.o: $(INPUTDIR)/GpuSInputPoisson.cu $(INPUTDIR)/ISInput.h $(INPUTDIR)/GpuSInputPoisson.h
	nvcc -c $(NVCCFLAGS) $(INPUTDIR)/GpuSInputPoisson.cu $(CGPUFLAGS) -o $(INPUTDIR)/GpuSInputPoisson.o

# Single Threaded
# ------------------------------------------------------------------------------

$(COREDIR)/BGDriver.o: $(COREDIR)/BGDriver.cpp $(UTILDIR)/Global.h 
	$(CXX) $(CXXFLAGS) $(COREDIR)/BGDriver.cpp -o $(COREDIR)/BGDriver.o

