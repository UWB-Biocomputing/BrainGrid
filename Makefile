################################################################################
# Default Target
################################################################################
all: growth 

################################################################################
# Source Directories
################################################################################
MAIN = .
CUDADIR = $(MAIN)/cuda
LIBDIR = $(MAIN)/common
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
CXXFLAGS = -I$(LIBDIR) -I$(MATRIXDIR) -I$(PARAMDIR) -I$(RNGDIR) -I$(XMLDIR) -Wall -g -pg -c -DTIXML_USE_STL -DDEBUG_OUT -DSTORE_SPIKEHISTORY
CGPUFLAGS = -DUSE_GPU
LDFLAGS = -lstdc++ 
LGPUFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

################################################################################
# Objects
################################################################################

CUDAOBJS = \

LIBOBJS = $(LIBDIR)/AllNeurons.o \
			$(LIBDIR)/AllSynapses.o \
			$(LIBDIR)/Global.o \
			$(LIBDIR)/HostSimulator.o \
			$(LIBDIR)/LIFModel.o \
			$(LIBDIR)/Network.o \
			$(LIBDIR)/ParseParamError.o \
			$(LIBDIR)/Timer.o \
			$(LIBDIR)/Util.o
		
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

clean:
	rm -f $(MAIN)/*.o $(LIBDIR)/*.o $(MATRIXDIR)/*.o $(PARAMDIR)/*.o $(RNGDIR)/*.o $(XMLDIR)/*.o ./growth
	

################################################################################
# Build Source Files
################################################################################

# CUDA
# ------------------------------------------------------------------------------
#TODO: Fill in when new CUDA code is implemented.  See './old/Makefile' for reference on compiling CUDA code.


# Library
# ------------------------------------------------------------------------------

$(LIBDIR)/AllNeurons.o: $(LIBDIR)/AllNeurons.cpp $(LIBDIR)/AllNeurons.h $(LIBDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/AllNeurons.cpp -o $(LIBDIR)/AllNeurons.o
	
$(LIBDIR)/AllSynapses.o: $(LIBDIR)/AllSynapses.cpp $(LIBDIR)/AllSynapses.h $(LIBDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/AllSynapses.cpp -o $(LIBDIR)/AllSynapses.o

$(LIBDIR)/Global.o: $(LIBDIR)/Global.cpp $(LIBDIR)/Global.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/Global.cpp -o $(LIBDIR)/Global.o

$(LIBDIR)/HostSimulator.o: $(LIBDIR)/HostSimulator.cpp $(LIBDIR)/HostSimulator.h $(LIBDIR)/Simulator.h $(LIBDIR)/Global.h $(LIBDIR)/SimulationInfo.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/HostSimulator.cpp -o $(LIBDIR)/HostSimulator.o

$(LIBDIR)/LIFModel.o: $(LIBDIR)/LIFModel.cpp $(LIBDIR)/LIFModel.h $(LIBDIR)/Model.h $(LIBDIR)/ParseParamError.h $(LIBDIR)/Util.h $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/LIFModel.cpp -o $(LIBDIR)/LIFModel.o

$(LIBDIR)/Network.o: $(LIBDIR)/Network.cpp $(LIBDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/Network.cpp -o $(LIBDIR)/Network.o

$(LIBDIR)/ParseParamError.o: $(LIBDIR)/ParseParamError.cpp $(LIBDIR)/ParseParamError.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/ParseParamError.cpp -o $(LIBDIR)/ParseParamError.o

$(LIBDIR)/Timer.o: $(LIBDIR)/Timer.cpp $(LIBDIR)/Timer.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/Timer.cpp -o $(LIBDIR)/Timer.o

$(LIBDIR)/Util.o: $(LIBDIR)/Util.cpp $(LIBDIR)/Util.h
	$(CXX) $(CXXFLAGS) $(LIBDIR)/Util.cpp -o $(LIBDIR)/Util.o


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

$(RNGDIR)/Norm.o: $(RNGDIR)/Norm.cpp $(RNGDIR)/Norm.h $(RNGDIR)/MersenneTwister.h $(LIBDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(RNGDIR)/Norm.cpp -o $(RNGDIR)/Norm.o


# XML
# ------------------------------------------------------------------------------

$(XMLDIR)/tinyxml.o: $(XMLDIR)/tinyxml.cpp $(XMLDIR)/tinyxml.h $(XMLDIR)/tinystr.h $(LIBDIR)/BGTypes.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxml.cpp -o $(XMLDIR)/tinyxml.o

$(XMLDIR)/tinyxmlparser.o: $(XMLDIR)/tinyxmlparser.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlparser.cpp -o $(XMLDIR)/tinyxmlparser.o

$(XMLDIR)/tinyxmlerror.o: $(XMLDIR)/tinyxmlerror.cpp $(XMLDIR)/tinyxml.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinyxmlerror.cpp -o $(XMLDIR)/tinyxmlerror.o

$(XMLDIR)/tinystr.o: $(XMLDIR)/tinystr.cpp $(XMLDIR)/tinystr.h
	$(CXX) $(CXXFLAGS) $(XMLDIR)/tinystr.cpp -o $(XMLDIR)/tinystr.o

# Single Threaded
# ------------------------------------------------------------------------------

$(MAIN)/BGDriver.o: $(MAIN)/BGDriver.cpp $(LIBDIR)/Global.h $(LIBDIR)/Network.h
	$(CXX) $(CXXFLAGS) $(MAIN)/BGDriver.cpp -o $(MAIN)/BGDriver.o



