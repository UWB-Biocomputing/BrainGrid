all: growth growth_omp growth_gpu

#
# Source directories
#
INCDIR = include
MATRIXDIR = Matrix
XMLDIR = tinyxml
PCDIR = paramcontainer
SVDIR = SourceVersions
RNGDIR = RNG
UTILDIR = Utils
MTWIST = MersenneTwister
EXECS = all

#
# Build tools
#
CXX = g++
LD = g++
OPT = g++ 

#
# Flags
#
CXXFLAGS = -I$(INCDIR) -I$(UTILDIR) -I$(MATRIXDIR) -I$(XMLDIR) -I$(PCDIR) -I$(SVDIR) -I$(RNGDIR) -DTIXML_USE_STL -DCDEBUG -Wall -DSTORE_SPIKEHISTORY
#CXXFLAGS = -I$(INCDIR) -I$(UTILDIR) -I$(MATRIXDIR) -I$(XMLDIR) -I$(PCDIR) -I$(SVDIR) -I$(RNGDIR) -DTIXML_USE_STL -DCDEBUG -Wall
#CXXFLAGS = -I$(INCDIR) -I$(UTILDIR) -I$(MATRIXDIR) -I$(XMLDIR) -I$(PCDIR) -I$(SVDIR) -I$(RNGDIR) -DTIXML_USE_STL -DCDEBUG -Wall -DPERFORMANCE_METRICS
#CXXFLAGS = -I$(INCDIR) -I$(UTILDIR) -I$(MATRIXDIR) -I$(XMLDIR) -I$(PCDIR) -I$(SVDIR) -I$(RNGDIR) -DTIXML_USE_STL -DCDEBUG -Wall -g -pg
COMPFLAGS = -DUSE_OMP -fopenmp
CGPUFLAGS = -DUSE_GPU
LDFLAGS = -lstdc++ 
#LDFLAGS = -g -pg -lstdc++ 
LOMPFLAGS = -fopenmp 
LGPUFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

#
# here are all the objects
#
MATRIXOBJS = $(MATRIXDIR)/Matrix.o $(MATRIXDIR)/VectorMatrix.o \
             $(MATRIXDIR)/CompleteMatrix.o $(MATRIXDIR)/SparseMatrix.o \
             $(MATRIXDIR)/MatrixFactory.o  

XMLOBJS = $(XMLDIR)/tinyxml.o $(XMLDIR)/tinyxmlparser.o $(XMLDIR)/tinyxmlerror.o $(XMLDIR)/tinystr.o

OTHEROBJS = $(SVDIR)/SourceVersions.o $(RNGDIR)/norm.o $(PCDIR)/ParamContainer.o $(UTILDIR)/Timer.o

GPUOBJS = GpuSim.o \
       HostSim.o \
       DynamicSpikingSynapse_struct.o \
       LifNeuron_struct.o \
       DynamicSpikingSynapse.o \
       LifNeuron.o \
       Network_gpu.o \
       GpuSim_struct.o \
       BGDriver_gpu.o \
       global_gpu.o \
       MersenneTwister_kernel.o

SINGLEOBJS = HostSim.o \
       SingleThreadedSim.o \
       DynamicSpikingSynapse.o \
       Network.o \
       BGDriver.o \
       global.o \
       LifNeuron.o 

MULTIOBJS = HostSim.o \
       MultiThreadedSim.o \
       DynamicSpikingSynapse_omp.o \
       Network_omp.o \
       BGDriver.o \
       global.o \
       LifNeuron_omp.o 

OBJS =  \
       

#
# how to make the target
#
growth:$(OBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS) $(SINGLEOBJS)
	$(LD) -o growth -g $(LDFLAGS) $(OBJS) $(SINGLEOBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS)

growth_omp:$(OBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS) $(MULTIOBJS)
	$(LD) -o growth_omp -g $(LDFLAGS) $(LOMPFLAGS) $(OBJS) $(MULTIOBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS)

growth_gpu:$(OBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS) $(GPUOBJS)
	nvcc -o growth_gpu -g -G $(LDFLAGS) $(LGPUFLAGS) $(OBJS) $(GPUOBJS) $(MATRIXOBJS) $(XMLOBJS) $(OTHEROBJS)

#
# some default targets
#
clean:
	rm -f *.o growth growth_omp growth_gpu $(INCDIR)/*.o $(MATRIXDIR)/*.o $(XMLDIR)/*.o $(PCDIR)/*.o $(SVDIR)/*.o $(RNGDIR)/*.o $(UTILDIR)/*.o 

paramcontainer/ParamContainer.o: paramcontainer/ParamContainer.h paramcontainer/ParamContainer.cpp
    
BGDriver.o: BGDriver.cpp global.h DynamicSpikingSynapse.h LifNeuron.h Network.h

DynamicSpikingSynapse.o: DynamicSpikingSynapse.cpp DynamicSpikingSynapse.h 

DynamicSpikingSynapse_omp.o: DynamicSpikingSynapse.cpp DynamicSpikingSynapse.h 
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c DynamicSpikingSynapse.cpp -o DynamicSpikingSynapse_omp.o

DynamicSpikingSynapse_struct.o: DynamicSpikingSynapse_struct.cpp DynamicSpikingSynapse_struct.h

global.o: global.h 

global_gpu.o: global.h 
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c global.cpp -o global_gpu.o

GpuSim.o: GpuSim.cpp GpuSim.h global.h LifNeuron_struct.h

GpuSim_struct.o: GpuSim_struct.cu global.h DynamicSpikingSynapse_struct.h LifNeuron_struct.h LifNeuron_struct_d.cu DynamicSpikingSynapse_struct_d.cu
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(XMLDIR) -I$(MATRIXDIR) -I$(RNGDIR)
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(XMLDIR) -I$(MATRIXDIR) -I$(RNGDIR) -DPERFORMANCE_METRICS
#	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(XMLDIR) -I$(MATRIXDIR) -I$(RNGDIR)
	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(XMLDIR) -I$(MATRIXDIR) -I$(RNGDIR) -DSTORE_SPIKEHISTORY

LifNeuron.o: LifNeuron.cpp LifNeuron.h 

LifNeuron_omp.o: LifNeuron.cpp LifNeuron.h 
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c LifNeuron.cpp -o LifNeuron_omp.o

LifNeuron_struct.o: LifNeuron_struct.cpp LifNeuron_struct.h

MersenneTwister_kernel.o: $(MTWIST)/MersenneTwister_kernel.cu $(MTWIST)/MersenneTwister.h
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v $(MTWIST)/MersenneTwister_kernel.cu -I$(MTWIST)
	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v $(MTWIST)/MersenneTwister_kernel.cu -I$(MTWIST)

MultiThreadedSim.o: MultiThreadedSim.cpp MultiThreadedSim.h
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c MultiThreadedSim.cpp 

Network.o: Network.cpp Network.h global.h

Network_omp.o: Network.cpp Network.h global.h
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c Network.cpp -o Network_omp.o

Network_gpu.o: Network.cpp Network.h global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c Network.cpp -o Network_gpu.o

BGDriver.o: BGDriver.cpp global.h DynamicSpikingSynapse.h LifNeuron.h Network.h

BGDriver_gpu.o: BGDriver.cpp global.h DynamicSpikingSynapse.h LifNeuron.h Network.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c BGDriver.cpp -o BGDriver_gpu.o

HostSim.o: HostSim.cpp HostSim.h ISimulation.h

SingleThreadedSim.o: SingleThreadedSim.cpp SingleThreadedSim.h

Utils/Timer.o: Utils/Timer.cpp Utils/Timer.h

RNG/norm.o: $(RNGDIR)/norm.cpp $(RNGDIR)/norm.h $(RNGDIR)/MersenneTwister.cpp $(RNGDIR)/MersenneTwister.h
