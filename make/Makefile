all: growth growth_omp clean
# growth_gpu

#
# Source directories
#
MAIN = ..
INCDIR = $(MAIN)/include
MATRIXDIR = $(MAIN)/matrix
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
CXXFLAGS = -I$(INCDIR) -I$(MATRIXDIR) -DTIXML_USE_STL -DCDEBUG -Wall -DSTORE_SPIKEHISTORY
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

XMLOBJS = $(INCDIR)/tinyxml.o $(INCDIR)/tinyxmlparser.o $(INCDIR)/tinyxmlerror.o $(INCDIR)/tinystr.o

OTHEROBJS = $(INCDIR)/norm.o $(INCDIR)/ParamContainer.o $(INCDIR)/Timer.o

GPUOBJS = $(MAIN)/GpuSim.o \
       $(MAIN)/HostSim.o \
       $(MAIN)/DynamicSpikingSynapse_struct.o \
       $(MAIN)/LifNeuron_struct.o \
       $(MAIN)/DynamicSpikingSynapse.o \
       $(MAIN)/LifNeuron.o \
       $(MAIN)/Network_gpu.o \
       $(MAIN)/GpuSim_struct.o \
       $(MAIN)/BGDriver_gpu.o \
       $(MAIN)/global_gpu.o \
#       $(MAIN)/MersenneTwister_kernel.o

SINGLEOBJS = $(MAIN)/HostSim.o \
       $(MAIN)/SingleThreadedSim.o \
       $(MAIN)/DynamicSpikingSynapse.o \
       $(MAIN)/Network.o \
       $(MAIN)/BGDriver.o \
       $(MAIN)/global.o \
       $(MAIN)/LifNeuron.o 

MULTIOBJS = $(MAIN)/HostSim.o \
       $(MAIN)/MultiThreadedSim.o \
       $(MAIN)/DynamicSpikingSynapse_omp.o \
       $(MAIN)/Network_omp.o \
       $(MAIN)/BGDriver.o \
       $(MAIN)/global.o \
       $(MAIN)/LifNeuron_omp.o 

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
	rm -f $(MAIN)/*.o $(INCDIR)/*.o $(MATRIXDIR)/*.o
	#$(MAIN)/growth $(MAIN)/growth_omp

$(MAIN)/ParamContainer.o: $(INCDIR)/ParamContainer.h $(INCDIR)/ParamContainer.cpp
    
$(MAIN)/BGDriver.o: $(MAIN)/BGDriver.cpp $(MAIN)/global.h $(MAIN)/DynamicSpikingSynapse.h $(MAIN)/LifNeuron.h $(MAIN)/Network.h

$(MAIN)/DynamicSpikingSynapse.o: $(MAIN)/DynamicSpikingSynapse.cpp $(MAIN)/DynamicSpikingSynapse.h 

$(MAIN)/DynamicSpikingSynapse_omp.o: $(MAIN)/DynamicSpikingSynapse.cpp $(MAIN)/DynamicSpikingSynapse.h 
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c $(MAIN)/DynamicSpikingSynapse.cpp -o $(MAIN)/DynamicSpikingSynapse_omp.o

$(MAIN)/DynamicSpikingSynapse_struct.o: $(MAIN)/DynamicSpikingSynapse_struct.cpp $(MAIN)/DynamicSpikingSynapse_struct.h

$(MAIN)/global.o: $(MAIN)/global.h 

$(MAIN)/global_gpu.o: $(MAIN)/global.h 
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c $(MAIN)/global.cpp -o $(MAIN)/global_gpu.o

$(MAIN)/GpuSim.o: $(MAIN)/GpuSim.cpp $(MAIN)/GpuSim.h $(MAIN)/global.h $(MAIN)/LifNeuron_struct.h

$(MAIN)/GpuSim_struct.o: $(MAIN)/GpuSim_struct.cu $(MAIN)/global.h $(MAIN)/DynamicSpikingSynapse_struct.h $(MAIN)/LifNeuron_struct.h $(MAIN)/LifNeuron_struct_d.cu $(MAIN)/DynamicSpikingSynapse_struct_d.cu
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(INCDIR) -I$(MATRIXDIR)
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(INCDIR) -I$(MATRIXDIR)-DPERFORMANCE_METRICS
#	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v GpuSim_struct.cu $(CGPUFLAGS) -I$(INCDIR) -I$(MATRIXDIR)
	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v $(MAIN)/GpuSim_struct.cu $(CGPUFLAGS) -Iinclude -I$(INCDIR) -I$(MATRIXDIR) -DSTORE_SPIKEHISTORY -o $(MAIN)/GpuSim_struct.o

$(MAIN)/LifNeuron.o: $(MAIN)/LifNeuron.cpp $(MAIN)/LifNeuron.h $(MAIN)/INeuron.h $(MAIN)/INeuron.h

$(MAIN)/LifNeuron_omp.o: $(MAIN)/LifNeuron.cpp $(MAIN)/LifNeuron.h 
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c $(MAIN)/LifNeuron.cpp -o $(MAIN)/LifNeuron_omp.o

$(MAIN)/LifNeuron_struct.o: $(MAIN)/LifNeuron_struct.cpp $(MAIN)/LifNeuron_struct.h

#$(MAIN)/MersenneTwister_kernel.o: $(INCDIR)/MersenneTwister_kernel.cu $(INCDIR)/MersenneTwister.h
#	nvcc -c -g -G -arch=sm_13 -Xptxas=-v $(INCDIR)/MersenneTwister_kernel.cu -I$(INCDIR)
#	nvcc -c -g -G -arch=sm_13 -maxrregcount=16 -Xptxas=-v $(INCDIR)/MersenneTwister_kernel.cu -I$(INCDIR)

$(MAIN)/MultiThreadedSim.o: $(MAIN)/MultiThreadedSim.cpp $(MAIN)/MultiThreadedSim.h
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c $(MAIN)/MultiThreadedSim.cpp -o $(MAIN)/MultiThreadedSim.o

$(MAIN)/Network.o: $(MAIN)/Network.cpp $(MAIN)/Network.h $(MAIN)/global.h

$(MAIN)/Network_omp.o: $(MAIN)/Network.cpp $(MAIN)/Network.h $(MAIN)/global.h
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) -c $(MAIN)/Network.cpp -o $(MAIN)/Network_omp.o

$(MAIN)/Network_gpu.o: $(MAIN)/Network.cpp $(MAIN)/Network.h $(MAIN)/global.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c $(MAIN)/Network.cpp -o $(MAIN)/Network_gpu.o

$(MAIN)/BGDriver.o: $(MAIN)/BGDriver.cpp $(MAIN)/global.h $(MAIN)/DynamicSpikingSynapse.h $(MAIN)/LifNeuron.h $(MAIN)/Network.h

$(MAIN)/BGDriver_gpu.o: $(MAIN)/BGDriver.cpp $(MAIN)/global.h $(MAIN)/DynamicSpikingSynapse.h $(MAIN)/LifNeuron.h $(MAIN)/Network.h
	$(CXX) $(CXXFLAGS) $(CGPUFLAGS) -c $(MAIN)/BGDriver.cpp -o $(MAIN)/BGDriver_gpu.o

$(MAIN)/HostSim.o: $(MAIN)/HostSim.cpp $(MAIN)/HostSim.h $(MAIN)/ISimulation.h

$(MAIN)/SingleThreadedSim.o: $(MAIN)/SingleThreadedSim.cpp $(MAIN)/SingleThreadedSim.h

$(MAIN)/Timer.o: $(INCDIR)/Timer.cpp $(INCDIR)/Timer.h

$(MAIN)/norm.o: $(INCDIR)/norm.cpp $(INCDIR)/norm.h $(INCDIR)/MersenneTwister.cpp $(INCDIR)/MersenneTwister.h
