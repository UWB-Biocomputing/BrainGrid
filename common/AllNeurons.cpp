#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() :
        size(0)
{
    C1.clear();
    C2.clear();
    Cm.clear();
    I0.clear();
    Iinject.clear();
    Inoise.clear();
    Isyn.clear();
    Rm.clear();
    Tau.clear();
    Trefract.clear();
    Vinit.clear();
    Vm.clear();
    Vreset.clear();
    Vrest.clear();
    Vthresh.clear();
    deltaT.clear();
	hasFired.clear();
    nStepsInRefr.clear();
    neuron_type_map.clear();
    spikeCount.clear();
    totalSpikeCount.clear();
    spike_history = NULL;
    starter_map = NULL;
    summation_map.clear();
}

AllNeurons::AllNeurons(const int size) :
        size(size)
{
    // TODO: Rename variables for easier identification
    C1.resize(size);
    C2.resize(size);
    Cm.resize(size);
    I0.resize(size);
    Iinject.resize(size);
    Inoise.resize(size);
    Isyn.resize(size);
    Rm.resize(size);
    Tau.resize(size);
    Trefract.resize(size);
    Vinit.resize(size);
    Vm.resize(size);
    Vreset.resize(size);
    Vrest.resize(size);
    Vthresh.resize(size);
    deltaT.resize(size);
	hasFired.resize(size);
    nStepsInRefr.resize(size);
    neuron_type_map.resize(size);
    spikeCount.resize(size);
    totalSpikeCount.resize(size);
    starter_map = NULL;
    summation_map.resize(size);

    starter_map = new bool[size];
    spike_history = new uint64_t*[size]();
}

AllNeurons::~AllNeurons()
{
	delete[] starter_map;
	delete[] spike_history;
    starter_map = NULL;
	spike_history = NULL;

    C1.clear();
    C2.clear();
    Cm.clear();
    I0.clear();
    Iinject.clear();
    Inoise.clear();
    Isyn.clear();
    Rm.clear();
    Tau.clear();
    Trefract.clear();
    Vinit.clear();
    Vm.clear();
    Vreset.clear();
    Vrest.clear();
    Vthresh.clear();
    deltaT.clear();
    hasFired.clear();
    nStepsInRefr.clear();
    neuron_type_map.clear();
    spikeCount.clear();
    totalSpikeCount.clear();
    summation_map.clear();
}
