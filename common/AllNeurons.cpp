#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() :
        size(0)
{
    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    deltaT = NULL;
	hasFired = NULL;
    nStepsInRefr = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    totalSpikeCount = NULL;
	summation = NULL;
    spike_history = NULL;
    starter_map = NULL;
}

AllNeurons::AllNeurons(const int size) :
        size(size)
{
    // TODO: Rename variables for easier identification
    C1 = new BGFLOAT[size]();
    C2 = new BGFLOAT[size]();
    Cm = new BGFLOAT[size]();
    I0 = new BGFLOAT[size]();
    Iinject = new BGFLOAT[size]();
    Inoise = new BGFLOAT[size]();
    Isyn = new BGFLOAT[size]();
    Rm = new BGFLOAT[size]();
    Tau = new BGFLOAT[size]();
    Trefract = new BGFLOAT[size]();
    Vinit = new BGFLOAT[size]();
    Vm = new BGFLOAT[size]();
    Vreset = new BGFLOAT[size]();
    Vrest = new BGFLOAT[size]();
    Vthresh = new BGFLOAT[size]();
    deltaT = new TIMEFLOAT[size]();
    hasFired = new bool[size]();
    nStepsInRefr = new uint32_t[size]();
    neuron_type_map = new neuronType[size]();
    spikeCount = new uint32_t[size]();
    totalSpikeCount = new uint32_t[size]();
    starter_map = new bool[size]();
    summation = new BGFLOAT[size]();
    spike_history = new uint64_t*[size]();

    starter_map = new bool[size]();
    spike_history = new uint64_t*[size]();
}

AllNeurons::~AllNeurons()
{
	delete[] starter_map;
	delete[] spike_history;
    starter_map = NULL;
	spike_history = NULL;

	delete[] C1;
	delete[] C2;
	delete[] Cm;
	delete[] I0;
	delete[] Iinject;
	delete[] Inoise;
	delete[] Isyn;
	delete[] Rm;
	delete[] Tau;
	delete[] Trefract;
	delete[] Vinit;
	delete[] Vreset;
	delete[] Vrest;
	delete[] Vthresh;
	delete[] deltaT;
	delete[] hasFired;
	delete[] nStepsInRefr;
	delete[] neuron_type_map;
	delete[] spikeCount;
	delete[] totalSpikeCount;
	delete[] summation;

    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    deltaT = NULL;
    hasFired = NULL;
    nStepsInRefr = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    totalSpikeCount = NULL;
	summation = NULL;
}
