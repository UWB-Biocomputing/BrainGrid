#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() : size(0)
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
    hasFired = NULL;
    nStepsInRefr = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    totalSpikeCount = NULL;
    spike_history = NULL;
    starter_map = NULL;
    summation_map = NULL;
}

AllNeurons::AllNeurons(const int size) :
        size(size)
{
    // TODO: Rename variables for easier identification
    C1 = new BGFLOAT[size];
    C2 = new BGFLOAT[size];
    Cm = new BGFLOAT[size];
    I0 = new BGFLOAT[size];
    Iinject = new BGFLOAT[size];
    Inoise = new BGFLOAT[size];
    Isyn = new BGFLOAT[size];
    Rm = new BGFLOAT[size];
    Tau = new BGFLOAT[size];
    Trefract = new BGFLOAT[size];
    Vinit = new BGFLOAT[size];
    Vm = new BGFLOAT[size];
    Vreset = new BGFLOAT[size];
    Vrest = new BGFLOAT[size];
    Vthresh = new BGFLOAT[size];
    hasFired = new bool[size];
    nStepsInRefr = new int[size];
    neuron_type_map = new neuronType[size];
    spikeCount = new int[size];
    totalSpikeCount = new int[size];
    starter_map = new bool[size];
    summation_map = new BGFLOAT[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
        nStepsInRefr[i] = 0;
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
        totalSpikeCount[i] = 0;
    }
}

AllNeurons::~AllNeurons()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
    	    delete[] spike_history[i];    
        }

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
        delete[] Vm;
        delete[] Vreset;
        delete[] Vrest;
        delete[] Vthresh;
        delete[] hasFired;
        delete[] nStepsInRefr;
        delete[] neuron_type_map;
        delete[] spikeCount;
        delete[] totalSpikeCount;
        delete[] starter_map;
        delete[] summation_map;
        delete[] spike_history;
    }

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
    hasFired = NULL;
    nStepsInRefr = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    totalSpikeCount = NULL;
    starter_map = NULL;
    summation_map = NULL;
    spike_history = NULL;
}
