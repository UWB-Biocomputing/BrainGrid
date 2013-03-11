#include "AllNeurons.h"

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
    spike_history = NULL;
    starter_map = NULL;
    summation_map = NULL;
}

AllNeurons::AllNeurons(const int size) :
        size(size)
{
    C1 = new FLOAT[size];
    C2 = new FLOAT[size];
    Cm = new FLOAT[size];
    I0 = new FLOAT[size];
    Iinject = new FLOAT[size];
    Inoise = new FLOAT[size];
    Isyn = new FLOAT[size];
    Rm = new FLOAT[size];
    Tau = new FLOAT[size];
    Trefract = new FLOAT[size];
    Vinit = new FLOAT[size];
    Vm = new FLOAT[size];
    Vreset = new FLOAT[size];
    Vrest = new FLOAT[size];
    Vthresh = new FLOAT[size];
    deltaT = new FLOAT[size];
    hasFired = new bool[size];
    nStepsInRefr = new int[size];
    neuron_type_map = new neuronType[size];
    spikeCount = new int[size];
    starter_map = new bool[size];
    summation_map = new FLOAT[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
        spike_history[i] = NULL;
    }
}

AllNeurons::~AllNeurons()
{
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
    delete[] deltaT;
    delete[] hasFired;
    delete[] nStepsInRefr;
    delete[] neuron_type_map;
    delete[] spikeCount;
    delete[] starter_map;
    delete[] summation_map;

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
    starter_map = NULL;
    summation_map = NULL;
}
