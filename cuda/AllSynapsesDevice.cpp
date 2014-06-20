#include "AllSynapsesDevice.h"

AllSynapsesDevice::AllSynapsesDevice() 
{
    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    type = NULL;
    tau = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
    lastSpike = NULL;
    in_use = NULL;
    synapse_counts = NULL;
    total_synapse_counts = 0;
    max_synapses = 0;
    max_total_synapses = 0;
}

AllSynapsesDevice::AllSynapsesDevice(const int num_neurons_c, const int max_synapses_c) 
{
    max_total_synapses = max_synapses_c * num_neurons_c;
    summationCoord = new Coordinate[max_total_synapses];
    W = new BGFLOAT[max_total_synapses];
    summationPoint = new BGFLOAT*[max_total_synapses];
    synapseCoord = new Coordinate[max_total_synapses];
    psr = new BGFLOAT[max_total_synapses];
    decay = new BGFLOAT[max_total_synapses];
    total_delay = new int[max_total_synapses];
    delayQueue = new uint32_t[max_total_synapses];
    delayIdx = new int[max_total_synapses];
    ldelayQueue = new int[max_total_synapses];
    type = new synapseType[max_total_synapses];
    tau = new BGFLOAT[max_total_synapses];
    r = new BGFLOAT[max_total_synapses];
    u = new BGFLOAT[max_total_synapses];
    D = new BGFLOAT[max_total_synapses];
    U = new BGFLOAT[max_total_synapses];
    F = new BGFLOAT[max_total_synapses];
    lastSpike = new uint64_t[max_total_synapses];
    in_use = new bool[max_total_synapses];
    synapse_counts = new size_t[num_neurons_c];
    total_synapse_counts = 0;
    max_synapses = max_synapses_c;
}


AllSynapsesDevice::~AllSynapsesDevice()
{
    if (max_total_synapses != 0) {
        delete[] summationCoord;
        delete[] W;
        delete[] summationPoint;
        delete[] synapseCoord;
        delete[] psr;
        delete[] decay;
        delete[] total_delay;
        delete[] delayQueue;
        delete[] delayIdx;
        delete[] ldelayQueue;
        delete[] type;
        delete[] tau;
        delete[] r;
        delete[] u;
        delete[] D;
        delete[] U;
        delete[] F;
        delete[] lastSpike;
        delete[] in_use;
        delete[] synapse_counts;
    }

    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    type = NULL;
    tau = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
    lastSpike = NULL;
    in_use = NULL;
    synapse_counts = NULL;
    max_total_synapses = 0;
}
