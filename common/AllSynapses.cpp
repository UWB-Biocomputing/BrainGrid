#include "AllSynapses.h"

AllSynapses::AllSynapses() :
        count_neurons(0),
        max_synapses(0)
{
    summationCoord = NULL;
    W = NULL;
    synapseCoord = NULL;
    deltaT = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delay = NULL;
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
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) :
        count_neurons(num_neurons),
        max_synapses(max_synapses)
{
	uint32_t total_synapses = num_neurons * max_synapses;

    summationCoord = new Coordinate[total_synapses]();
    W = new BGFLOAT[total_synapses]();
    synapseCoord = new Coordinate[total_synapses]();
    deltaT = new TIMEFLOAT[total_synapses]();
    psr = new BGFLOAT[total_synapses]();
    decay = new BGFLOAT[total_synapses]();
    total_delay = new uint32_t[total_synapses]();
    delayQueue = new uint32_t[total_synapses]();
    delay = new uint32_t[total_synapses]();
    type = new synapseType[total_synapses]();
    tau = new BGFLOAT[total_synapses]();
    r = new BGFLOAT[total_synapses]();
    u = new BGFLOAT[total_synapses]();
    D = new BGFLOAT[total_synapses]();
    U = new BGFLOAT[total_synapses]();
    F = new BGFLOAT[total_synapses]();
    lastSpike = new uint64_t[total_synapses]();
    in_use = new GPU_COMPAT_BOOL[total_synapses]();
    synapse_counts = new uint32_t[total_synapses]();	
}

AllSynapses::~AllSynapses()
{
	delete[] summationCoord;
	delete[] W;
	delete[] synapseCoord;
	delete[] deltaT;
	delete[] psr;
	delete[] decay;
	delete[] total_delay;
	delete[] delayQueue;
	delete[] delay;
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

    summationCoord = NULL;
    W = NULL;
    synapseCoord = NULL;
    deltaT = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delay = NULL;
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
}
