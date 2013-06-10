#include "AllSynapses.h"

AllSynapses::AllSynapses() :
        count_neurons(0),
        max_synapses(0)
{
    summationCoord.clear();
    W.clear();
    synapseCoord.clear();
    deltaT.clear();
    psr.clear();
    decay.clear();
    total_delay.clear();
    delayQueue.clear();
    delay.clear();
    type.clear();
    tau.clear();
    r.clear();
    u.clear();
    D.clear();
    U.clear();
    F.clear();
    lastSpike.clear();
    in_use.clear();
    synapse_counts.clear();
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) :
        count_neurons(num_neurons),
        max_synapses(max_synapses)
{
	uint32_t total_synapses = num_neurons * max_synapses;

    summationCoord.resize(total_synapses);
    W.resize(total_synapses);
    synapseCoord.resize(total_synapses);
    deltaT.resize(total_synapses);
    psr.resize(total_synapses);
    decay.resize(total_synapses);
    total_delay.resize(total_synapses);
    delayQueue.resize(total_synapses);
    delay.resize(total_synapses);
    type.resize(total_synapses);
    tau.resize(total_synapses);
    r.resize(total_synapses);
    u.resize(total_synapses);
    D.resize(total_synapses);
    U.resize(total_synapses);
    F.resize(total_synapses);
    lastSpike.resize(total_synapses);
	in_use.resize(total_synapses);
    synapse_counts.resize(total_synapses);
}

AllSynapses::~AllSynapses()
{
    summationCoord.clear();
    W.clear();
    synapseCoord.clear();
    deltaT.clear();
    psr.clear();
    decay.clear();
    total_delay.clear();
    delayQueue.clear();
    delay.clear();
    type.clear();
    tau.clear();
    r.clear();
    u.clear();
    D.clear();
    U.clear();
    F.clear();
    lastSpike.clear();
    in_use.clear();
    synapse_counts.clear();
}
