#include "AllSynapses.h"

AllSynapses::AllSynapses() :
        count_neurons(0),
        max_synapses(0)
{
    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    deltaT = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue[0] = NULL;
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
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) :
        count_neurons(num_neurons),
        max_synapses(max_synapses)
{
    summationCoord = new Coordinate*[num_neurons];
    W = new FLOAT*[num_neurons];
    summationPoint = new FLOAT**[num_neurons];
    synapseCoord = new Coordinate*[num_neurons];
    deltaT = new FLOAT*[num_neurons];
    psr = new FLOAT*[num_neurons];
    decay = new FLOAT*[num_neurons];
    total_delay = new int*[num_neurons];
    delayQueue[0] = new uint32_t*[num_neurons];
    delayIdx = new int*[num_neurons];
    ldelayQueue = new int*[num_neurons];
    type = new synapseType*[num_neurons];
    tau = new FLOAT*[num_neurons];
    r = new FLOAT*[num_neurons];
    u = new FLOAT*[num_neurons];
    D = new FLOAT*[num_neurons];
    U = new FLOAT*[num_neurons];
    F = new FLOAT*[num_neurons];
    lastSpike = new uint64_t*[num_neurons];
    in_use = new bool*[num_neurons];
    synapse_counts = new size_t[num_neurons];

    for (int i = 0; i < num_neurons; i++) {
        summationCoord[i] = new Coordinate[max_synapses];
        W[i] = new FLOAT[max_synapses];
        summationPoint[i] = new FLOAT*[max_synapses];
        synapseCoord[i] = new Coordinate[max_synapses];
        deltaT[i] = new FLOAT[max_synapses];
        psr[i] = new FLOAT[max_synapses];
        decay[i] = new FLOAT[max_synapses];
        total_delay[i] = new int[max_synapses];
        delayQueue[0][i] = new uint32_t[max_synapses];
        delayIdx[i] = new int[max_synapses];
        ldelayQueue[i] = new int[max_synapses];
        type[i] = new synapseType[max_synapses];
        tau[i] = new FLOAT[max_synapses];
        r[i] = new FLOAT[max_synapses];
        u[i] = new FLOAT[max_synapses];
        D[i] = new FLOAT[max_synapses];
        U[i] = new FLOAT[max_synapses];
        F[i] = new FLOAT[max_synapses];
        lastSpike[i] = new uint64_t[max_synapses];
        in_use[i] = new bool[max_synapses];

        for (int j = 0; j < max_synapses; j++) {
            in_use[i][j] = false;
            summationPoint[i][j] = NULL;
        }

        synapse_counts[i] = 0;
    }
}

AllSynapses::~AllSynapses()
{
    for (int i = 0; i < count_neurons; i++) {
        // Release references to summation points - while held by the synapse, these references are
        // not owned by the synapse.
        for (size_t j = 0; j < max_synapses; j++) {
            summationPoint[i][j] = NULL;
        }

        delete[] summationCoord[i];
        delete[] W[i];
        delete[] summationPoint[i];
        delete[] synapseCoord[i];
        delete[] deltaT[i];
        delete[] psr[i];
        delete[] decay[i];
        delete[] total_delay[i];
        delete[] delayQueue[i];
        delete[] delayIdx[i];
        delete[] ldelayQueue[i];
        delete[] type[i];
        delete[] tau[i];
        delete[] r[i];
        delete[] u[i];
        delete[] D[i];
        delete[] U[i];
        delete[] F[i];
        delete[] lastSpike[i];
        delete[] in_use[i];
    }

    delete[] summationCoord;
    delete[] W;
    delete[] summationPoint;
    delete[] synapseCoord;
    delete[] deltaT;
    delete[] psr;
    delete[] decay;
    delete[] total_delay;
    delete[] delayQueue[0];
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

    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    deltaT = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue[0] = NULL;
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
}
