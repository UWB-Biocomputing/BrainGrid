#include "AllDynamicSTDPSynapses.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#include "AllSynapsesDeviceFuncs.h"
#endif // USE_GPU

// Default constructor
AllDynamicSTDPSynapses::AllDynamicSTDPSynapses()
{
}

AllDynamicSTDPSynapses::~AllDynamicSTDPSynapses()
{
}

/*
 *  Create and setup synapses properties.
 */
void AllDynamicSTDPSynapses::createSynapsesProps()
{
    m_pSynapsesProps = new AllDynamicSTDPSynapsesProps();
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllDynamicSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllDynamicSTDPSynapsesProps *pSynapsesProps = dynamic_cast<AllDynamicSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSTDPSynapses::resetSynapse(iSyn, deltaT);

    pSynapsesProps->u[iSyn] = DEFAULT_U;
    pSynapsesProps->r[iSyn] = 1.0;
    pSynapsesProps->lastSpike[iSyn] = ULONG_MAX;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  source      Coordinates of the source Neuron.
 *  @param  dest        Coordinates of the destination Neuron.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllDynamicSTDPSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllDynamicSTDPSynapsesProps *pSynapsesProps = dynamic_cast<AllDynamicSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSTDPSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    pSynapsesProps->U[iSyn] = DEFAULT_U;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            break;
        default:
            assert( false );
            break;
    }

    pSynapsesProps->U[iSyn] = U;
    pSynapsesProps->D[iSyn] = D;
    pSynapsesProps->F[iSyn] = F;
}

#if !defined(USE_GPU)
/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn             Index of the synapse to set.
 *  @param  deltaT           Inner simulation step duration.
 *  @param  iStepOffset      Offset from the current simulation step.
 *  @param  pISynapsesProps  Pointer to the synapses properties.
 */
CUDA_CALLABLE void AllDynamicSTDPSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, int iStepOffset, IAllSynapsesProps* pISynapsesProps)
{
    AllDynamicSTDPSynapsesProps *pSynapsesProps = dynamic_cast<AllDynamicSTDPSynapsesProps*>(pISynapsesProps);

    BGFLOAT &psr = pSynapsesProps->psr[iSyn];
    BGFLOAT &W = pSynapsesProps->W[iSyn];
    BGFLOAT &decay = pSynapsesProps->decay[iSyn];
    uint64_t &lastSpike = pSynapsesProps->lastSpike[iSyn];
    BGFLOAT &r = pSynapsesProps->r[iSyn];
    BGFLOAT &u = pSynapsesProps->u[iSyn];
    BGFLOAT &D = pSynapsesProps->D[iSyn];
    BGFLOAT &F = pSynapsesProps->F[iSyn];
    BGFLOAT &U = pSynapsesProps->U[iSyn];

    // adjust synapse parameters
    uint64_t simulationStep = g_simulationStep + iStepOffset;
    if (lastSpike != ULONG_MAX) {
        BGFLOAT isi = (simulationStep - lastSpike) * deltaT ;
        r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
        u = U + u * ( 1 - U ) * exp( -isi / F );
    }
    psr += ( ( W / decay ) * u * r );    // calculate psr
    lastSpike = simulationStep;          // record the time of the spike
}
#endif // !defined(USE_GPU)

#if defined(USE_GPU)
/**
 *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
 *  The class ID will be set to classSynapses_d in device memory,
 *  and the classSynapses_d will be referred to call a device function for the
 *  particular synapse class.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
 *  (see issue#137).
 */
void AllDynamicSTDPSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDynamicSTDPSynapses;

    checkCudaErrors( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}
#endif // USE_GPU
