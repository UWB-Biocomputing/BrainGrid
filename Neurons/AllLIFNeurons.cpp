#include "AllLIFNeurons.h"
#include "ParseParamError.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
CUDA_CALLABLE AllLIFNeurons::AllLIFNeurons()
{
}

CUDA_CALLABLE AllLIFNeurons::~AllLIFNeurons()
{
}


/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *
 *  @param  index                 Index of the Neuron to update.
 *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param  deltaT                Inner simulation step duration.
 *  @param  simulationStep        The current simulation step.
 *  @param  pINeuronsProps        Pointer to the neurons properties. 
 *  @param  randNoise             Pointer to device random noise array.
 *  @param  normRand              Pointer to the normalized random number generator.
 */
CUDA_CALLABLE void AllLIFNeurons::advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, 
#if defined(USE_GPU)
float* randNoise)
#else // defined(USE_GPU)
Norm* normRand)
#endif // defined(USE_GPU)
{
    AllIFNeuronsProps *pNeuronsProps = reinterpret_cast<AllIFNeuronsProps*>(pINeuronsProps);
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProps->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProps->summation_map[index];
    BGFLOAT &I0 = pNeuronsProps->I0[index];
    BGFLOAT &Inoise = pNeuronsProps->Inoise[index];
    BGFLOAT &C1 = pNeuronsProps->C1[index];
    BGFLOAT &C2 = pNeuronsProps->C2[index];
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    bool &hasFired = pNeuronsProps->hasFired[index];

    hasFired = false;

    if (nStepsInRefr > 0) { // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) { // should it fire?
        fire(index, maxSpikes, deltaT, simulationStep, pINeuronsProps);
    } else {
        summationPoint += I0; // add IO
        // add noise
#if defined(USE_GPU)
        BGFLOAT noise = randNoise[index];
#else // defined(USE_GPU)
        BGFLOAT noise = (*normRand)();
#endif // defined(USE_GPU)
        DEBUG_MID(printf("ADVANCE NEURON[%d] :: noise = %f\n", index, noise);)
        summationPoint += noise * Inoise; // add noise
        Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;

    DEBUG_MID(printf("%d %f\n",  index, Vm);)
        DEBUG_MID(printf("NEURON[%d] {\n", index);
            printf("\tVm = %f\n", Vm);
            printf("\tVthresh = %f\n", Vthresh);
            printf("\tsummationPoint = %f\n", summationPoint);
            printf("\tI0 = %f\n", I0);
            printf("\tInoise = %f\n", Inoise);
            printf("\tC1 = %f\n", C1);
            printf("\tC2 = %f\n}\n", C2);
    ;)
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index                 Index of the Neuron to update.
 *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param  deltaT                Inner simulation step duration.
 *  @param  simulationStep        The current simulation step.
 *  @param  pINeuronsProps        Pointer to the neurons properties.
 */
CUDA_CALLABLE void AllLIFNeurons::fire(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps) const
{
    AllIFNeuronsProps *pNeuronsProps = reinterpret_cast<AllIFNeuronsProps*>(pINeuronsProps);
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    BGFLOAT &Trefract = pNeuronsProps->Trefract[index];
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vreset = pNeuronsProps->Vreset[index];

    AllSpikingNeurons::fire(index, maxSpikes, deltaT, simulationStep, pINeuronsProps);

    // calculate the number of steps in the absolute refractory period
    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = Vreset;
}

#if defined(USE_GPU)

/*
 *  Create an AllNeurons class object in device
 *
 *  @param pAllNeurons_d       Device memory address to save the pointer of created AllNeurons object.
 *  @param pAllNeuronsProps_d  Pointer to the neurons properties in device memory.
 */
void AllLIFNeurons::createAllNeuronsInDevice(IAllNeurons** pAllNeurons_d, IAllNeuronsProps *pAllNeuronsProps_d)
{
    IAllNeurons **pAllNeurons_t; // temporary buffer to save pointer to IAllNeurons object.

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pAllNeurons_t, sizeof( IAllNeurons * ) ) );

    // create an AllNeurons object in device memory.
    allocAllLIFNeuronsDevice <<< 1, 1 >>> ( pAllNeurons_t, pAllNeuronsProps_d );

    // save the pointer of the object.
    checkCudaErrors( cudaMemcpy ( pAllNeurons_d, pAllNeurons_t, sizeof( IAllNeurons * ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pAllNeurons_t ) );
}

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllLIFNeuronsDevice(IAllNeurons **pAllNeurons, IAllNeuronsProps *pAllNeuronsProps)
{
    *pAllNeurons = new AllLIFNeurons();
    (*pAllNeurons)->setNeuronsProps(pAllNeuronsProps);
}

#endif // USE_GPU
