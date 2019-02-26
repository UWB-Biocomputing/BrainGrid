/*
 * AllIZHNeurons.cpp
 *
 */

#include "AllIZHNeurons.h"
#if defined(USE_GPU)
#include "AllNeuronsDeviceFuncs.h"
#include <helper_cuda.h>
#endif

// Default constructor
CUDA_CALLABLE AllIZHNeurons::AllIZHNeurons()
{
}

CUDA_CALLABLE AllIZHNeurons::~AllIZHNeurons()
{
}

/*
 *  Create and setup neurons properties.
 */
void AllIZHNeurons::createNeuronsProps()
{
    m_pNeuronsProps = new AllIZHNeuronsProps();
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
CUDA_CALLABLE void AllIZHNeurons::advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, 
#if defined(USE_GPU)
float* randNoise)
#else // defined(USE_GPU)
Norm* normRand)
#endif // defined(USE_GPU)
{
    AllIZHNeuronsProps *pNeuronsProps = reinterpret_cast<AllIZHNeuronsProps*>(pINeuronsProps);
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProps->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProps->summation_map[index];
    BGFLOAT &I0 = pNeuronsProps->I0[index];
    BGFLOAT &Inoise = pNeuronsProps->Inoise[index];
    BGFLOAT &C2 = pNeuronsProps->C2[index];
    BGFLOAT &C3 = pNeuronsProps->C3[index];
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    BGFLOAT &a = pNeuronsProps->Aconst[index];
    BGFLOAT &b = pNeuronsProps->Bconst[index];
    BGFLOAT &u = pNeuronsProps->u[index];
    BGFLOAT &Cconst = pNeuronsProps->Cconst[index];
    BGFLOAT &Dconst = pNeuronsProps->Dconst[index];
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

        BGFLOAT Vint = Vm * 1000;

        // Izhikevich model integration step
        BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
        u = u + C3 * a * (b * Vint - u);

        Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
    }

    DEBUG_MID(printf("%d %f\n",  index, Vm);)
        DEBUG_MID(printf("NEURON[%d] {\n", index);
            printf("\tVm = %f\n", Vm);
            printf("\ta = %f\n", a);
            printf("\tb = %f\n", b);
            printf("\tCconst = %f\n", Cconst);
            printf("\tDconst = %f\n", Dconst);
            printf("\tu = %f\n", u);
            printf("\tVthresh = %f\n", Vthresh);
            printf("\tsummationPoint = %f\n", summationPoint);
            printf("\tI0 = %f\n", I0);
            printf("\tInoise = %f\n", Inoise);
            printf("\tC2 = %f\n", C2);
            printf("\tC3 = %f\n}\n", C3);
    ;)

    // clear synaptic input for next time step
    summationPoint = 0;
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
CUDA_CALLABLE void AllIZHNeurons::fire(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps) const
{
    AllSpikingNeurons::fire(index, maxSpikes, deltaT, simulationStep, pINeuronsProps);

    // calculate the number of steps in the absolute refractory period
    AllIZHNeuronsProps *pNeuronsProps = reinterpret_cast<AllIZHNeuronsProps*>(pINeuronsProps);
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    BGFLOAT &Trefract = pNeuronsProps->Trefract[index];
    BGFLOAT &c = pNeuronsProps->Cconst[index];
    BGFLOAT &d = pNeuronsProps->Dconst[index];
    BGFLOAT &u = pNeuronsProps->u[index];

    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = c * 0.001;
    u = u + d;
}

#if defined(USE_GPU)

/*
 *  Create an AllNeurons class object in device
 *
 *  @param pAllNeurons_d       Device memory address to save the pointer of created AllNeurons object.
 *  @param pAllNeuronsProps_d  Pointer to the neurons properties in device memory.
 */
void AllIZHNeurons::createAllNeuronsInDevice(IAllNeurons** pAllNeurons_d, IAllNeuronsProps *pAllNeuronsProps_d)
{
    IAllNeurons **pAllNeurons_t; // temporary buffer to save pointer to IAllNeurons object.

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pAllNeurons_t, sizeof( IAllNeurons * ) ) );

    // create an AllNeurons object in device memory.
    allocAllIZHNeuronsDevice <<< 1, 1 >>> ( pAllNeurons_t, pAllNeuronsProps_d );

    // save the pointer of the object.
    checkCudaErrors( cudaMemcpy ( pAllNeurons_d, pAllNeurons_t, sizeof( IAllNeurons * ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pAllNeurons_t ) );
}

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllIZHNeuronsDevice(IAllNeurons **pAllNeurons, IAllNeuronsProps *pAllNeuronsProps)
{
    *pAllNeurons = new AllIZHNeurons();
    (*pAllNeurons)->setNeuronsProps(pAllNeuronsProps);
}

#endif // USE_GPU
