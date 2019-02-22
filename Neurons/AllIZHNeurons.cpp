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

#if !defined(USE_GPU)

/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
CUDA_CALLABLE void AllIZHNeurons::advanceNeuron(const int index, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    AllIZHNeuronsProps *pNeuronsProps = dynamic_cast<AllIZHNeuronsProps*>(m_pNeuronsProps);
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
        fire(index, sim_info, iStepOffset);
    } else {
        summationPoint += I0; // add IO

        // add noise
        BGFLOAT noise = (*clr_info->normRand)();
        DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; // add noise

        BGFLOAT Vint = Vm * 1000;

        // Izhikevich model integration step
        BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
        u = u + C3 * a * (b * Vint - u);

        Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
    }

    DEBUG_MID(cout << index << " " << Vm << endl;)
        DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
            << "\tVm = " << Vm << endl
            << "\ta = " << a << endl
            << "\tb = " << b << endl
            << "\tc = " << Cconst << endl
            << "\td = " << Dconst << endl
            << "\tu = " << u << endl
            << "\tVthresh = " << Vthresh << endl
            << "\tsummationPoint = " << summationPoint << endl
            << "\tI0 = " << I0 << endl
            << "\tInoise = " << Inoise << endl
            << "\tC2 = " << C2 << endl
            << "\tC3 = " << C3 << endl
            << "}" << endl
    ;)

    // clear synaptic input for next time step
    summationPoint = 0;
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
CUDA_CALLABLE void AllIZHNeurons::fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const
{
    const BGFLOAT deltaT = sim_info->deltaT;
    AllSpikingNeurons::fire(index, sim_info, iStepOffset);

    // calculate the number of steps in the absolute refractory period
    AllIZHNeuronsProps *pNeuronsProps = dynamic_cast<AllIZHNeuronsProps*>(m_pNeuronsProps);
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

#else // USE_GPU

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsProps       Reference to the allNeurons struct on device memory.
 *  @param  allSynapsesProps      Reference to the allSynapses struct on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo class to read information from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllIZHNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsProps, void* allSynapsesProps, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset)
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int neuron_count = clr_info->totalClusterNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIZHNeuronsProps *)allNeuronsProps, (AllSpikingSynapsesProps*)allSynapsesProps, synapseIndexMapDevice, m_fAllowBackPropagation, iStepOffset );
}

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
