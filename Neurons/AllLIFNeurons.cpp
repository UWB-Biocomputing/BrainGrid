#include "AllLIFNeurons.h"
#include "ParseParamError.h"
#if defined(USE_GPU)
#include "AllNeuronsDeviceFuncs.h"
#include <helper_cuda.h>
#endif

// Default constructor
AllLIFNeurons::AllLIFNeurons()
{
}

AllLIFNeurons::~AllLIFNeurons()
{
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
void AllLIFNeurons::advanceNeuron(const int index, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    AllIFNeuronsProps *pNeuronsProps = dynamic_cast<AllIFNeuronsProps*>(m_pNeuronsProps);
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProps->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProps->summation_map[index];
    BGFLOAT &I0 = pNeuronsProps->I0[index];
    BGFLOAT &Inoise = pNeuronsProps->Inoise[index];
    BGFLOAT &C1 = pNeuronsProps->C1[index];
    BGFLOAT &C2 = pNeuronsProps->C2[index];
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(index, sim_info, iStepOffset);
    } else {
        summationPoint += I0; // add IO
        // add noise
        BGFLOAT noise = (*clr_info->normRand)();
        DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; // add noise
        Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;

    DEBUG_MID(cout << index << " " << Vm << endl;)
        DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
            << "\tVm = " << Vm << endl
            << "\tVthresh = " << Vthresh << endl
            << "\tsummationPoint = " << summationPoint << endl
            << "\tI0 = " << I0 << endl
            << "\tInoise = " << Inoise << endl
            << "\tC1 = " << C1 << endl
            << "\tC2 = " << C2 << endl
            << "}" << endl
    ;)
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllLIFNeurons::fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const
{
    AllIFNeuronsProps *pNeuronsProps = dynamic_cast<AllIFNeuronsProps*>(m_pNeuronsProps);
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    BGFLOAT &Trefract = pNeuronsProps->Trefract[index];
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vreset = pNeuronsProps->Vreset[index];

    const BGFLOAT deltaT = sim_info->deltaT;
    AllSpikingNeurons::fire(index, sim_info, iStepOffset);

    // calculate the number of steps in the absolute refractory period
    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = Vreset;
}

#else // USE_GPU

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsProps       Reference to the allNeuronsProps struct
 *                                 on device memory.
 *  @param  allSynapsesProps      Reference to the allSynapsesProps struct
 *                                 on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllLIFNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsProps, void* allSynapsesProps, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset )
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
    advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIFNeuronsProps *)allNeuronsProps, (AllSpikingSynapsesProps*)allSynapsesProps, synapseIndexMapDevice, m_fAllowBackPropagation, iStepOffset );
}

#endif // USE_GPU
