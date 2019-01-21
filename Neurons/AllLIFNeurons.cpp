#include "AllLIFNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllLIFNeurons::AllLIFNeurons()
{
}

AllLIFNeurons::~AllLIFNeurons()
{
}

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
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Vm = pNeuronsProperties->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProperties->summation_map[index];
    BGFLOAT &I0 = pNeuronsProperties->I0[index];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[index];
    BGFLOAT &C1 = pNeuronsProperties->C1[index];
    BGFLOAT &C2 = pNeuronsProperties->C2[index];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[index];

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
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[index];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[index];
    BGFLOAT &Vm = pNeuronsProperties->Vm[index];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[index];

    const BGFLOAT deltaT = sim_info->deltaT;
    AllSpikingNeurons::fire(index, sim_info, iStepOffset);

    // calculate the number of steps in the absolute refractory period
    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = Vreset;
}
