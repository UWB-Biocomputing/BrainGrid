#include "AllLIFNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllLIFNeurons::AllLIFNeurons() : AllIFNeurons()
{
}

AllLIFNeurons::~AllLIFNeurons()
{
}

#if !defined(USE_GPU)
/**
 *  Update the indexed Neuron.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration.
 */
void AllLIFNeurons::advanceNeuron(const int index, const BGFLOAT deltaT)
{
    BGFLOAT &Vm = this->Vm[index];
    BGFLOAT &Vthresh = this->Vthresh[index];
    BGFLOAT &summationPoint = this->summation_map[index];
    BGFLOAT &I0 = this->I0[index];
    BGFLOAT &Inoise = this->Inoise[index];
    BGFLOAT &C1 = this->C1[index];
    BGFLOAT &C2 = this->C2[index];
    int &nStepsInRefr = this->nStepsInRefr[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(index, deltaT);
    } else {
        summationPoint += I0; // add IO
        // add noise
        BGFLOAT noise = (*rgNormrnd[0])();
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

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration
 */
void AllLIFNeurons::fire(const int index, const BGFLOAT deltaT) const
{
    AllSpikingNeurons::fire(index, deltaT);

    // calculate the number of steps in the absolute refractory period
    nStepsInRefr[index] = static_cast<int> ( Trefract[index] / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm[index] = Vreset[index];
}
#endif
