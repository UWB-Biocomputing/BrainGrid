#include "LIFSingleThreadedModel.h"
#include "AllIFNeurons.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
LIFSingleThreadedModel::LIFSingleThreadedModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
    SingleThreadedSpikingModel(conns, neurons, synapses, layout)
{
}

/*
* Destructor
*/
LIFSingleThreadedModel::~LIFSingleThreadedModel() 
{
	//Let Model base class handle de-allocation
}

/* -----------------
* # Helper Functions
* ------------------
*/


/**
 *  Update the indexed Neuron.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration.
 */
void LIFSingleThreadedModel::advanceNeuron(AllNeurons &neurons, const int index, const BGFLOAT deltaT)
{
    AllIFNeurons &ifNeurons = dynamic_cast<AllIFNeurons&>(neurons);

    BGFLOAT &Vm = ifNeurons.Vm[index];
    BGFLOAT &Vthresh = ifNeurons.Vthresh[index];
    BGFLOAT &summationPoint = ifNeurons.summation_map[index];
    BGFLOAT &I0 = ifNeurons.I0[index];
    BGFLOAT &Inoise = ifNeurons.Inoise[index];
    BGFLOAT &C1 = ifNeurons.C1[index];
    BGFLOAT &C2 = ifNeurons.C2[index];
    int &nStepsInRefr = ifNeurons.nStepsInRefr[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(ifNeurons, index, deltaT);
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
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration
 */
void LIFSingleThreadedModel::fire(AllSpikingNeurons &neurons, const int index, const BGFLOAT deltaT) const
{
    SingleThreadedSpikingModel::fire(neurons, index, deltaT);

    // calculate the number of steps in the absolute refractory period
    AllIFNeurons &ifNeurons = dynamic_cast<AllIFNeurons&>(neurons);
    ifNeurons.nStepsInRefr[index] = static_cast<int> ( ifNeurons.Trefract[index] / deltaT + 0.5 );

    // reset to 'Vreset'
    ifNeurons.Vm[index] = ifNeurons.Vreset[index];
}

