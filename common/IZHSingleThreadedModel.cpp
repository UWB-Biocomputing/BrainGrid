#include "IZHSingleThreadedModel.h"
#include "AllIZHNeurons.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
IZHSingleThreadedModel::IZHSingleThreadedModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
    SingleThreadedSpikingModel(conns, neurons, synapses, layout)
{
}

/*
* Destructor
*/
IZHSingleThreadedModel::~IZHSingleThreadedModel() 
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
void IZHSingleThreadedModel::advanceNeuron(AllNeurons &neurons, const int index, const BGFLOAT deltaT)
{
    AllIZHNeurons &izhNeurons = dynamic_cast<AllIZHNeurons&>(neurons);

    BGFLOAT &Vm = izhNeurons.Vm[index];
    BGFLOAT &Vthresh = izhNeurons.Vthresh[index];
    BGFLOAT &summationPoint = izhNeurons.summation_map[index];
    BGFLOAT &I0 = izhNeurons.I0[index];
    BGFLOAT &Inoise = izhNeurons.Inoise[index];
    BGFLOAT &C1 = izhNeurons.C1[index];
    BGFLOAT &C2 = izhNeurons.C2[index];
    int &nStepsInRefr = izhNeurons.nStepsInRefr[index];

    BGFLOAT &a = izhNeurons.Aconst[index];
    BGFLOAT &b = izhNeurons.Bconst[index];
    BGFLOAT &u = izhNeurons.u[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(neurons, index, deltaT);
    } else {
        summationPoint += I0; // add IO
        // add noise
        BGFLOAT noise = (*rgNormrnd[0])();
        DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; // add noise

        // do the Euler integration step
        BGFLOAT Vb;
        Vb = Vm + deltaT * (0.04 * Vm * Vm + 5 * Vm + 140 - u) + C2 * summationPoint;
        u = u + deltaT * a * (b * Vm - u);
        Vm = Vb;
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
void IZHSingleThreadedModel::fire(AllNeurons &neurons, const int index, const BGFLOAT deltaT) const
{
    SingleThreadedSpikingModel::fire(neurons, index, deltaT);

    // calculate the number of steps in the absolute refractory period
    AllIZHNeurons &izhNeurons = dynamic_cast<AllIZHNeurons&>(neurons);

    BGFLOAT &Vm = izhNeurons.Vm[index];
    int &nStepsInRefr = izhNeurons.nStepsInRefr[index];
    BGFLOAT &Trefract = izhNeurons.Trefract[index];

    BGFLOAT &c = izhNeurons.Cconst[index];
    BGFLOAT &d = izhNeurons.Dconst[index];
    BGFLOAT &u = izhNeurons.u[index];

    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = c;
    u = u + d;
}
