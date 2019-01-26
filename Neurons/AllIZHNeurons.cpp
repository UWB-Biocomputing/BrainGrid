/*
 * AllIZHNeurons.cpp
 *
 */

#include "AllIZHNeurons.h"

// Default constructor
AllIZHNeurons::AllIZHNeurons()
{
}

AllIZHNeurons::~AllIZHNeurons()
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
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllIZHNeurons::advanceNeuron(const int index, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    AllIZHNeuronsProps *pNeuronsProps = dynamic_cast<AllIZHNeuronsProps*>(m_pNeuronsProps);
    BGFLOAT &Vm = pNeuronsProps->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProps->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProps->summation_map[index];
    BGFLOAT &I0 = pNeuronsProps->I0[index];
    BGFLOAT &Inoise = pNeuronsProps->Inoise[index];
    BGFLOAT &C1 = pNeuronsProps->C1[index];
    BGFLOAT &C2 = pNeuronsProps->C2[index];
    BGFLOAT &C3 = pNeuronsProps->C3[index];
    int &nStepsInRefr = pNeuronsProps->nStepsInRefr[index];
    BGFLOAT &a = pNeuronsProps->Aconst[index];
    BGFLOAT &b = pNeuronsProps->Bconst[index];
    BGFLOAT &u = pNeuronsProps->u[index];
    BGFLOAT &Cconst = pNeuronsProps->Cconst[index];
    BGFLOAT &Dconst = pNeuronsProps->Dconst[index];

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
            << "\tC1 = " << C1 << endl
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
void AllIZHNeurons::fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const
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
