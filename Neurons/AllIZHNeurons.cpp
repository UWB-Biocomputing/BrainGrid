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
void AllIZHNeurons::setupNeuronsProps()
{
    m_pNeuronsProperties = new AllIZHNeuronsProperties();
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
    AllIZHNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIZHNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Vm = pNeuronsProperties->Vm[index];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[index];
    BGFLOAT &summationPoint = pNeuronsProperties->summation_map[index];
    BGFLOAT &I0 = pNeuronsProperties->I0[index];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[index];
    BGFLOAT &C1 = pNeuronsProperties->C1[index];
    BGFLOAT &C2 = pNeuronsProperties->C2[index];
    BGFLOAT &C3 = pNeuronsProperties->C3[index];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[index];
    BGFLOAT &a = pNeuronsProperties->Aconst[index];
    BGFLOAT &b = pNeuronsProperties->Bconst[index];
    BGFLOAT &u = pNeuronsProperties->u[index];
    BGFLOAT &Cconst = pNeuronsProperties->Cconst[index];
    BGFLOAT &Dconst = pNeuronsProperties->Dconst[index];

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
    AllIZHNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIZHNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Vm = pNeuronsProperties->Vm[index];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[index];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[index];
    BGFLOAT &c = pNeuronsProperties->Cconst[index];
    BGFLOAT &d = pNeuronsProperties->Dconst[index];
    BGFLOAT &u = pNeuronsProperties->u[index];

    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = c * 0.001;
    u = u + d;
}
