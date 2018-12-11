#include "AllIFNeuronsProperties.h"

// Default constructor
AllIFNeuronsProperties::AllIFNeuronsProperties() : AllSpikingNeuronsProperties()
{
    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    nStepsInRefr = NULL;
}

AllIFNeuronsProperties::~AllIFNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIFNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProperties::setupNeuronsProperties(sim_info, clr_info);

    // TODO: Rename variables for easier identification
    C1 = new BGFLOAT[size];
    C2 = new BGFLOAT[size];
    Cm = new BGFLOAT[size];
    I0 = new BGFLOAT[size];
    Iinject = new BGFLOAT[size];
    Inoise = new BGFLOAT[size];
    Isyn = new BGFLOAT[size];
    Rm = new BGFLOAT[size];
    Tau = new BGFLOAT[size];
    Trefract = new BGFLOAT[size];
    Vinit = new BGFLOAT[size];
    Vm = new BGFLOAT[size];
    Vreset = new BGFLOAT[size];
    Vrest = new BGFLOAT[size];
    Vthresh = new BGFLOAT[size];
    nStepsInRefr = new int[size];

    for (int i = 0; i < size; ++i) {
        nStepsInRefr[i] = 0;
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIFNeuronsProperties::cleanupNeuronsProperties()
{
    if (size != 0) {
        delete[] C1;
        delete[] C2;
        delete[] Cm;
        delete[] I0;
        delete[] Iinject;
        delete[] Inoise;
        delete[] Isyn;
        delete[] Rm;
        delete[] Tau;
        delete[] Trefract;
        delete[] Vinit;
        delete[] Vm;
        delete[] Vreset;
        delete[] Vrest;
        delete[] Vthresh;
        delete[] nStepsInRefr;
    }

    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    nStepsInRefr = NULL;

    AllSpikingNeuronsProperties::cleanupNeuronsProperties();
}
