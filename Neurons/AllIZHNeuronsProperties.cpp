#include "AllIZHNeuronsProperties.h"

// Default constructor
AllIZHNeuronsProperties::AllIZHNeuronsProperties() : AllIFNeuronsProperties()
{
    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
    C3 = NULL;
}

AllIZHNeuronsProperties::~AllIZHNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIZHNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllIFNeuronsProperties::setupNeuronsProperties(sim_info, clr_info);

    Aconst = new BGFLOAT[size];
    Bconst = new BGFLOAT[size];
    Cconst = new BGFLOAT[size];
    Dconst = new BGFLOAT[size];
    u = new BGFLOAT[size];
    C3 = new BGFLOAT[size];
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIZHNeuronsProperties::cleanupNeuronsProperties()
{
    if (size != 0) {
        delete[] Aconst;
        delete[] Bconst;
        delete[] Cconst;
        delete[] Dconst;
        delete[] u;
        delete[] C3;
    }

    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
    C3 = NULL;

    AllIFNeuronsProperties::cleanupNeuronsProperties();
}
