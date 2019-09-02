#include "SimulationInfo.h"
#include "ParseParamError.h"

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  simDoc  the TiXmlDocument to read from.
 *  @return true if successful, false otherwise.
 */
bool SimulationInfo::readParameters(ParameterManager* paramMgr)
{
    if (paramMgr->getIntByXpath("//PoolSize//x/text()", width) &&
        paramMgr->getIntByXpath("//PoolSize//y/text()", height) && 
        paramMgr->getBGFloatByXpath("//SimParams//Tsim/text()", epochDuration) && 
        paramMgr->getIntByXpath("//SimParams//numSims/text()", maxSteps) &&
        paramMgr->getIntByXpath("//SimConfig//maxFiringRate/text()", maxFiringRate) &&
        paramMgr->getIntByXpath("//SimConfig//maxSynapsesPerNeuron/text()", maxSynapsesPerNeuron) &&
        paramMgr->getLongByXpath("//Seed//value/text()", seed) &&
        paramMgr->getStringByXpath("//stateOutputFileName/text()", stateOutputFileName)){
        // TODO: should the output file name have its format checked?
        if (maxFiringRate < 0 || maxSynapsesPerNeuron < 0 ||
            width < 0 || height < 0 || epochDuration < 0 ||
            maxSteps < 0) {
            // TODO: throw exception?
            cerr << "ERROR: one or more global parameters " 
                 << "has an illegal negative value." << endl;
            return false;
        }
        totalNeurons = width * height;
        return true;
    }
    return false;
}

/*
 *  Prints out loaded parameters to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void SimulationInfo::printParameters(ostream &output) const
{
    cout << "poolsize x:" << width << " y:" << height
         //z dimmension is for future expansion and not currently supported
         //<< " z:" <<
         << endl;
    cout << "Simulation Parameters:\n";
    cout << "\tTime between growth updates (in seconds): " << epochDuration << endl;
    cout << "\tNumber of simulations to run: " << maxSteps << endl;
}
