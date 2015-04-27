#pragma once
#include "GPUSpikingModel.h"
#include "AllIFNeurons.h"

class LIFGPUModel : public GPUSpikingModel {

public:
    LIFGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
    virtual ~LIFGPUModel();

protected:
    virtual void advanceNeurons(const SimulationInfo *sim_info);

private:

    /*----------------------------------------------*\
    |  Member variables
    \*----------------------------------------------*/

};
