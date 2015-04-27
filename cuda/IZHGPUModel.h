#pragma once
#include "GPUSpikingModel.h"
#include "AllIZHNeurons.h"

class IZHGPUModel : public GPUSpikingModel {

public:
    IZHGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
    virtual ~IZHGPUModel();

protected:
    virtual void advanceNeurons(const SimulationInfo *sim_info);

private:

    /*----------------------------------------------*\
    |  Member variables
    \*----------------------------------------------*/

};
