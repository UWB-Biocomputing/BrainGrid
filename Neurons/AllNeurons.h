/**
 *      @file AllNeurons.h
 *
 *      @brief A container of the base class of all neuron data
 */

/**
 ** @class AllNeurons AllNeurons.h "AllNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of the base class of all neuron data.
 **
 ** The class uses a data-centric structure, which utilizes a structure as the containers of
 ** all neuron.
 **
 ** The container holds neuron parameters of all neurons.
 ** Each kind of neuron parameter is stored in a 1D array, of which length
 ** is number of all neurons. Each array of a neuron parameter is pointed by a
 ** corresponding member variable of the neuron parameter in the class.
 **
 ** This structure was originally designed for the GPU implementation of the
 ** simulator, and this refactored version of the simulator simply uses that design for
 ** all other implementations as well. This is to simplify transitioning from
 ** single-threaded to multi-threaded.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/

#pragma once

using namespace std;

#include "Global.h"
#include "IAllNeurons.h"
#include "SimulationInfo.h"
#include "Layout.h"
#include "AllNeuronsProps.h"

class AllNeurons : public IAllNeurons
{
    public:
        CUDA_CALLABLE AllNeurons();
        CUDA_CALLABLE virtual ~AllNeurons();

        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        virtual void cleanupNeurons();

        /**
         *  Assignment operator: copy neurons parameters.
         *
         *  @param  r_neurons  Neurons class object to copy from.
         */
        virtual IAllNeurons &operator=(const IAllNeurons &r_neurons);

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info);

#if defined(USE_GPU)
        /**
         * Delete an AllNeurons class object in device
         *
         * @param pAllNeurons_d    Pointer to the AllNeurons object to be deleted in device.
         */
        virtual void deleteAllNeuronsInDevice(IAllNeurons* pAllNeurons_d);

        /**
         *  Set neurons properties.
         *
         *  @param  pAllNeuronsProps  Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void setNeuronsProps(void *pAllNeuronsProps);
#endif // USE_GPU

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters();

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element);

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const;

        /**
         *  Creates all the Neurons and assigns initial data for them.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  clr_info    ClusterInfo class to read information from.
         */
        virtual void createAllNeurons(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info);

    public:
        /**
         * Pointer to the neurons property data.
         */
        class AllNeuronsProps* m_pNeuronsProps;
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void deleteAllNeuronsDevice(IAllNeurons *pAllNeurons);

#endif // USE_GPU
