/**
 *      @file XmlGrowthRecorder.h
 *
 *      @brief Header file for XmlGrowthRecorder.h
 */
//! An implementation for recording spikes history on xml file

/**
 ** \class XmlGrowthRecorder XmlGrowthRecorder.h "XmlGrowthRecorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The XmlGrowthRecorder provides a mechanism for recording neuron's layout, spikes history,
 ** and compile history information on xml file:
 **     -# neuron's locations, and type map,
 **     -# individual neuron's spike rate in epochs,
 **     -# network wide burstiness index data in 1s bins,
 **     -# network wide spike count in 10ms bins,
 **     -# individual neuron's radius history of every epoch.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#include "XmlRecorder.h"
#include "Model.h"
#include <fstream>

class XmlGrowthRecorder : public XmlRecorder
{
public:
    //! THe constructor and destructor
    XmlGrowthRecorder(const SimulationInfo* sim_info);
    ~XmlGrowthRecorder();

    /**
     * Init radii and rates history matrices with default values
     */
    virtual void initDefaultValues();

    /**
     * Init radii and rates history matrices with current radii and rates
     */
    virtual void initValues();

    /**
     * Get the current radii and rates vlaues
     */
    virtual void getValues();

    /**
     * Compile history information in every epoch
     *
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(IAllNeurons &neurons);

    /**
     * Writes simulation results to an output destination.
     *
     * @param  neurons the Neuron list to search from.
     **/
    virtual void saveSimData(const IAllNeurons &neurons);

private:
    // track firing rate
    CompleteMatrix ratesHistory;

    // track radii
    CompleteMatrix radiiHistory;
};

