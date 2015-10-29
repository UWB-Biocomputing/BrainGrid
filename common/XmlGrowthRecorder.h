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
 ** The XmlGrowthRecorder provides a mechanism for recording spikes history,
 ** and compile history information on xml file:
 ** 	(1) individual neuron's spike rate in epochs,
 **	(2) burstiness index data in 1s bins,
 **     (3) network wide spike count in 10ms bins.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
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
    XmlGrowthRecorder(IModel *model, const SimulationInfo* sim_info);
    ~XmlGrowthRecorder();

    /*
     * Init radii and rates history matrices with default values
     */
    virtual void initDefaultValues();

    /*
     * Init radii and rates history matrices with current radii and rates
     */
    virtual void initValues();

    /*
     * Get the current radii and rates vlaues
     */
    virtual void getValues();

    /**
     * Compile history information in every epoch
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(AllNeurons &neurons);

    /**
     * Save current simulation state to XML
     * @param  neurons the Neuron list to search from.
     **/
    virtual void saveSimState(const AllNeurons &neurons);

private:
    // track radii
    CompleteMatrix radiiHistory;

    // track firing rate
    CompleteMatrix ratesHistory;
};

