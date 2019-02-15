/**
 *      @file AllIFNeurons.h
 *
 *      @brief A container of all Integate and Fire (IF) neuron data
 */

/** 
 ** @authors Aaron Oziel, Sean Blackbourn
 **
 ** @class AllIFNeurons AllIFNeurons.h "AllIFNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of all Integate and Fire (IF) neuron data.
 ** This is the base class of all Integate and Fire (IF) neuron classes.
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

#include "Global.h"
#include "AllSpikingNeurons.h"
#include "AllIFNeuronsProps.h"

class AllIFNeurons : public AllSpikingNeurons
{
    public:
        AllIFNeurons();
        virtual ~AllIFNeurons();

        /**
         *  Create and setup neurons properties.
         */
        virtual void createNeuronsProps();
};

