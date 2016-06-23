/**
 *      @file DynamicLayout.h
 *
 *      @brief The DynamicLayout class defines the layout of neurons in neunal networks
 */

/**
 *
 * @class DynamicLayout DynamicLayout.h "DynamicLayout.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The DynamicLayout class maintains neurons locations (x, y coordinates), 
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons), and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 * The DynamicLayout class generates layout information dynamically.
 */

#pragma once

#include "Layout.h"

using namespace std;

class DynamicLayout : public Layout
{
    public:
        DynamicLayout();
        virtual ~DynamicLayout();

        static Layout* Create() { return new DynamicLayout(); }

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
         *  Creates a randomly ordered distribution with the specified numbers of neuron types.
         *
         *  @param  num_neurons number of the neurons to have in the type map.
         */
        virtual void generateNeuronTypeMap(int num_neurons);

        /**
         *  Populates the starter map.
         *  Selects num_endogenously_active_neurons excitory neurons 
         *  and converts them into starter neurons.
         *
         *  @param  num_neurons number of neurons to have in the map.
         */
        virtual void initStarterMap(const int num_neurons);

    private:
        //! Fraction of endogenously active neurons.
        BGFLOAT m_frac_starter_neurons;

        //! Fraction of exitatory neurons.
        BGFLOAT m_frac_excitatory_neurons;
};

