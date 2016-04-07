/**
 *      @file FixedLayout.h
 *
 *      @brief The Layout class defines the layout of neurons in neunal networks
 */

/**
 *
 * @class FixedLayout FixedLayout.h "FixedLayout.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The FixedLayout class maintains neurons locations (x, y coordinates), 
 * distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons), and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 * The FixedLayout class reads all layout information from parameter description file.
 *
 */

#pragma once

#include "Layout.h"

using namespace std;

class FixedLayout : public Layout
{
    public:
        FixedLayout();
        virtual ~FixedLayout();

        static Layout* Create() { return new FixedLayout(); }

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
         *  Creates a neurons type map.
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

    protected:

    private:
};

