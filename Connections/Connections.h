/**
 *      @file Connections.h
 *
 *      @brief The base class of all connections classes
 */

/**
 *
 * @class Connections Connections.h "Connections.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * A placeholder to define connections of neunal networks.
 * In neunal networks, neurons are connected through synapses where messages are exchanged.
 * The strength of connections is characterized by synapse's weight. 
 * The connections classes define topologies, the way to connect neurons,  
 * and dynamics, the way to change connections as time elapses, of the networks. 
 * 
 * Connections can be either static or dynamic. The static connectons are ones where
 * connections are established at initialization and never change. 
 * The dynamic connections can be changed as the networks evolve, so in the dynamic networks
 * synapses will be created, deleted, or their weight will be modifed.  
 *
 * Connections classes may maintains intra-epoch state of connections in the network. 
 * This includes history and parameters that inform how new connections are made during growth.
 * Therefore, connections classes will have customized recorder classes, and provide
 * a function to craete the recorder class.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Layout.h"
#include "IRecorder.h"
#include <vector>
#include <iostream>

using namespace std;

class IModel;
class Cluster;

class Connections
{
    public:
        Connections();
        virtual ~Connections();

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo) = 0;

        /**
         *  Cleanup the class (deallocate memories).
         */
        virtual void cleanupConnections() = 0;

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters() = 0;

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element) = 0;

        /**
         *  Prints out all parameters of the connections to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const = 0;

        /**
         *  Reads the intermediate connection status from istream.
         *
         *  @param  input    istream to read status from.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        virtual void deserialize(istream& input, const SimulationInfo *sim_info) = 0;

        /**
         *  Writes the intermediate connection status to ostream.
         *
         *  @param  output   ostream to write status to.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info) = 0;

        /**
         *  Update the connections status in every epoch.
         *  By default, this method does nothing. Override in a subclass to
         *  implement desired functionality.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
   virtual void updateConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo) {}

        /**
         *  Creates a recorder class object for the connection.
         *  This function tries to create either Xml recorder or
         *  Hdf5 recorder based on the extension of the file name.
         *
         *  @param  simInfo              SimulationInfo to refer from.
         *  @return Pointer to the recorder class object.
         */
        virtual IRecorder* createRecorder(const SimulationInfo *sim_info) = 0;

    protected:
        //!  Number of parameters read.
        int nParams;
};

