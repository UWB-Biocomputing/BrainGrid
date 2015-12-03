/**
 *      @file ConnStatic.h
 *
 *      @brief The model of the small world network
 */

/**
 *
 * @class ConnStatic ConnStatic.h "ConnStatic.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The small-world networks are regular networks rewired to introduce increasing amounts
 * of disorder, which can be highly clustered, like regular lattices, yet have small
 * characterisic path length, like random graphs. 
 *
 * The structural properties of these graphs are quantified by their characteristic path
 * length \f$L(p)\f$ and clustering coefficient \f$C(p)\f$. Here \f$L\f$ is defined as the number of edges
 * in the shortest path between two vertices, average over all pairs of vertices.
 * The clustering coefficient \f$C(p)\f$ is defined as follows. Suppose that a vertex \f$v\f$ has \f$k_v\f$
 * neighbours; then at most \f$k_v (k_v - 1) / 2\f$ edges can exist between them (this occurs when
 * every neighbour of \f$v\f$ is connected to every other neighbour of \f$v\f$).
 * Let \f$C_v\f$ denote the fracion of these allowable edges that actually exist.
 * Define \f$C\f$ as the avarage of \f$C_v\f$ over all \f$v\f$ (Watts etal. 1998).
 *
 * We first create a regular network characterised by two parameters: number of maximum 
 * connections per neurons and connection radius threshold, then rewire it according 
 * to the small-world rewiring probability.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "SimulationInfo.h"
#include <vector>
#include <iostream>

using namespace std;

class ConnStatic : public Connections
{
    public:
        // TODO
        ConnStatic();
        virtual ~ConnStatic();

        static Connections* Create() { return new ConnStatic(); }

        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, AllNeurons *neurons, AllSynapses *synapses);
        virtual void cleanupConnections();
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void readConns(istream& input, const SimulationInfo *sim_info);
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info);
        virtual IRecorder* createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *sim_info);

    private:
        // number of maximum connections per neurons
        int nConnsPerNeuron;
        // Connection radius threshold
        BGFLOAT threshConnsRadius;
        // Small-world rewiring probability
        BGFLOAT pRewiring;

        struct DistDestNeuron
        {
            BGFLOAT dist;
            int dest_neuron;

            bool operator<(const DistDestNeuron& other) const
            {
                return (dist < other.dist);
            }
        };
};
