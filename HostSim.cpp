/**
 *	\file HostSim.cpp
 *	
 *	\author Fumitaka Kawasaki
 *
 *	\brief A super class of MultiThreadedSim and SingleThreadedSim classes.
 */
#include "HostSim.h"

/**
 * Allocate all matixes that are used for updating the network.
 * @param[in] psi       Pointer to the simulation information.
 * @post All data structures are allocated. 
 */
HostSim::HostSim(SimulationInfo* psi) :
	W("complete", "const", psi->cNeurons, psi->cNeurons, 0),
	radii("complete", "const", 1, psi->cNeurons, psi->startRadius),
	rates("complete", "const", 1, psi->cNeurons, 0),
	dist2("complete", "const", psi->cNeurons, psi->cNeurons),
	delta("complete", "const", psi->cNeurons, psi->cNeurons),
	dist("complete", "const", psi->cNeurons, psi->cNeurons),
	area("complete", "const", psi->cNeurons, psi->cNeurons, 0),
	outgrowth("complete", "const", 1, psi->cNeurons),
	deltaR("complete", "const", 1, psi->cNeurons)
{ 
}

/**
* Destructor
*
*/
HostSim::~HostSim() 
{ 
}

/**
 * Compute dist2, dist and delta.
 * @param[in] psi       Pointer to the simulation information.  
 * @param[in] xloc      X location of neurons.
 * @param[in] yloc      Y location of neurons.
 */
void HostSim::init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc)
{
// MODEL DEPENDENT
    // calculate the distance between neurons
    for (int n = 0; n < psi->cNeurons - 1; n++)
    {
        for (int n2 = n + 1; n2 < psi->cNeurons; n2++)
        {
            // distance^2 between two points in point-slope form
            dist2(n, n2) = (xloc[n] - xloc[n2]) * (xloc[n] - xloc[n2]) +
                (yloc[n] - yloc[n2]) * (yloc[n] - yloc[n2]);

            // both points are equidistant from each other
            dist2(n2, n) = dist2(n, n2);
        }
    }

    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    dist = sqrt(dist2);

    // Init connection frontier distance change matrix with the current distances
    delta = dist;
}

/**
 * Terminate process
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSim::term(SimulationInfo* psi)
{
}

/**
 * Set initial radii data
 * @param[in] newRadii	Radii data to set
 * MODEL DEPENDENT
 */
void HostSim::initRadii(VectorMatrix& newRadii)
{
    radii = newRadii;
}

/**
 * Returns a type of Neuron to be used in the Network
 */
INeuron* HostSim::returnNeuron()
{
	return new LifNeuron();
}

/**
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 * @param[in] source_x	X location of source.
 * @param[in] source_y	Y location of source.
 * @param[in] dest_x	X location of destination.
 * @param[in] dest_y	Y location of destination.
 * @return reference to a DSS
 */
ISynapse* HostSim::addSynapse(SimulationInfo* psi, int source_x, int source_y, int dest_x, int dest_y)
{
    // locate summation point
    BGFLOAT* sp = &(psi->pSummationMap[dest_x + dest_y * psi->width]);

    // determine the synapse type
    synapseType type = synType(psi, Coordinate(source_x, source_y), Coordinate(dest_x, dest_y));

    // create synapse;
    DynamicSpikingSynapse* syn = 
		new DynamicSpikingSynapse(source_x, source_y, dest_x, dest_y, *sp, DEFAULT_delay_weight, psi->deltaT, type);

    // add it to the list
    psi->rgSynapseMap[source_x + source_y * psi->width].push_back(syn);

    return psi->rgSynapseMap[source_x + source_y * psi->width].back();
}

/**
 * Returns the type of synapse at the given coordinates
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] a		Source coordinate.
 * @param[in] b		Destination coordinate.
 * @return type of synapse at the given coordinate or -1 on error
 */
synapseType HostSim::synType(SimulationInfo* psi, Coordinate a, Coordinate b)
{
    neuronType a_type = psi->rgNeuronTypeMap[a.x + a.y * psi->width];
    neuronType b_type = psi->rgNeuronTypeMap[b.x + b.y * psi->width];
    if (a_type == INH && b_type == INH)
        return II;
    else if (a_type == INH && b_type == EXC)
        return IE;
    else if (a_type == EXC && b_type == INH)
        return EI;
    else if (a_type == EXC && b_type == EXC)
        return EE;

    assert(false);
    return STYPE_UNDEF;
}

/**
 * Return 1 if originating neuron is excitatory, -1 otherwise.
 * @param[in] synapseType I to I, I to E, E to I, or E to E
 * @return 1 or -1
 */
int HostSim::synSign(synapseType t)
{
    switch (t)
    {
    case II:
    case IE:
        return -1;
    case EI:
    case EE:
        return 1;
    case STYPE_UNDEF:
    default:
        assert(false);
    }

    return 0;
}

/**
 * Print network radii to console
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] networkRadii	Array to store neuron radii.
 */
void HostSim::printNetworkRadii(SimulationInfo* psi, VectorMatrix networkRadii) const
{
    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < psi->height; y++)
    {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < psi->width; x++)
        {
            switch (psi->rgNeuronTypeMap[x + y * psi->width])
            {
            case EXC:
                if (psi->rgEndogenouslyActiveNeuronMap[x + y * psi->width])
                    ss << "s";
                else
                    ss << "e";
                break;
            case INH:
                ss << "i";
                break;
            case NTYPE_UNDEF:
                assert(false);
            }

            ss << " " << networkRadii[x + y * psi->width];
            ss << " " << networkRadii[x + y * psi->width];

            if (x + 1 < psi->width)
            {
                ss.width(2);
                ss << "|";
                ss.width(2);
            }
        }

        ss << endl;

        for (int i = ss.str().length() - 1; i >= 0; i--)
        {
            ss << "_";
        }

        ss << endl;
        cout << ss.str();
    }
}

