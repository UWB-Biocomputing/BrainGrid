#include "Connections.h"
#include "ParseParamError.h"
#include "AllSynapses.h"

/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
// TODO comment
const string Connections::MATRIX_TYPE = "complete";
// TODO comment
const string Connections::MATRIX_INIT = "const";
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * ------------------- CAUSE ------------------- *|
 * As simulations expand in size the number of
 * neurons in total increases exponentially. When
 * using a MATRIX_TYPE = “complete” the amount of
 * used memory increases by another order of magnitude.
 * Once enough memory is used no more memory can be
 * allocated and a “bsd_alloc” will be thrown.
 * The following members of the connection constructor
 * consume equally vast amounts of memory as the
 * simulation sizes grow:
 *      - W             - radii
 *      - rates         - dist2
 *      - delta         - dist
 *      - areai
 * ----------------- 1/25/14 ------------------- *|
 * Currently when running a simulation of sizes
 * equal to or greater than 100 * 100 the above
 * error is thrown. After some testing we have
 * determined that this is a hardware dependent
 * issue, not software. We are also looking into
 * switching matrix types from "complete" to
 * "sparce". If successful it is possible the
 * problematic matricies mentioned above will use
 * only 1/250 of their current space.
\* --------------------------------------------- */
Connections::Connections()
{
    xloc = NULL;
    yloc = NULL;
    dist2 = NULL;
    dist = NULL;
}

Connections::~Connections()
{
    cleanupConnections();
}

void Connections::setupConnections(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;

    xloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    yloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    dist2 = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    dist = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);

    // Initialize neuron locations 
    for (int i = 0; i < num_neurons; i++) {
        (*xloc)[i] = i % sim_info->width;
        (*yloc)[i] = i / sim_info->width;
    }   

    // calculate the distance between neurons
    for (int n = 0; n < num_neurons - 1; n++)
    {           
        for (int n2 = n + 1; n2 < num_neurons; n2++)
        {
            // distance^2 between two points in point-slope form
            (*dist2)(n, n2) = ((*xloc)[n] - (*xloc)[n2]) * ((*xloc)[n] - (*xloc)[n2]) +             
                ((*yloc)[n] - (*yloc)[n2]) * ((*yloc)[n] - (*yloc)[n2]);

            // both points are equidistant from each other
            (*dist2)(n2, n) = (*dist2)(n, n2);
        }
    }
 
    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    (*dist) = sqrt((*dist2)); 
}

void Connections::cleanupConnections()
{
    if (xloc != NULL) delete xloc;
    if (yloc != NULL) delete yloc;
    if (dist2 != NULL) delete dist2;
    if (dist != NULL) delete dist;

    xloc = NULL;
    yloc = NULL;
    dist2 = NULL;
    dist = NULL;
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool Connections::readParameters(const TiXmlElement& element)
{
    return true;
}

/**
 *  Prints out all parameters of the connections to ostream.
 *  @param  output  ostream to send output to.
 */
void Connections::printParameters(ostream &output) const
{
}

void Connections::readConns(istream& input, const SimulationInfo *sim_info)
{
}

void Connections::writeConns(ostream& output, const SimulationInfo *sim_info)
{
}

bool Connections::updateConnections(AllNeurons &neurons, const SimulationInfo *sim_info)
{
    return false;
}

#if !defined(USE_GPU)
/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void Connections::updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
}
#endif // !USE_GPU
