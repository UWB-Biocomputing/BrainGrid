#include "Layout.h"
#include "ParseParamError.h"
#include "Util.h"

Layout::Layout() :
    num_endogenously_active_neurons(0),
    nParams(0),
    m_grid_layout(true)
{
    xloc = NULL;
    yloc = NULL;
    dist2 = NULL;
    dist = NULL;
    neuron_type_map = NULL;
    starter_map = NULL;
}

Layout::~Layout()
{
    if (xloc != NULL) delete xloc;
    if (yloc != NULL) delete yloc;
    if (dist2 != NULL) delete dist2;
    if (dist != NULL) delete dist;
    if (neuron_type_map != NULL) delete[] neuron_type_map;
    if (starter_map != NULL) delete[] starter_map;

    xloc = NULL;
    yloc = NULL;
    dist2 = NULL;
    dist = NULL;
    neuron_type_map = NULL;
    starter_map = NULL;
}

/*
 *  Setup the internal structure of the class. 
 *  Allocate memories to store all layout state.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void Layout::setupLayout(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;

    xloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    yloc = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    dist2 = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    dist = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);

    // Initialize neuron locations
    initNeuronsLocs(sim_info);

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

    neuron_type_map = new neuronType[num_neurons];
    starter_map = new bool[num_neurons];
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool Layout::readParameters(const TiXmlElement& element)
{
    return false;
}

/*
 *  Prints out all parameters of the layout to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void Layout::printParameters(ostream &output) const
{
}

/*
 *  Creates a neurons type map.
 *
 *  @param  num_neurons number of the neurons to have in the type map.
 */
void Layout::generateNeuronTypeMap(int num_neurons)
{
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);

    for (int i = 0; i < num_neurons; i++) {
        neuron_type_map[i] = EXC;
    }
}

/*
 *  Populates the starter map.
 *  Selects num_endogenously_active_neurons excitory neurons and converts them into starter neurons.
 *
 *  @param  num_neurons number of neurons to have in the map.
 */
void Layout::initStarterMap(const int num_neurons)
{
    for (int i = 0; i < num_neurons; i++) {
        starter_map[i] = false;
    }
}

/*
 *  Returns the type of synapse at the given coordinates
 *
 *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
 *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
 *  @return type of the synapse.
 */
synapseType Layout::synType(const int src_neuron, const int dest_neuron)
{
    if ( neuron_type_map[src_neuron] == INH && neuron_type_map[dest_neuron] == INH )
        return II;
    else if ( neuron_type_map[src_neuron] == INH && neuron_type_map[dest_neuron] == EXC )
        return IE;
    else if ( neuron_type_map[src_neuron] == EXC && neuron_type_map[dest_neuron] == INH )
        return EI;
    else if ( neuron_type_map[src_neuron] == EXC && neuron_type_map[dest_neuron] == EXC )
        return EE;

    return STYPE_UNDEF;
}

/*
 *  Initialize the location maps (xloc and yloc).
 *
 *  @param sim_info   SimulationInfo class to read information from.
 */
void Layout::initNeuronsLocs(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;

    // Initialize neuron locations
    if (m_grid_layout) {
        // grid layoug
        for (int i = 0; i < num_neurons; i++) {
            (*xloc)[i] = i % sim_info->width;
            (*yloc)[i] = i / sim_info->width;
        }
    } else {
        // random layout
        for (int i = 0; i < num_neurons; i++) {
            (*xloc)[i] = rng.inRange(0, sim_info->width);
            (*yloc)[i] = rng.inRange(0, sim_info->height);
        }
    }
}
