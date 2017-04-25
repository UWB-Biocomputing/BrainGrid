#include "DynamicLayout.h"
#include "ParseParamError.h"
#include "Util.h"

DynamicLayout::DynamicLayout() : Layout()
{
}

DynamicLayout::~DynamicLayout()
{
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool DynamicLayout::checkNumParameters()
{
    return (nParams >= 1);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool DynamicLayout::readParameters(const TiXmlElement& element)
{
    if (Layout::readParameters(element)) {
        // this parameter was already handled
        return true;
    }

    if (element. ValueStr().compare("LayoutFiles") == 0){
	nParams++;
	return true;
    }
/*
    if (element.ValueStr().compare("DynamicLayoutParams") == 0) {
        if (element.QueryFLOATAttribute("frac_EXC", &m_frac_excitatory_neurons) != TIXML_SUCCESS) {
            throw ParseParamError("frac_EXC", "Fraction Excitatory missing in XML.");
        }
        if (m_frac_excitatory_neurons < 0 || m_frac_excitatory_neurons > 1) {
            throw ParseParamError("frac_EXC", "Invalid range for a fraction.");
        }

        if (element.QueryFLOATAttribute("starter_neurons", &m_frac_starter_neurons) != TIXML_SUCCESS) {
            throw ParseParamError("starter_neurons", "Fraction endogenously active missing in XML.");
        }
        if (m_frac_starter_neurons < 0 || m_frac_starter_neurons > 1) {
            throw ParseParamError("starter_neurons", "Invalid range for a fraction.");
        }
        nParams++;
        return true;
    }
*/
    if(element.Parent()->ValueStr().compare("LayoutFiles") == 0){
	if(element.ValueStr().compare("frac_EXC") == 0){
	    m_frac_excitatory_neurons = atof(element.GetText());

            if (m_frac_excitatory_neurons < 0 || m_frac_excitatory_neurons > 1) {
                throw ParseParamError("frac_EXC", "Invalid range for a fraction.");
            }
	}
	else if(element.ValueStr().compare("starter_neurons") == 0){
	    m_frac_starter_neurons = atof(element.GetText());

            if (m_frac_starter_neurons < 0 || m_frac_starter_neurons > 1) {
                throw ParseParamError("starter_neurons", "Invalid range for a fraction.");
            }
	}
    }

    return false;
}

/*
 *  Prints out all parameters of the layout to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void DynamicLayout::printParameters(ostream &output) const
{
    Layout::printParameters(output);

    output << "frac_EXC:" << m_frac_excitatory_neurons
           << " starter_neurons:" << m_frac_starter_neurons
           << endl;
}

/*
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *
 *  @param  num_neurons number of the neurons to have in the type map.
 */
void DynamicLayout::generateNeuronTypeMap(int num_neurons)
{
    Layout::generateNeuronTypeMap(num_neurons);

    int num_excititory_neurons = (int) (m_frac_excitatory_neurons * num_neurons + 0.5);
    int num_inhibitory_neurons = num_neurons - num_excititory_neurons;

    DEBUG(cout << "Total neurons: " << num_neurons << endl;)
    DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
    DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)

    DEBUG(cout << endl << "Randomly selecting inhibitory neurons..." << endl;)

    int* rg_inhibitory_layout = new int[num_inhibitory_neurons];

    for (int i = 0; i < num_inhibitory_neurons; i++) {
        rg_inhibitory_layout[i] = i;
    }

    for (int i = num_inhibitory_neurons; i < num_neurons; i++) {
        int j = static_cast<int>(rng() * num_neurons);
        if (j < num_inhibitory_neurons) {
            rg_inhibitory_layout[j] = i;
        }
    }

    for (int i = 0; i < num_inhibitory_neurons; i++) {
        neuron_type_map[rg_inhibitory_layout[i]] = INH;
    }
    delete[] rg_inhibitory_layout;

    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/*
 *  Populates the starter map.
 *  Selects num_endogenously_active_neurons excitory neurons 
 *  and converts them into starter neurons.
 *
 *  @param  num_neurons number of neurons to have in the map.
 */
void DynamicLayout::initStarterMap(const int num_neurons)
{
    Layout::initStarterMap(num_neurons);

    num_endogenously_active_neurons = (BGSIZE) (m_frac_starter_neurons * num_neurons + 0.5);
    BGSIZE starters_allocated = 0;

    DEBUG(cout << "\nRandomly initializing starter map\n";);
    DEBUG(cout << "Total neurons: " << num_neurons << endl;)
    DEBUG(cout << "Starter neurons: " << num_endogenously_active_neurons << endl;)

    // randomly set neurons as starters until we've created enough
    while (starters_allocated < num_endogenously_active_neurons) {
        // Get a random integer
        int i = static_cast<int>(rng.inRange(0, num_neurons));

        // If the neuron at that index is excitatory and a starter map
        // entry does not already exist, add an entry.
        if (neuron_type_map[i] == EXC && starter_map[i] == false) {
            starter_map[i] = true;
            starters_allocated++;
            DEBUG(cout << "allocated EA neuron at random index [" << i << "]" << endl;);
        }
    }

    DEBUG(cout <<"Done randomly initializing starter map\n\n";)
}
