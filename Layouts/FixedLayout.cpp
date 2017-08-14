#include "FixedLayout.h"
#include "ParseParamError.h"
#include "Util.h"

FixedLayout::FixedLayout() : Layout()
{
}

FixedLayout::~FixedLayout()
{
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool FixedLayout::checkNumParameters()
{
    return (nParams >= 1);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool FixedLayout::readParameters(const TiXmlElement& element)
{
	cerr << "Attempting to parse a layout param" << endl;
    if (Layout::readParameters(element)) {
        // this parameter was already handled
        return true;
    }

    if (element.ValueStr().compare("LayoutFiles") == 0) {
	nParams++;
        return true;
    }
	
	/*
	*  Following statements exist because although the logic for reading the parameters
    *  contained in these elements is later in this function, the traversal of nodes
    *  causes this function to be called to handle the each of the elements individually. 
	*/
/*needed?
	if (element.ValueStr().compare("A") == 0){
		if(element.Parent()->ValueStr().compare("FixedLayoutParams") == 0){
			return true;
		}
	}
	if (element.ValueStr().compare("I") == 0){
		if(element.Parent()->ValueStr().compare("FixedLayoutParams") == 0){
			return true;
		}
	}
*/
/*
    // Parse fixed layout (overrides random layouts)
    if (element.ValueStr().compare("FixedLayoutParams") == 0) {
        const TiXmlNode* pNode = NULL;
        while ((pNode = element.IterateChildren(pNode)) != NULL) {
            string activeNListFileName;
            string inhNListFileName;
            string probedNListFileName;

            if (strcmp(pNode->Value(), "A") == 0) {
                getValueList(pNode->ToElement()->GetText(), &m_endogenously_active_neuron_list);
                num_endogenously_active_neurons = m_endogenously_active_neuron_list.size();
            } else if (strcmp(pNode->Value(), "I") == 0) {
                getValueList(pNode->ToElement()->GetText(), &m_inhibitory_neuron_layout);
            }
            else if (strcmp(pNode->Value(), "LayoutFiles") == 0)
            {
                if (pNode->ToElement()->QueryValueAttribute( "inhNListFileName", &inhNListFileName ) == TIXML_SUCCESS)
                {
                    TiXmlDocument simDoc( inhNListFileName.c_str( ) );
                    if (!simDoc.LoadFile( ))
                    {
                        cerr << "Failed loading positions of inhibitory neurons list file " << inhNListFileName << ":" << "\n\t"
                            << simDoc.ErrorDesc( ) << endl;
                        cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                        return false;
                    }
                    TiXmlNode* temp2 = NULL;
                    if (( temp2 = simDoc.FirstChildElement( "I" ) ) == NULL)
                    {
                        cerr << "Could not find <I> in positons of inhibitory neurons list file " << inhNListFileName << endl;
                        return false;
                    }
                    getValueList(temp2->ToElement()->GetText(), &m_inhibitory_neuron_layout);
                }
                if (pNode->ToElement()->QueryValueAttribute( "activeNListFileName", &activeNListFileName ) == TIXML_SUCCESS)
                {
                    TiXmlDocument simDoc( activeNListFileName.c_str( ) );
                    if (!simDoc.LoadFile( ))
                    {
                        cerr << "Failed loading positions of endogenously active neurons list file " << activeNListFileName << ":" << "\n\t"
                            << simDoc.ErrorDesc( ) << endl;
                        cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                        return false;
                    }
                    TiXmlNode* temp2 = NULL;
                    if (( temp2 = simDoc.FirstChildElement( "A" ) ) == NULL)
                    {
                        cerr << "Could not find <A> in positons of endogenously active neurons list file " << activeNListFileName << endl;
                        return false;
                    }
                    getValueList(temp2->ToElement()->GetText(), &m_endogenously_active_neuron_list);
                    num_endogenously_active_neurons = m_endogenously_active_neuron_list.size();
                }

                if (pNode->ToElement()->QueryValueAttribute( "probedNListFileName", &probedNListFileName ) == TIXML_SUCCESS) {
                    TiXmlDocument simDoc( probedNListFileName.c_str( ) );
                    if (!simDoc.LoadFile( ))
                    {
                        cerr << "Failed loading positions of probed neurons list file " << probedNListFileName << ":" << "\n\t"
                            << simDoc.ErrorDesc( ) << endl;
                        cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                        return false;
                    }
                    TiXmlNode* temp2 = NULL;
                    if (( temp2 = simDoc.FirstChildElement( "P" ) ) == NULL)
                    {
                        cerr << "Could not find <P> in positions of probed neurons list file " << probedNListFileName << endl;
                        return false;
                    }
                    getValueList(temp2->ToElement()->GetText(), &m_probed_neuron_list);
               }
            }
        }
        nParams++;
*/
    // Parse fixed layout (changed to utilize the Visiter Pattern provided by Tinyxml
    if (element.Parent()->ValueStr().compare("LayoutFiles") == 0) {
	if(element.ValueStr().compare("activeNListFileName") == 0){
	    const char* activeNListFileName = element.GetText();
	    if(activeNListFileName == NULL){
		return true;
	    }
            TiXmlDocument simDoc(activeNListFileName);
            if (!simDoc.LoadFile( ))
            {
                cerr << "Failed loading positions of endogenously active neurons list file " << activeNListFileName << ":" << "\n\t"
                    << simDoc.ErrorDesc( ) << endl;
                cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                return false;
            }
            TiXmlNode* temp2 = NULL;
            if (( temp2 = simDoc.FirstChildElement( "A" ) ) == NULL)
            {
                cerr << "Could not find <A> in positons of endogenously active neurons list file " << activeNListFileName << endl;
                return false;
            }
            getValueList(temp2->ToElement()->GetText(), &m_endogenously_active_neuron_list);
            num_endogenously_active_neurons = m_endogenously_active_neuron_list.size();

	    return true;
	}
	if(element.ValueStr().compare("inhNListFileName") == 0){
	    const char* inhNListFileName = element.GetText();
	    if(inhNListFileName == NULL){
		return true;
	    }
            TiXmlDocument simDoc(inhNListFileName);
            if (!simDoc.LoadFile( ))
            {
                cerr << "Failed loading positions of inhibitory neurons list file " << inhNListFileName << ":" << "\n\t"
                    << simDoc.ErrorDesc( ) << endl;
                cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                return false;
            }
            TiXmlNode* temp2 = NULL;
            if (( temp2 = simDoc.FirstChildElement( "I" ) ) == NULL)
            {
                cerr << "Could not find <I> in positions of inhibitory neurons list file " << inhNListFileName << endl;
                return false;
            }
            getValueList(temp2->ToElement()->GetText(), &m_inhibitory_neuron_layout);
	    return true;
	}
	if(element.ValueStr().compare("probedNListFileName") == 0){
	    const char* probedNListFileName = element.GetText();
	    if(probedNListFileName == NULL){
		return true;
	    }
            TiXmlDocument simDoc(probedNListFileName);
            if (!simDoc.LoadFile( ))
            {
                cerr << "Failed loading positions of probed neurons list file " << probedNListFileName << ":" << "\n\t"
                    << simDoc.ErrorDesc( ) << endl;
                cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                return false;
            }
            TiXmlNode* temp2 = NULL;
            if (( temp2 = simDoc.FirstChildElement( "P" ) ) == NULL)
            {
                cerr << "Could not find <P> in positions of probed neurons list file " << probedNListFileName << endl;
                return false;
            }
            getValueList(temp2->ToElement()->GetText(), &m_probed_neuron_list);
	    return true;
	}
    }

    return false;
}

/*
 *  Prints out all parameters of the layout to ostream.
 *  @param  output  ostream to send output to.
 */
void FixedLayout::printParameters(ostream &output) const
{
    Layout::printParameters(output);

    output << "Layout parameters:" << endl;

    cout << "\tEndogenously active neuron positions: ";
    for (BGSIZE i = 0; i < num_endogenously_active_neurons; i++) {
        output << m_endogenously_active_neuron_list[i] << " ";
    }

    cout << endl;

    cout << "\tInhibitory neuron positions: ";
    for (BGSIZE i = 0; i < m_inhibitory_neuron_layout.size(); i++) {
        output << m_inhibitory_neuron_layout[i] << " ";
    }

    cout << endl;

    cout << "\tProbed neuron positions: ";
    for (BGSIZE i = 0; i < m_probed_neuron_list.size(); i++) {
        output << m_probed_neuron_list[i] << " ";
    }

    output << endl;
}

/*
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @param  num_neurons number of the neurons to have in the type map.
 *  @return a flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
void FixedLayout::generateNeuronTypeMap(int num_neurons)
{
    Layout::generateNeuronTypeMap(num_neurons);

    int num_inhibitory_neurons = m_inhibitory_neuron_layout.size();
    int num_excititory_neurons = num_neurons - num_inhibitory_neurons;

    DEBUG(cout << "Total neurons: " << num_neurons << endl;)
    DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
    DEBUG(cout << "Excitatory Neurons: " << num_excititory_neurons << endl;)

    for (int i = 0; i < num_inhibitory_neurons; i++) {
        assert(m_inhibitory_neuron_layout.at(i) < num_neurons);
        neuron_type_map[m_inhibitory_neuron_layout.at(i)] = INH;
    }

    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/*
 *  Populates the starter map.
 *  Selects \e numStarter excitory neurons and converts them into starter neurons.
 *  @param  num_neurons number of neurons to have in the map.
 */
void FixedLayout::initStarterMap(const int num_neurons)
{
   Layout::initStarterMap(num_neurons);

    for (BGSIZE i = 0; i < num_endogenously_active_neurons; i++) {
        assert(m_endogenously_active_neuron_list.at(i) < num_neurons);
        starter_map[m_endogenously_active_neuron_list.at(i)] = true;
    }
}
