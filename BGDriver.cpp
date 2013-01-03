/**
 **	@file BGDriver.cpp
 **  The driver for braingrid.
 **  The driver performs the following steps:\n
 **  1) reads parameters from an xml file (specified as the first argument)\n
 **  2) creates the network\n
 **  3) launches the simulation\n
 **
 **  @authors Allan Ortiz and Cory Mayberry.
 **/

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <time.h>
#include <deque>
#include <ctime>

#include "global.h"
#include "tinyxml/tinyxml.h"
#include "Matrix/VectorMatrix.h"
#include "paramcontainer/ParamContainer.h"
#include "DynamicSpikingSynapse.h"
#include "LifNeuron.h"
#include "Network.h"

using namespace std;

// state file name
string stateOutputFileName;
string stateInputFileName;
vector<int> endogenouslyActiveNeuronLayout;
vector<int> inhibitoryNeuronLayout;

// memory dump file name
string memOutputFileName;
string memInputFileName;
bool fReadMemImage = false; // True if dumped memory image is read before starting simulation
bool fWriteMemImage = false; // True if dumped memory image is written after simulation

// Parameters for LSM
int poolsize[3]; // size of pool of neurons [x y z]
FLOAT frac_EXC; // Fraction of excitatory neurons
FLOAT Iinject[2]; // [A] Interval of constant injected current
FLOAT Inoise[2]; // [A] Interval of STD of (gaussian) noise current
FLOAT Vthresh[2]; // [V] Interval of firing threshold
FLOAT Vresting[2]; // [V] Interval of asymptotic voltage
FLOAT Vreset[2]; // [V] Interval of reset voltage
FLOAT Vinit[2]; // [V] Interval of initial membrance voltage
bool starter_flag = true; // true = use endogenously active neurons in simulation
FLOAT starter_neurons; // percent of endogenously active neurons
FLOAT starter_vthresh[2]; // default Vthresh is 15e-3
FLOAT starter_vreset[2]; // Interval of reset voltage
bool fFixedLayout; // True if a fixed layout has been provided; neuron positions are passed in endogenouslyActiveNeuronLayout and inhibitoryNeuronLayout

// Paramters for growth
FLOAT epsilon;
FLOAT beta;
FLOAT rho;
FLOAT targetRate; // Spikes/second
FLOAT maxRate; // = targetRate / epsilon;
FLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
// other neurons, when within their RFS.
FLOAT startRadius; // No need to wait a long time before RFs start to overlap

// Simulation Parameters
FLOAT Tsim; // Simulation time (s) (between growth updates)
int numSims; // Number of Tsim simulation to run
int maxFiringRate; // Maximum firing rate (only used by GPU version)
int maxSynapsesPerNeuron; //Maximum number of synapses per neuron (only used by GPU version)

// functions
void LoadSimParms(TiXmlElement*);
void SaveSimState(ostream &);
void printParams();
bool parseCommandLine(int argc, char* argv[]);
void getValueList(const string& valString, vector<int>* pList);

int main(int argc, char* argv[]) {

    DEBUG(cout << "reading parameters from xml file" << endl;)

	if (!parseCommandLine( argc, argv )) {
		cerr << "! ERROR: failed during command line parse" << endl;
		exit( -1 );
	}

	/*	verify that params were read correctly */
	DEBUG(printParams();)

	/* open input and output files */
	TiXmlDocument simDoc( stateInputFileName.c_str( ) );
	if (!simDoc.LoadFile( )) {
		cerr << "Failed loading simulation parameter file " << stateInputFileName << ":" << "\n\t"
				<< simDoc.ErrorDesc( ) << endl;
		return -1;
	}

	// aquire the in/out file
	ofstream state_out( stateOutputFileName.c_str( ) );
	ofstream memory_out;
	if (fWriteMemImage) {
		memory_out.open( memOutputFileName.c_str( ), ofstream::binary | ofstream::trunc );
	}
	ifstream memory_in;
	if (fReadMemImage) {
		memory_in.open( memInputFileName.c_str( ), ofstream::binary | ofstream::in );
	}

	// calculate the number of inhibitory, excitory, and endogenously active neurons
	int numNeurons = poolsize[0] * poolsize[1];
	int nInhNeurons = (int) ( ( 1.0 - frac_EXC ) * numNeurons + 0.5 );
	int nExcNeurons = numNeurons - nInhNeurons;
	int nStarterNeurons = 0;
	if (starter_flag) {
		nStarterNeurons = (int) ( starter_neurons * numNeurons + 0.5 );
	}

	// calculate their ratios, out of the whole
	FLOAT inhFrac = nInhNeurons / (FLOAT) numNeurons;
	FLOAT excFrac = nExcNeurons / (FLOAT) numNeurons;
	FLOAT startFrac = nStarterNeurons / (FLOAT) numNeurons;

	// create the network
	Network network( poolsize[0], poolsize[1], inhFrac, excFrac, startFrac, Iinject, Inoise, Vthresh, Vresting, Vreset,
			Vinit, starter_vthresh, starter_vreset, epsilon, beta, rho, targetRate, maxRate, minRadius, startRadius,
			DEFAULT_dt, state_out, memory_out, fWriteMemImage, memory_in, fReadMemImage, fFixedLayout, &endogenouslyActiveNeuronLayout, &inhibitoryNeuronLayout);

	time_t start_time, end_time;
	time(&start_time);

	network.simulate( Tsim, numSims, maxFiringRate, maxSynapsesPerNeuron );

	time(&end_time);
	double time_elapsed = difftime(end_time, start_time);
	double ssps = Tsim * numSims / time_elapsed;
	cout << "time simulated: " << Tsim * numSims << endl;
	cout << "time elapsed: " << time_elapsed << endl;
	cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;

	// close input and output files
	if (fWriteMemImage) {
		memory_out.close();		
	}
	if (fReadMemImage) {
		memory_in.close();
	}

	exit( EXIT_SUCCESS );

}

void printParams() {
	cout << "\nPrinting parameters...\n";
	cout << "frac_EXC:" << frac_EXC << " " << "starter_neurons:" << starter_neurons << endl;
	cout << "poolsize x:" << poolsize[0] << " y:" << poolsize[1] << " z:" << poolsize[2] << endl;
	cout << "Interval of constant injected current: [" << Iinject[0] << ", " << Iinject[1] << "]" << endl;
	cout << "Interval of STD of (gaussian) noise current: [" << Inoise[0] << ", " << Inoise[1] << "]\n";
	cout << "Interval of firing threshold: [" << Vthresh[0] << ", " << Vthresh[1] << "]\n";
	cout << "Interval of asymptotic voltage (Vresting): [" << Vresting[0] << ", " << Vresting[1] << "]\n";
	cout << "Interval of reset voltage: [" << Vreset[0] << ", " << Vreset[1] << "]\n";
	cout << "Interval of initial membrance voltage: [" << Vinit[0] << ", " << Vinit[1] << "]\n";
	cout << "Starter firing threshold: [" << starter_vthresh[0] << ", " << starter_vthresh[1] << "]\n";
	cout << "Starter reset threshold: [" << starter_vreset[0] << ", " << starter_vreset[1] << "]\n";
	cout << "Growth parameters: " << endl << "\tepsilon: " << epsilon << ", beta: " << beta << ", rho: " << rho
			<< ", targetRate: " << targetRate << ",\n\tminRadius: " << minRadius << ", startRadius: " << startRadius
			<< endl;
	cout << "Simulation Parameters:\n";
	cout << "\tTime between growth updates (in seconds): " << Tsim << endl;
	cout << "\tNumber of simulations to run: " << numSims << endl;

    if (fFixedLayout)
    {
        cout << "Layout parameters:" << endl;

        cout << "\tEndogenously active neuron positions: ";
        for (size_t i = 0; i < endogenouslyActiveNeuronLayout.size(); i++)
        {
            cout << endogenouslyActiveNeuronLayout[i] << " ";
        }
        cout << endl;

        cout << "\tInhibitory neuron positions: ";
        for (size_t i = 0; i < inhibitoryNeuronLayout.size(); i++)
        {
            cout << inhibitoryNeuronLayout[i] << " ";
        }
        cout << endl;
    }

	cout << "Done printing parameters" << endl;
}

void LoadSimParms(TiXmlElement* parms)
{
	TiXmlElement* temp = NULL;
    fFixedLayout = false;

	// Flag indicating that the variables were correctly read
	bool fSet = true;

	// Note that we just grab the first child with the right value;
	// multiple children with the same values are ignored. This might
	// not be as quick as iterating through the children and setting the
	// parameters as each one's element is found, but the code is
	// simpler this way and the performance penalty is insignificant.

	if (( temp = parms->FirstChildElement( "LsmParams" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("frac_EXC", &frac_EXC ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error frac_EXC" << endl;
		}
		if (temp->QueryFLOATAttribute("starter_neurons", &starter_neurons ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error starter_neurons" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing LsmParams" << endl;
	}

	if (( temp = parms->FirstChildElement( "PoolSize" ) ) != NULL) {
		if (temp->QueryIntAttribute( "x", &poolsize[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error x" << endl;
		}
		if (temp->QueryIntAttribute( "y", &poolsize[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error y" << endl;
		}
		if (temp->QueryIntAttribute( "z", &poolsize[2] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error z" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing PoolSize" << endl;
	}

	if (( temp = parms->FirstChildElement( "Iinject" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Iinject[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Iinject min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Iinject[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Iinject max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Iinject" << endl;
	}

	if (( temp = parms->FirstChildElement( "Inoise" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Inoise[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Inoise min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Inoise[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Inoise max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Inoise" << endl;
	}

	if (( temp = parms->FirstChildElement( "Vthresh" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Vthresh[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vthresh min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Vthresh[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vthresh min" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Vthresh" << endl;
	}

	if (( temp = parms->FirstChildElement( "Vresting" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Vresting[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vresting min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Vresting[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vresting max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Vresting" << endl;
	}

	if (( temp = parms->FirstChildElement( "Vreset" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Vreset[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vreset min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Vreset[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vreset max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Vreset" << endl;
	}

	if (( temp = parms->FirstChildElement( "Vinit" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &Vinit[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vinit min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &Vinit[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Vinit max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing Vinit" << endl;
	}

	if (( temp = parms->FirstChildElement( "starter_vthresh" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &starter_vthresh[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error starter_vthresh min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &starter_vthresh[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error starter_vthresh max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing starter_vthresh" << endl;
	}

	if (( temp = parms->FirstChildElement( "starter_vreset" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("min", &starter_vreset[0] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error starter_vreset min" << endl;
		}
		if (temp->QueryFLOATAttribute("max", &starter_vreset[1] ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error starter_vreset max" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing starter_vreset" << endl;
	}

	if (( temp = parms->FirstChildElement( "GrowthParams" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("epsilon", &epsilon ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error epsilon" << endl;
		}
		if (temp->QueryFLOATAttribute("beta", &beta ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error beta" << endl;
		}
		if (temp->QueryFLOATAttribute("rho", &rho ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error rho" << endl;
		}
		if (temp->QueryFLOATAttribute("targetRate", &targetRate ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error targetRate" << endl;
		}
		if (temp->QueryFLOATAttribute("minRadius", &minRadius ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error minRadius" << endl;
		}
		if (temp->QueryFLOATAttribute("startRadius", &startRadius ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error startRadius" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing GrowthParams" << endl;
	}

	if (( temp = parms->FirstChildElement( "SimParams" ) ) != NULL) {
		if (temp->QueryFLOATAttribute("Tsim", &Tsim ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error Tsim" << endl;
		}
		if (temp->QueryIntAttribute( "numSims", &numSims ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error numSims" << endl;
		}
		if (temp->QueryIntAttribute( "maxFiringRate", &maxFiringRate ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error maxFiringRate" << endl;
		}
		if (temp->QueryIntAttribute( "maxSynapsesPerNeuron", &maxSynapsesPerNeuron ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error maxSynapsesPerNeuron" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing SimParams" << endl;
	}

	if (( temp = parms->FirstChildElement( "OutputParams" ) ) != NULL) {
		if (temp->QueryValueAttribute( "stateOutputFileName", &stateOutputFileName ) != TIXML_SUCCESS) {
			fSet = false;
			cerr << "error stateOutputFileName" << endl;
		}
	} else {
		fSet = false;
		cerr << "missing OutputParams" << endl;
	}

    // Parse fixed layout (overrides random layouts)
	if ((temp = parms->FirstChildElement( "FixedLayout")) != NULL)
    {
        TiXmlNode* pNode = NULL;

        fFixedLayout = true;

        while ((pNode = temp->IterateChildren(pNode)) != NULL)
        {
            if (strcmp(pNode->Value(), "A") == 0)
            {
                getValueList(pNode->ToElement()->GetText(), &endogenouslyActiveNeuronLayout);
            }
            else if (strcmp(pNode->Value(), "I") == 0)
            {
                getValueList(pNode->ToElement()->GetText(), &inhibitoryNeuronLayout);
            }
        }
	}
    
    // Ideally, an error message would be output for each failed Query
	// above, but that's just too much code for me right now.
	if (!fSet) throw KII_exception( "Failed to initialize one or more simulation parameters; check XML" );
}

void getValueList(const string& valString, vector<int>* pList)
{
    std::istringstream valStream(valString);
    int i;

    // Parse integers out of the string and add them to a list
    while (valStream.good())
    {
        valStream >> i;
        pList->push_back(i);
    }
}

bool parseCommandLine(int argc, char* argv[])
{
	ParamContainer cl;
	cl.initOptions( false ); // don't allow unknown parameters
	cl.setHelpString( string( "The DCT growth modeling simulator\nUsage: " ) + argv[0] + " " );

#if defined(USE_GPU)
	if (( cl.addParam( "stateoutfile", 'o', ParamContainer::filename, "simulation state output filename" ) != ParamContainer::errOk ) 
			|| ( cl.addParam( "stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename" ) != ParamContainer::errOk )
			|| ( cl.addParam( "deviceid", 'd', ParamContainer::regular, "CUDA device id" ) != ParamContainer::errOk )
			|| ( cl.addParam( "meminfile", 'r', ParamContainer::filename, "simulation memory image input filename" ) != ParamContainer::errOk )
			|| ( cl.addParam( "memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename" ) != ParamContainer::errOk )) {
		cerr << "Internal error creating command line parser" << endl;
		return false;
	}
#else	// !USE_GPU
	if (( cl.addParam( "stateoutfile", 'o', ParamContainer::filename, "simulation state output filename" ) != ParamContainer::errOk ) 
			|| ( cl.addParam( "stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename" ) != ParamContainer::errOk ) 
			|| ( cl.addParam( "meminfile", 'r', ParamContainer::filename, "simulation memory image filename" ) != ParamContainer::errOk )
			|| ( cl.addParam( "memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename" ) != ParamContainer::errOk )) {
		cerr << "Internal error creating command line parser" << endl;
		return false;
	}
#endif // USE_GPU

	// Parse the command line
	if (cl.parseCommandLine( argc, argv ) != ParamContainer::errOk) {
		cl.dumpHelp( stderr, true, 78 );
		return false;
	}

	// Get the values
	stateOutputFileName = cl["stateoutfile"];
	stateInputFileName = cl["stateinfile"];
	memInputFileName = cl["meminfile"];
	memOutputFileName = cl["memoutfile"];
	if (!memInputFileName.empty()) {
		fReadMemImage = true;
	}
	if (!memOutputFileName.empty()) {
		fWriteMemImage = true;
	}
#if defined(USE_GPU)
	if ( EOF == sscanf(cl["deviceid"].c_str( ), "%d", &g_deviceId ) ) {
		g_deviceId = 0;
	}
#endif // USE_GPU

	TiXmlDocument simDoc( stateInputFileName.c_str( ) );
	if (!simDoc.LoadFile( )) {
		cerr << "Failed loading simulation parameter file " << stateInputFileName << ":" << "\n\t"
				<< simDoc.ErrorDesc( ) << endl;
		cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
		return false;
	}

	TiXmlElement* parms = NULL;

	if (( parms = simDoc.FirstChildElement( "SimParams" ) ) == NULL) {
		cerr << "Could not find <SimParms> in simulation parameter file " << stateInputFileName << endl;
		return false;
	}

	try {
		LoadSimParms( parms );
	} catch (KII_exception e) {
		cerr << "Failure loading simulation parameters from file " << stateInputFileName << ":\n\t" << e.what( )
				<< endl;
		return false;
	}
	return true;
}

