#include "Connections.h"
#include "ParseParamError.h"

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
    W = NULL;
    radii = NULL;
    rates = NULL;
    dist2 = NULL;
    delta = NULL;
    dist = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;
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
    W = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    radii = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, m_growth.startRadius);
    rates = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0);
    dist2 = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    delta = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    dist = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    area = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    outgrowth = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    deltaR = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);

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

    // Init connection frontier distance change matrix with the current distances
    (*delta) = (*dist);
}

void Connections::cleanupConnections()
{
    if (xloc != NULL) delete xloc;
    if (yloc != NULL) delete yloc;
    if (W != NULL) delete W;
    if (radii != NULL) delete radii;
    if (rates != NULL) delete rates;
    if (dist2 != NULL) delete dist2;
    if (delta != NULL) delete delta;
    if (dist != NULL) delete dist;
    if (area != NULL) delete area;
    if (outgrowth != NULL) delete outgrowth;
    if (deltaR != NULL) delete deltaR;

    xloc = NULL;
    yloc = NULL;
    W = NULL;
    radii = NULL;
    rates = NULL;
    dist2 = NULL;
    delta = NULL;
    dist = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool Connections::readParameters(const TiXmlElement& element)
{
    if (element.ValueStr().compare("GrowthParams") == 0) {
        if (element.QueryFLOATAttribute("epsilon", &m_growth.epsilon) != TIXML_SUCCESS) {
                throw ParseParamError("epsilon", "Growth param 'epsilon' missing in XML.");
        }
        if (m_growth.epsilon < 0) {
                throw ParseParamError("epsilon", "Invalid negative Growth param 'epsilon' value.");
        }

        if (element.QueryFLOATAttribute("beta", &m_growth.beta) != TIXML_SUCCESS) {
                throw ParseParamError("beta", "Growth param 'beta' missing in XML.");
        }
        if (m_growth.beta < 0) {
                throw ParseParamError("beta", "Invalid negative Growth param 'beta' value.");
        }

        if (element.QueryFLOATAttribute("rho", &m_growth.rho) != TIXML_SUCCESS) {
                throw ParseParamError("rho", "Growth param 'rho' missing in XML.");
        }
        if (m_growth.rho < 0) {
                throw ParseParamError("rho", "Invalid negative Growth param 'rho' value.");
        }

        //check if 'beta' is erroneous info
        if (element.QueryFLOATAttribute("targetRate", &m_growth.targetRate) != TIXML_SUCCESS) {
                throw ParseParamError("targetRate", "Growth targetRate 'beta' missing in XML.");
        }
        if (m_growth.targetRate < 0) {
                throw ParseParamError("targetRate", "Invalid negative Growth targetRate.");
        }

        if (element.QueryFLOATAttribute("minRadius", &m_growth.minRadius) != TIXML_SUCCESS) {
                throw ParseParamError("minRadius", "Growth minRadius 'beta' missing in XML.");
        }
        if (m_growth.minRadius < 0) {
                throw ParseParamError("minRadius", "Invalid negative Growth minRadius.");
        }

        if (element.QueryFLOATAttribute("startRadius", &m_growth.startRadius) != TIXML_SUCCESS) {
                throw ParseParamError("startRadius", "Growth startRadius 'beta' missing in XML.");
        }
        if (m_growth.startRadius < 0) {
                throw ParseParamError("startRadius", "Invalid negative Growth startRadius.");
        }
    }

    // initial maximum firing rate
    m_growth.maxRate = m_growth.targetRate / m_growth.epsilon;
        
    return true;
}

/**
 *  Prints out all parameters of the connections to ostream.
 *  @param  output  ostream to send output to.
 */
void Connections::printParameters(ostream &output) const
{
    output << "Growth parameters: " << endl
           << "\tepsilon: " << m_growth.epsilon
           << ", beta: " << m_growth.beta
           << ", rho: " << m_growth.rho
           << ", targetRate: " << m_growth.targetRate << "," << endl
           << "\tminRadius: " << m_growth.minRadius
           << ", startRadius: " << m_growth.startRadius
           << endl;

}

void Connections::readConns(istream& input, const SimulationInfo *sim_info)
{
    // read the radii
    for (int i = 0; i < sim_info->totalNeurons; i++) {
            input >> (*radii)[i]; input.ignore();
    }

    // read the rates
    for (int i = 0; i < sim_info->totalNeurons; i++) {
            input >> (*rates)[i]; input.ignore();
    }
}

void Connections::writeConns(ostream& output, const SimulationInfo *sim_info)
{
    // write the final radii
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        output << (*radii)[i] << ends;
    }

    // write the final rates
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        output << (*rates)[i] << ends;
    }
}

void Connections::updateConns(AllNeurons &neurons, const SimulationInfo *sim_info)
{
    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        // Calculate firing rate
        assert(neurons.spikeCount[i] < max_spikes);
        (*rates)[i] = neurons.spikeCount[i] / sim_info->epochDuration;
    }

    // compute neuron radii change and assign new values
    (*outgrowth) = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - *rates / m_growth.maxRate) / m_growth.beta));
    (*deltaR) = sim_info->epochDuration * m_growth.rho * *outgrowth;
    (*radii) += (*deltaR);
}

/**
 *  Update the distance between frontiers of Neurons.
 *  @param  num_neurons in the simulation to update.
 */
void Connections::updateFrontiers(const int num_neurons)
{
    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < num_neurons - 1; unit++) {
        for (int i = unit + 1; i < num_neurons; i++) {
            (*delta)(unit, i) = (*dist)(unit, i) - ((*radii)[unit] + (*radii)[i]);
            (*delta)(i, unit) = (*delta)(unit, i);
        }
    }
}

/**
 *  Update the areas of overlap in between Neurons.
 *  @param  num_neurons number of Neurons to update.
 */
void Connections::updateOverlap(BGFLOAT num_neurons)
{
    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_neurons; j++) {
                (*area)(i, j) = 0.0;

                if ((*delta)(i, j) < 0) {
                        BGFLOAT lenAB = (*dist)(i, j);
                        BGFLOAT r1 = (*radii)[i];
                        BGFLOAT r2 = (*radii)[j];

                    if (lenAB + min(r1, r2) <= max(r1, r2)) {
                        (*area)(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                        logFile << "Completely overlapping (i, j, r1, r2, area): "
                            << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                        } else {
                                // Partially overlapping unit
                                BGFLOAT lenAB2 = (*dist2)(i, j);
                                BGFLOAT r12 = r1 * r1;
                                BGFLOAT r22 = r2 * r2;

                                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                                BGFLOAT angCBA = acos(cosCBA);
                                BGFLOAT angCBD = 2.0 * angCBA;

                                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                                BGFLOAT angCAB = acos(cosCAB);
                                BGFLOAT angCAD = 2.0 * angCAB;

                                (*area)(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                        }
                }
        }
    }
}

