
#if defined(BOOST_PYTHON)

#include "XmlGrowthRecorder.h"
#include "Simulator.h"
#if defined(USE_HDF5)
#include "Hdf5GrowthRecorder.h"
#endif // USE_HDF5
#if defined(USE_GPU)
    #include "GPUSpikingCluster.h"
#else // USE_GPU
    #include "SingleThreadedCluster.h"
#endif // USE_GPU
#include "AllLIFNeurons.h"
#include "AllIZHNeurons.h"
#include "AllDSSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "ConnStatic.h"
#include "ConnGrowth.h"
#include "FixedLayout.h"
#include "DynamicLayout.h"

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/foreach.hpp>

#include "array_ref.h"
#include "array_indexing_suite.h"

extern bool LoadAllParameters(SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);
extern bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo);
extern IRecorder* createRecorder(const SimulationInfo *simInfo);

using namespace boost::python;

/*
 * @brief Transfer ownership to a Python object.  If the transfer fails,
 *        then object will be destroyed and an exception is thrown.
*/
template <typename T>
boost::python::object transfer_to_python(T* t)
{
  // Transfer ownership to a smart pointer, allowing for proper cleanup
  // incase Boost.Python throws.
  std::unique_ptr<T> ptr(t);

  // Use the manage_new_object generator to transfer ownership to Python.
  namespace python = boost::python;
  typename python::manage_new_object::apply<T*>::type converter;

  // Transfer ownership to the Python handler and release ownership
  // from C++.
  python::handle<> handle(converter(*ptr));
  ptr.release();

  return python::object(handle);
}

/*
 *  Convert C++ vector to Python list
 *  Assume that the vector contains C++ object pointers.
 *
 *  @param vector  C++ vector.
 *  @returns       Python list.
 */
template <class T>
inline
void std_vector_to_py_list(std::vector<T> vector, boost::python::list &list) {
    typename std::vector<T>::iterator iter;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(transfer_to_python(*iter));
    }
}

/*
 *  Convert Python list to C++ vector
 *  Assume that the vector contains C++ object pointers.
 *
 *  @param o       Python list
 *  @param vector  C++ vector.
 */
template<typename T>
void python_to_vector(boost::python::object o, vector<T>* v) {
    stl_input_iterator<T> begin(o);
    stl_input_iterator<T> end;
    v->clear();
    v->insert(v->end(), begin, end);
}

typedef vector<int> int_vector;

/*
 *  Python lvalue converter
 *  Boost Python converter that allowed to convert C++ vector to Python list.
 *  This converter will be specified in to_python_converter<>.
 */
template<typename T_>
class vector_to_pylist_converter {
public:
    typedef T_ native_type;

    static PyObject* convert(native_type const& v) {
        namespace py = boost::python;
        py::list retval;
        BOOST_FOREACH(typename boost::range_value<native_type>::type i, v)
        {
            retval.append(py::object(i));
        }
        return py::incref(retval.ptr());
    }
};

/*
 *  Load parameters from a file.
 *
 *  @param  simInfo       SimulationInfo class to read information from.
 *  @param  cluster       List to store Cluster class objects to be created.
 *  @param  clusterInfo   List to store ClusterInfo class objects to be ceated.
 *  @return true if successful, false if not
 */
bool LoadAllParametersWrapper(SimulationInfo *simInfo, boost::python::list &ltClr, boost::python::list &ltClrInfo) {
    static vector<ClusterInfo *> vtClrInfo;   // Vector of Cluster information
    static vector<Cluster *> vtClr;           // Vector of Cluster object

    bool ret = LoadAllParameters(simInfo, vtClr, vtClrInfo);

    // convert C++ vector to Python list
    std_vector_to_py_list(vtClr, ltClr);
    std_vector_to_py_list(vtClrInfo, ltClrInfo);

    return ret;
}

/*
 *  Convert Python list to C++ argv
 *
 *  @param  cnt    argument count.
 *  @param  list   Python list containing arguments.
 *  @returns       C++ arguments.
 */
char **list_to_argv_array(int cnt, boost::python::list lst)
{
    char **ret = new char*[cnt + 1];
    for (int i = 0; i < cnt; i++) {
        string s = extract<string>(lst[i]);
        size_t len = s.length();
        char *copy = new char[len + 1];
        strcpy(copy, s.c_str());
        ret[i] = copy;
    }
    ret[cnt] = NULL;
    return ret;
}

/*
 *  Handles parsing of the command line
 *
 *  @param  argList   arguments.
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @returns    true if successful, false otherwise.
 */
bool parseCommandLineWrapper(boost::python::list &argList, SimulationInfo *simInfo)
{
    int argc = len(argList);
    char** argv = list_to_argv_array(argc, argList);

    return parseCommandLine(argc, argv, simInfo);
}

/*
 *  Set recorder object to SimulationInfo.
 *  (simRecorder property)
 *
 *  @param simInfo   SimulationInfo class to read information from.
 */
void setRecorder(SimulationInfo *simInfo, IRecorder *recorder)
{
    simInfo->simRecorder = recorder;
}

/*
 *  Get recorder object from SimulationInfo.
 *  (simRecorder property in simulatorInfo)
 *
 *  @param simInfo   SimulationInfo class to read information from.
 *  @returns         Python object to store C++ recorder object.
 */
object getRecorder(const SimulationInfo *simInfo)
{
    return object(boost::ref(simInfo->simRecorder));
}

/*
 *  Get neurons' property object pointer.
 *  This function is called through 'neuronsProps' property.
 *  (ex. neuronsProps = neurons.neuronsProps)
 *
 *  @param neurons   AllNeurons class pointer that owns the neurons
 *                   property object.
 *  @returns         Neurons' property object pointer wrapped by boost:python::object.
 *       
 */
object getNeuronsProperty(const AllNeurons *neurons)
{
    return object(boost::ref(neurons->m_pNeuronsProps));
}

/*
 *  Get synapses' property object pointer.
 *  This function is called through 'synapsesProps' property.
 *  (ex. synapsesProps = synapses.synapsesProps)
 *
 *  @param neurons   AllSynapsess class pointer that owns the synapses
 *                   property object.
 *  @returns         Synapses' property object pointer wrapped by boost:python::object.
 *       
 */
object getSynapsesProperty(const AllSynapses *synapses)
{
    return object(boost::ref(synapses->m_pSynapsesProps));
}

/*
 *  Get model object from SimulationInfo.
 *  (model property in simulatorInfo)
 *  The ownership of the C++ object will be transfered to Python.
 *
 *  @param simInfo   SimulationInfo class to read information from.
 *  @returns         Python object to store C++ model object.
 */
object getModel(const SimulationInfo *simInfo)
{
    return object(transfer_to_python(simInfo->model));
}

/*
 *  Set model object to SimulationInfo.
 *
 *  @param simInfo   SimulationInfo class to read information from.
 */
void setModel(SimulationInfo *simInfo, IModel *model)
{
    simInfo->model = model;
}

/*
 *  Get epsilon value (null firing rate) in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       epsilon value.
 */
BGFLOAT getConnGrowth_epsilon(const ConnGrowth *connG)
{
    return connG->m_growth.epsilon;
}

/*
 *   Set epsilon value to ConnGrowth object.
 *
 *   @param connG      ConnGrowth class to be set the value.
 *   @param epsilon    epsilon value to set.
 */
void setConnGrowth_epsilon(ConnGrowth *connG, BGFLOAT epsilon)
{
    connG->m_growth.epsilon = epsilon;    
}

/*
 *  Get beta value (sensitivity of outgrowth to firing rate) in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       beta value.
 */
BGFLOAT getConnGrowth_beta(const ConnGrowth *connG)
{
    return connG->m_growth.beta;
}

/*
 *   Set beta value to ConnGrowth object.
 *
 *   @param connG   ConnGrowth class to be set the value.
 *   @param beta    beta value to set.
 */
void setConnGrowth_beta(ConnGrowth *connG, BGFLOAT beta)
{
    connG->m_growth.beta = beta;    
}

/*
 *  Get rho value (outgrowth rate constant) in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       rho value.
 */
BGFLOAT getConnGrowth_rho(const ConnGrowth *connG)
{
    return connG->m_growth.rho;
}

/*
 *   Set rho value to ConnGrowth object.
 *
 *   @param connG      ConnGrowth class to be set the value.
 *   @param rho        rho value to set.
 */
void setConnGrowth_rho(ConnGrowth *connG, BGFLOAT rho)
{
    connG->m_growth.rho = rho;    
}

/*
 *  Get targetRate value (Spikes/second) in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       targetRate value.
 */
BGFLOAT getConnGrowth_targetRate(const ConnGrowth *connG)
{
    return connG->m_growth.targetRate;
}

/*
 *   Set targetRat value to ConnGrowth object.
 *
 *   @param connG      ConnGrowth class to be set the value.
 *   @param targetRat  targetRat value to set.
 */
void setConnGrowth_targetRate(ConnGrowth *connG, BGFLOAT targetRate)
{
    connG->m_growth.targetRate = targetRate;    
}

/*
 *  Get minRadius value in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       minRadius value.
 */
BGFLOAT getConnGrowth_minRadius(const ConnGrowth *connG)
{
    return connG->m_growth.minRadius;
}

/*
 *   Set minRadius value to ConnGrowth object.
 *
 *   @param connG      ConnGrowth class to be set the value.
 *   @param minRadius  minRadius value to set.
 */
void setConnGrowth_minRadius(ConnGrowth *connG, BGFLOAT minRadius)
{
    connG->m_growth.minRadius = minRadius;    
}

/*
 *  Get startRadius value in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       startRadius value.
 */
BGFLOAT getConnGrowth_startRadius(const ConnGrowth *connG)
{
    return connG->m_growth.startRadius;
}

/*
 *   Set startRadius value to ConnGrowth object.
 *
 *   @param connG        ConnGrowth class to be set the value.
 *   @param startRadius  startRadius value to set.
 */
void setConnGrowth_startRadius(ConnGrowth *connG, BGFLOAT startRadius)
{
    connG->m_growth.startRadius = startRadius;    
}

/*
 *  Get maxRate (= targetRate / epsilon) value in growth parameter
 *
 *  @param connG   ConnGrowth class to read information from.
 *  @returns       maxRate value.
 */
BGFLOAT getConnGrowth_maxRate(const ConnGrowth *connG)
{
    return connG->m_growth.maxRate;
}

/*
 *   Set maxRate value to ConnGrowth object.
 *
 *   @param connG    ConnGrowth class to be set the value.
 *   @param maxRate  maxRate value to set.
 */
void setConnGrowth_maxRate(ConnGrowth *connG, BGFLOAT maxRate)
{
    connG->m_growth.maxRate = maxRate;    
}


/*
 *  Get Endogenously active neurons list.
 *
 *  @param layout  Layoiut class to read information from.
 *  @returns       Endogenously active neurons list
 */
int_vector const&  getLayout_endogenously_active_neuron_list(Layout *layout) 
{
    return layout->m_endogenously_active_neuron_list;
}

/*
 *   Set Endogenously active neurons list to Layout class object.
 *
 *   @param layout   Layout class to be set the list.
 *   @param list     Endogenously active neurons list to set.
 */
void setLayout_endogenously_active_neuron_list(Layout *layout, boost::python::list &list)
{
    python_to_vector(list, &(layout->m_endogenously_active_neuron_list));
}

/*
 *  Get Inhibitory neurons list.
 *
 *  @param layout  Layoiut class to read information from.
 *  @returns       Inhibitory neurons list.
 */
int_vector const& getLayout_inhibitory_neuron_layout(Layout *layout) 
{
    return layout->m_inhibitory_neuron_layout;
}

/*
 *   Set Inhibitory neurons list to Layout class object.
 *
 *   @param layout   Layout class to be set the list.
 *   @param list     Inhibitory neurons list to set.
 */
void setLayout_inhibitory_neuron_layout(Layout *layout, boost::python::list &list)
{
    python_to_vector(list, &(layout->m_inhibitory_neuron_layout));
}

/*
 *  Get Probed neurons list.
 *
 *  @param layout  Layoiut class to read information from.
 *  @returns       Probed neurons list.
 */
int_vector const& getLayout_probed_neuron_list(Layout *layout) 
{
    return layout->m_probed_neuron_list;
}

/*
 *   Set Probed neurons list to Layout class object.
 *
 *   @param layout   Layout class to be set the list.
 *   @param list     Probed neurons list to set.
 */
void setLayout_probed_neuron_list(Layout *layout, boost::python::list &list)
{
    python_to_vector(list, &(layout->m_probed_neuron_list));
}

/*
 *  Create a AllLIFNeurons class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllLIFNeurons> create_AllLIFNeurons()
{
    return boost::shared_ptr<AllLIFNeurons>( new AllLIFNeurons(), boost::mem_fn(&AllLIFNeurons::destroy) );
}

/*
 *  Create a AllIZHNeurons class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllIZHNeurons> create_AllIZHNeurons()
{
    return boost::shared_ptr<AllIZHNeurons>( new AllIZHNeurons(), boost::mem_fn(&AllIZHNeurons::destroy) );
}

/*
 *  Create a AllSpikingSynapses class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllSpikingSynapses> create_AllSpikingSynapses()
{
    return boost::shared_ptr<AllSpikingSynapses>( new AllSpikingSynapses(), boost::mem_fn(&AllSpikingSynapses::destroy) );
}

/*
 *  Create a AllDSSynapses class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllDSSynapses> create_AllDSSynapses()
{
    return boost::shared_ptr<AllDSSynapses>( new AllDSSynapses(), boost::mem_fn(&AllDSSynapses::destroy) );
}

/*
 *  Create a AllSTDPSynapses class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllSTDPSynapses> create_AllSTDPSynapses()
{
    return boost::shared_ptr<AllSTDPSynapses>( new AllSTDPSynapses(), boost::mem_fn(&AllSTDPSynapses::destroy) );
}

/*
 *  Create a AllDynamicSTDPSynapses class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllDynamicSTDPSynapses> create_AllDynamicSTDPSynapses()
{
    return boost::shared_ptr<AllDynamicSTDPSynapses>( new AllDynamicSTDPSynapses(), boost::mem_fn(&AllDynamicSTDPSynapses::destroy) );
}

/*
 *  Create a Model class object and return a shared pointer of it. 
 *  Convert Python list to C++ vector.
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<Model> create_Model(Connections* conns, Layout* layout, boost::python::list &ltClr, boost::python::list &ltClrInfo)
{
    static vector<ClusterInfo *> vtClrInfo;   // Vector of Cluster information
    static vector<Cluster *> vtClr;           // Vector of Cluster object

    // convert Python list to C++ vector
    python_to_vector(ltClr, &vtClr);
    python_to_vector(ltClrInfo, &vtClrInfo);

    return boost::shared_ptr<Model>( new Model(conns, layout, vtClr, vtClrInfo) );
}

/*
 *  Create a ConnStatic class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<ConnStatic> create_ConnStatic()
{
    return boost::shared_ptr<ConnStatic>( new ConnStatic(), boost::mem_fn(&ConnStatic::destroy) );
}

/*
 *  Create a ConnGrowth class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<ConnGrowth> create_ConnGrowth()
{
    return boost::shared_ptr<ConnGrowth>( new ConnGrowth(), boost::mem_fn(&ConnGrowth::destroy) );
}

/*
 *  Create a FixedLayout class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<FixedLayout> create_FixedLayout()
{
    return boost::shared_ptr<FixedLayout>( new FixedLayout(), boost::mem_fn(&FixedLayout::destroy) );
}

/*
 *  Create a DynamicLayout class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<DynamicLayout> create_DynamicLayout()
{
    return boost::shared_ptr<DynamicLayout>( new DynamicLayout(), boost::mem_fn(&DynamicLayout::destroy) );
}

#if defined(USE_GPU)
BOOST_PYTHON_MODULE(growth_cuda)
#else // USE_GPU
BOOST_PYTHON_MODULE(growth)
#endif // USE_GPU
{
    // Register C++ vector to python list converter
    to_python_converter<int_vector, vector_to_pylist_converter<int_vector>>();

    class_<array_ref<BGFLOAT>>( "bgfloat_array" )
        .def( array_indexing_suite<array_ref<BGFLOAT>>() )
        ;

    class_<IRecorder, boost::noncopyable>("IRecorder", no_init)
        .def("init", pure_virtual(&IRecorder::init))
        .def("initDefaultValues", pure_virtual(&IRecorder::initDefaultValues))
        .def("initValues", pure_virtual(&IRecorder::initValues))
        .def("getValues", pure_virtual(&IRecorder::getValues))
        .def("term", pure_virtual(&IRecorder::term))
        .def("compileHistories", pure_virtual(&IRecorder::compileHistories))
        .def("saveSimData", pure_virtual(&IRecorder::saveSimData))
    ;

    class_<XmlRecorder, bases<IRecorder>>("XmlRecorder", init<const SimulationInfo*>())
        .def("term", &XmlRecorder::term)
    ;

    class_<XmlGrowthRecorder, bases<XmlRecorder>>("XmlGrowthRecorder", init<const SimulationInfo*>())
    ;

#if defined(USE_HDF5)
    class_<Hdf5Recorder, bases<IRecorder>>("Hdf5Recorder", init<const SimulationInfo*>())
        .def("term", &Hdf5Recorder::term)
    ;

    class_<Hdf5GrowthRecorder, bases<Hdf5Recorder>>("Hdf5GrowthRecorder", init<const SimulationInfo*>())
        .def("term", &Hdf5GrowthRecorder::term)
    ;
#endif // USE_HDF5

    class_<IModel, boost::noncopyable>("IModel", no_init)
    ;

    class_<Model, boost::shared_ptr<Model>, bases<IModel>>("Model", no_init)
        .def("__init__", make_constructor(create_Model))
    ;

    class_<SimulationInfo>("SimulationInfo")
        .add_property("simRecorder", &getRecorder, &setRecorder)
        .add_property("model", &getModel, &setModel)
        .def_readwrite("width", &SimulationInfo::width)
        .def_readwrite("height", &SimulationInfo::height)
        .def_readwrite("totalNeurons", &SimulationInfo::totalNeurons)
        .def_readwrite("maxSteps", &SimulationInfo::maxSteps)
        .def_readwrite("epochDuration", &SimulationInfo::epochDuration)
        .def_readwrite("maxFiringRate", &SimulationInfo::maxFiringRate)
        .def_readwrite("maxSynapsesPerNeuron", &SimulationInfo::maxSynapsesPerNeuron)
        .def_readwrite("seed", &SimulationInfo::seed)
        .def_readwrite("numClusters", &SimulationInfo::numClusters)
        .def_readwrite("stateOutputFileName", &SimulationInfo::stateOutputFileName)
    ;

    class_<Cluster, boost::noncopyable>("Cluster", no_init)
    ;

#if defined(USE_GPU)
    class_<GPUSpikingCluster, bases<Cluster>>("GPUSpikingCluster", init<IAllNeurons *, IAllSynapses *>())
    ;
#else // USE_GPU
    class_<SingleThreadedCluster, bases<Cluster>>("SingleThreadedCluster", init<IAllNeurons *, IAllSynapses *>())
    ;
#endif // USE_GPU

    class_<ClusterInfo>("ClusterInfo")
        .def_readwrite("clusterID", &ClusterInfo::clusterID)
        .def_readwrite("clusterNeuronsBegin", &ClusterInfo::clusterNeuronsBegin)
        .def_readwrite("totalClusterNeurons", &ClusterInfo::totalClusterNeurons)
        .def_readwrite("seed", &ClusterInfo::seed)
    ;

    class_<Simulator>("Simulator")
        .def("setup", &Simulator::setup)
        .def("simulate", &Simulator::simulate)
        .def("saveData", &Simulator::saveData)
        .def("finish", &Simulator::finish)
    ;

    def("parseCommandLine", parseCommandLineWrapper);
    def("LoadAllParameters", LoadAllParametersWrapper);
    def("createRecorder", createRecorder, return_value_policy<manage_new_object>());

    // Neurons classes
    class_<IAllNeurons, boost::noncopyable>("IAllNeurons", no_init)
        .def("createNeuronsProps", pure_virtual(&IAllNeurons::createNeuronsProps))
    ;

    class_<AllNeurons, boost::noncopyable, bases<IAllNeurons>>("AllNeurons", no_init)
        .add_property("neuronsProps", &getNeuronsProperty)
    ;

    class_<AllSpikingNeurons, boost::noncopyable, bases<AllNeurons>>("AllSpikingNeurons", no_init)
    ;

    class_<AllIFNeurons, boost::noncopyable, bases<AllSpikingNeurons>>("AllIFNeurons", no_init)
        .def("createNeuronsProps", &AllIFNeurons::createNeuronsProps)
    ;

    class_<AllLIFNeurons, boost::shared_ptr<AllLIFNeurons>, boost::noncopyable, bases<AllIFNeurons>>("AllLIFNeurons", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllLIFNeurons))
    ;

    class_<AllIZHNeurons, boost::shared_ptr<AllIZHNeurons>, boost::noncopyable, bases<AllIFNeurons>>("AllIZHNeurons", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllIZHNeurons))
        .def("createNeuronsProps", &AllIZHNeurons::createNeuronsProps)
    ;

    // Neurons property classes
    class_<IAllNeuronsProps, boost::noncopyable>("IAllNeuronsProps", no_init)
    ;

    class_<AllNeuronsProps, bases<IAllNeuronsProps>>("AllNeuronsProps")
    ;

    class_<AllSpikingNeuronsProps, bases<AllNeuronsProps>>("AllSpikingNeuronsProps")
    ;

    class_<AllIFNeuronsProps, bases<AllSpikingNeuronsProps>>("AllIFNeuronsProps")
        .add_property("Iinject", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Iinject );
                      }))
        .add_property("Inoise", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Inoise );
                      }))
        .add_property("Vthresh", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Vthresh );
                      }))
        .add_property("Vresting", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Vresting );
                      }))
        .add_property("Vreset", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Vreset );
                      }))
        .add_property("Vinit", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_Vinit );
                      }))
        .add_property("starter_Vthresh", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_starter_Vthresh );
                      }))
        .add_property("starter_Vreset", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIFNeuronsProps * )>(
                      []( AllIFNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_starter_Vreset );
                      }))
    ;

    class_<AllIZHNeuronsProps, bases<AllIFNeuronsProps>>("AllIZHNeuronsProps")
        .add_property("excAconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_excAconst );
                      }))
        .add_property("inhAconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_inhAconst );
                      }))
        .add_property("excBconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_excBconst );
                      }))
        .add_property("inhBconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_inhBconst );
                      }))
        .add_property("excCconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_excCconst );
                      }))
        .add_property("inhCconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_inhCconst );
                      }))
        .add_property("excDconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_excDconst );
                      }))
        .add_property("inhDconst", 
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( AllIZHNeuronsProps * )>(
                      []( AllIZHNeuronsProps *obj ) {
                        return array_ref<BGFLOAT>( obj->m_inhDconst );
                      }))
    ;
    // Synapses classes
    class_<IAllSynapses, boost::noncopyable>("IAllSynapses", no_init)
        .def("createSynapsesProps", pure_virtual(&IAllSynapses::createSynapsesProps))
    ;

    class_<AllSynapses, boost::noncopyable, bases<IAllSynapses>>("AllSynapses", no_init)
        .add_property("synapsesProps", &getSynapsesProperty)
    ;

    class_<AllSpikingSynapses, boost::shared_ptr<AllSpikingSynapses>, bases<AllSynapses>>("AllSpikingSynapses", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllSpikingSynapses))
        .def("createSynapsesProps", &AllSpikingSynapses::createSynapsesProps)
    ;

    class_<AllDSSynapses, boost::shared_ptr<AllDSSynapses>, bases<AllSpikingSynapses>>("AllDSSynapses", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllDSSynapses))
        .def("createSynapsesProps", &AllDSSynapses::createSynapsesProps)
    ;

    class_<AllSTDPSynapses, boost::shared_ptr<AllSTDPSynapses>, bases<AllSpikingSynapses>>("AllSTDPSynapses", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllSTDPSynapses))
        .def("createSynapsesProps", &AllSTDPSynapses::createSynapsesProps)
    ;

    class_<AllDynamicSTDPSynapses, boost::shared_ptr<AllDynamicSTDPSynapses>, bases<AllSTDPSynapses>>("AllDynamicSTDPSynapses", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllDynamicSTDPSynapses))
        .def("createSynapsesProps", &AllDynamicSTDPSynapses::createSynapsesProps)
    ;

    // Synapses property classes
    class_<IAllSynapsesProps, boost::noncopyable>("IAllSynapsesProps", no_init)
    ;

    class_<AllSynapsesProps, bases<IAllSynapsesProps>>("AllSynapsesProps")
    ;

    class_<AllSpikingSynapsesProps, bases<AllSynapsesProps>>("AllSpikingSynapsesProps")
    ;

    class_<AllDSSynapsesProps, bases<AllSpikingSynapsesProps>>("AllDSSynapsesProps")
    ;

    class_<AllSTDPSynapsesProps, bases<AllSpikingSynapsesProps>>("AllSTDPSynapsesProps")
    ;

    class_<AllDynamicSTDPSynapsesProps, bases<AllSTDPSynapsesProps>>("AllDynamicSTDPSynapsesProps")
    ;

    // Connections classes
    class_<Connections, boost::noncopyable>("Connections", no_init)
    ;

    class_<ConnStatic, boost::shared_ptr<ConnStatic>, bases<Connections>>("ConnStatic", no_init)
        // We need to replace the original constructor.
        // Because connections class object will be deleted by model class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_ConnStatic))
        .def_readwrite("nConnsPerNeuron", &ConnStatic::m_nConnsPerNeuron)
        .def_readwrite("threshConnsRadius", &ConnStatic::m_threshConnsRadius)
        .def_readwrite("pRewiring", &ConnStatic::m_pRewiring)
        .add_property("excWeight",
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( ConnStatic * )>(
                      []( ConnStatic *obj ) {
                        return array_ref<BGFLOAT>( obj->m_excWeight );
                      }))
        .add_property("inhWeight",
                   /* getter that returns an array_ref view into the array */
                   static_cast<array_ref<BGFLOAT>(*)( ConnStatic * )>(
                      []( ConnStatic *obj ) {
                        return array_ref<BGFLOAT>( obj->m_inhWeight );
                      }))
    ;

    class_<ConnGrowth, boost::shared_ptr<ConnGrowth>, bases<Connections>>("ConnGrowth", no_init)
        // We need to replace the original constructor.
        // Because connections class object will be deleted by model class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_ConnGrowth))
        .add_property("epsilon", &getConnGrowth_epsilon, &setConnGrowth_epsilon)
        .add_property("beta", &getConnGrowth_beta, &setConnGrowth_beta)
        .add_property("rho", &getConnGrowth_rho, &setConnGrowth_rho)
        .add_property("targetRate", &getConnGrowth_targetRate, &setConnGrowth_targetRate)
        .add_property("minRadius", &getConnGrowth_minRadius, &setConnGrowth_minRadius)
        .add_property("startRadius", &getConnGrowth_startRadius, &setConnGrowth_startRadius)
        .add_property("maxRate", &getConnGrowth_maxRate, &setConnGrowth_maxRate)
    ;

    // Layout classes
    class_<Layout, boost::noncopyable>("Layout", no_init)
        .def("get_endogenously_active_neuron_list", &getLayout_endogenously_active_neuron_list, return_value_policy<copy_const_reference>())
        .def("set_endogenously_active_neuron_list", &setLayout_endogenously_active_neuron_list)
        .def_readwrite("num_endogenously_active_neurons", &Layout::num_endogenously_active_neurons)
        .def("get_inhibitory_neuron_layout", &getLayout_inhibitory_neuron_layout, return_value_policy<copy_const_reference>())
        .def("set_inhibitory_neuron_layout", &setLayout_inhibitory_neuron_layout)
        .def("get_probed_neuron_list", &getLayout_probed_neuron_list, return_value_policy<copy_const_reference>())
        .def("set_probed_neuron_list", &setLayout_probed_neuron_list)
    ;

    class_<FixedLayout, boost::shared_ptr<FixedLayout>, bases<Layout>>("FixedLayout", no_init)
        // We need to replace the original constructor.
        // Because layout class object will be deleted by model class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_FixedLayout))
    ;

    class_<DynamicLayout, boost::shared_ptr<DynamicLayout>, bases<Layout>>("DynamicLayout", no_init)
        // We need to replace the original constructor.
        // Because layout class object will be deleted by model class, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_DynamicLayout))
    ;
};

#endif // BOOST_PYTHON
