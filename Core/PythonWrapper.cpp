
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
 *  Create a AllLIFNeurons class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllLIFNeurons> create_AllLIFNeurons()
{
    return boost::shared_ptr<AllLIFNeurons>( new AllLIFNeurons(), boost::mem_fn(&AllLIFNeurons::destroy) );
}

/*
 *  Create a AllDSSynapses class object and return a shared pointer of it. 
 *  This function is the replacement of default constructor.
 */
boost::shared_ptr<AllDSSynapses> create_AllDSSynapses()
{
    return boost::shared_ptr<AllDSSynapses>( new AllDSSynapses(), boost::mem_fn(&AllDSSynapses::destroy) );
}

#if defined(USE_GPU)
BOOST_PYTHON_MODULE(growth_cuda)
#else // USE_GPU
BOOST_PYTHON_MODULE(growth)
#endif // USE_GPU
{
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

    class_<Model, bases<IModel>>("Model", init<Connections*, Layout*, vector<Cluster*>&, vector<ClusterInfo*>&>())
    ;

    class_<SimulationInfo>("SimulationInfo")
        .add_property("simRecorder", &getRecorder, &setRecorder)
        .add_property("model", &getModel)
    ;

#if defined(USE_GPU)
    class_<GPUSpikingCluster>("GPUSpikingCluster", init<IAllNeurons *, IAllSynapses *>())
    ;
#else // USE_GPU
    class_<SingleThreadedCluster>("SingleThreadedCluster", init<IAllNeurons *, IAllSynapses *>())
    ;
#endif // USE_GPU

    class_<ClusterInfo>("ClusterInfo")
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
        // Because neurons class object will be deleted by cluster calss, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllLIFNeurons))
    ;

    class_<AllIZHNeurons, bases<AllIFNeurons>>("AllIZHNeurons")
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

    class_<AllSpikingSynapses, bases<AllSynapses>>("AllSpikingSynapses")
        .def("createSynapsesProps", &AllSpikingSynapses::createSynapsesProps)
    ;

    class_<AllDSSynapses, boost::shared_ptr<AllDSSynapses>, bases<AllSpikingSynapses>>("AllDSSynapses", no_init)
        // We need to replace the original constructor.
        // Because neurons class object will be deleted by cluster calss, 
        // we need to suppress deletion by Python.
        .def("__init__", make_constructor(create_AllDSSynapses))
        .def("createSynapsesProps", &AllDSSynapses::createSynapsesProps)
    ;

    class_<AllSTDPSynapses, bases<AllSpikingSynapses>>("AllSTDPSynapses")
        .def("createSynapsesProps", &AllSTDPSynapses::createSynapsesProps)
    ;

    class_<AllDynamicSTDPSynapses, bases<AllSTDPSynapses>>("AllDynamicSTDPSynapses")
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

    class_<ConnStatic, bases<Connections>>("ConnStatic")
    ;

    class_<ConnGrowth, bases<Connections>>("ConnGrowth")
    ;

    // Layout classes
    class_<Layout, boost::noncopyable>("Layout", no_init)
    ;

    class_<FixedLayout, bases<Layout>>("FixedLayout")
    ;

    class_<DynamicLayout, bases<Layout>>("DynamicLayout")
    ;
};

#endif // BOOST_PYTHON
