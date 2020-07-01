
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

#include <boost/python.hpp>
#include <boost/python/list.hpp>

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

#if defined(USE_GPU)
BOOST_PYTHON_MODULE(growth_cuda)
#else // USE_GPU
BOOST_PYTHON_MODULE(growth)
#endif // USE_GPU
{
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

    class_<SimulationInfo>("SimulationInfo")
        .def_readwrite("numClusters", &SimulationInfo::numClusters)
        .def_readwrite("stateOutputFileName", &SimulationInfo::stateOutputFileName)
        .def_readwrite("stateInputFileName", &SimulationInfo::stateInputFileName)
        .def_readwrite("memInputFileName", &SimulationInfo::memInputFileName)
        .def_readwrite("memOutputFileName", &SimulationInfo::memOutputFileName)
        .def_readwrite("stimulusInputFileName", &SimulationInfo::stimulusInputFileName)
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
};

#endif // BOOST_PYTHON
