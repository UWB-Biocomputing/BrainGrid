package edu.uwb.braingrid.workbench.model;
// CLEANED

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.*;

/**
 * Maintains data for an input configuration.
 *
 * @author Del Davis
 */
 public class InputConfiguration {

    //private HashMap<String, String> inputConfig;
    //private Document inputConfig;
    private ArrayList<Element> inputConfig;
    //private HashMap<String, String> defaultValues;

    /* Config State Data */
    public static final String LSM_FRAC_EXC = "lsmFracExc";
    public static final String LSM_START_NEURONS = "lsmStartNeurons";
    public static final String POOL_SIZE_X = "poolSizeX";
    public static final String POOL_SIZE_Y = "poolSizeY";
    public static final String POOL_SIZE_Z = "poolSizeZ";
    public static final String I_INJECT_MIN = "iInjectMin";
    public static final String I_INJECT_MAX = "iInjectMax";
    public static final String I_NOISE_MIN = "iNoiseMin";
    public static final String I_NOISE_MAX = "iNoiseMax";
    public static final String V_THRESH_MIN = "vThreshMin";
    public static final String V_THRESH_MAX = "vThreshMax";
    public static final String V_RESTING_MIN = "vRestingMin";
    public static final String V_RESTING_MAX = "vRestingMax";
    public static final String V_RESET_MIN = "vResetMin";
    public static final String V_RESET_MAX = "vResetMax";
    public static final String V_INIT_MIN = "vInitMin";
    public static final String V_INIT_MAX = "vInitMax";
    public static final String STARTER_V_THRESH_MIN = "starterVThreshMin";
    public static final String STARTER_V_THRESH_MAX = "starterVThreshMax";
    public static final String STARTER_V_RESET_MIN = "starterVResetMin";
    public static final String STARTER_V_RESET_MAX = "starterVResetMax";
    public static final String GROWTH_PARAMS_EPSILON = "growthParamsEpsilon";
    public static final String GROWTH_BETA = "growthBeta";
    public static final String GROWTH_PARAMS_RHO = "growthParamsRho";
    public static final String GROWTH_PARAMS_TARGET_RATE
            = "growthParamsTargetRate";
    public static final String GROWTH_PARAMS_MIN_RADIUS
            = "growthParamsMinRadius";
    public static final String GROWTH_PARAMS_START_RADIUS
            = "growthParamsStartRadius";
    public static final String SIM_PARAMS_T_SIM = "simParamsTSim";
    public static final String SIM_PARAMS_NUM_SIMS = "simParamsNumSims";
    public static final String SIM_PARAMS_MAX_FIRING_RATE
            = "simParamsMaxFiringRate";
    public static final String SIM_PARAMS_MAX_SYNAPSES_PER_NEURON
            = "simParamsMaxSynapsesPerNeuron";
    public static final String OUTPUT_PARAMS_STATE_OUTPUT_FILENAME
            = "outputParamsStateOutputFileName";
    public static final String SEED_VALUE = "seedValue";
    public static final String LAYOUT_TYPE = "layoutType";
    public static final String LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME
            = "layoutFilesActiveNListFileName";
    public static final String LAYOUT_FILES_INH_N_LIST_FILE_NAME
            = "layoutFilesInhNListFileName";
    public static final String LAYOUT_FILES_PROBED_N_LIST_FILE_NAME
            = "probedNListFileName";
    //Testing
    public static final String NEURONS_PARAMS_CLASS = "neuronsParamsClass";
    public static final String SYNAPSES_PARAMS_CLASS = "synapsesParamsClass";
    public static final String CONNECTIONS_PARAMS_CLASS = "connectionsParamsClass";
    public static final String LAYOUT_PARAMS_CLASS = "layoutParamsClass";

    /**
     * Responsible for initializing containers for parameters/values and their
     * default values, as well as constructing this input configuration object.
     */
    public InputConfiguration() {
        inputConfig = new ArrayList<Element>();
//        defaultValues = DocumentBuilderFactory.newInstance().
//                newDocumentBuilder().newDocument();
        fillDefaultParams();
    }
    
    private void fillDefaultParams() {
        defaultValues.put(LSM_FRAC_EXC, "0.98");
        defaultValues.put(LSM_START_NEURONS, "0.10");
        defaultValues.put(POOL_SIZE_X, "10");
        defaultValues.put(POOL_SIZE_Y, "10");
        defaultValues.put(POOL_SIZE_Z, "1");
        defaultValues.put(I_INJECT_MIN, "13.5e-09");
        defaultValues.put(I_INJECT_MAX, "13.5e-09");
        defaultValues.put(I_NOISE_MIN, "1.0e-09");
        defaultValues.put(I_NOISE_MAX, "1.5e-09");
        defaultValues.put(V_THRESH_MIN, "15.0e-03");
        defaultValues.put(V_THRESH_MAX, "15.0e-03");
        defaultValues.put(V_RESTING_MIN, "0.0");
        defaultValues.put(V_RESTING_MAX, "0.0");
        defaultValues.put(V_RESET_MIN, "13.5e-03");
        defaultValues.put(V_RESET_MAX, "13.5e-03");
        defaultValues.put(V_INIT_MIN, "13.0e-03");
        defaultValues.put(V_INIT_MAX, "13.0e-03");
        defaultValues.put(STARTER_V_THRESH_MIN, "13.565e-3");
        defaultValues.put(STARTER_V_THRESH_MAX, "13.655e-3");
        defaultValues.put(STARTER_V_RESET_MIN, "13.0e-3");
        defaultValues.put(STARTER_V_RESET_MAX, "13.0e-3");
        defaultValues.put(GROWTH_PARAMS_EPSILON, "0.60");
        defaultValues.put(GROWTH_BETA, "0.10");
        defaultValues.put(GROWTH_PARAMS_RHO, "0.0001");
        defaultValues.put(GROWTH_PARAMS_TARGET_RATE, "1.9");
        defaultValues.put(GROWTH_PARAMS_MIN_RADIUS, "0.1");
        defaultValues.put(GROWTH_PARAMS_START_RADIUS, "0.4");
        defaultValues.put(SIM_PARAMS_T_SIM, "100.0");
        defaultValues.put(SIM_PARAMS_NUM_SIMS, "1");
        defaultValues.put(SIM_PARAMS_MAX_FIRING_RATE, "200");
        defaultValues.put(SIM_PARAMS_MAX_SYNAPSES_PER_NEURON, "200");
        defaultValues.put(OUTPUT_PARAMS_STATE_OUTPUT_FILENAME, "results/tR_1.9--fE_0.98_historyDump.xml");
        defaultValues.put(SEED_VALUE, "1");
        //defaultValues.put(LAYOUT_TYPE, "FixedLayout"); // obsolete?
        defaultValues.put(LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME, "Unknown");
        defaultValues.put(LAYOUT_FILES_INH_N_LIST_FILE_NAME, "Unknown");
        defaultValues.put(LAYOUT_FILES_PROBED_N_LIST_FILE_NAME, "Unknown");
        
        //Testing
        defaultValues.put(NEURONS_PARAMS_CLASS, "AllLIFNeurons");
        defaultValues.put(SYNAPSES_PARAMS_CLASS, "AllDSSynapses");
        defaultValues.put(CONNECTIONS_PARAMS_CLASS, "ConnGrowth");
        defaultValues.put(LAYOUT_PARAMS_CLASS, "FixedLayout");
    }

    /**
     * Sets the value of a parameter in the input configuration. May also be
     * used to add a new parameter.
     *
     * @param key - The parameter to be added or that's value should be set
     * @param value - The value corresponding to the parameter specified by key
     * @return The previous value associated with the parameter, or null if
     * there was no mapping for the parameter key
     */
    public String setValue(String key, String value) {
        return inputConfig.put(key, value);
    }

    /**
     * Provides the value for a parameter specified by key in the current input
     * configuration.
     *
     * @param key - The parameter that's value should be returned
     * @return The value for the parameter specified or null if the parameter is
     * not mapped in the current input configuration
     */
    public String getValue(String key) {
        return inputConfig.get(key);
    }

    /**
     * Indicates whether the specified parameter fits the model provided in the
     * default values for the configuration
     *
     * @param parameter - The key for the parameter
     * @return True if the parameter was found in the default values, otherwise
     * false
     */
    public boolean isLegalParameter(String parameter) {
        return defaultValues.containsKey(parameter);
    }

    /**
     * Provides the default value for a specified parameter
     *
     * @param parameter - The key for the parameter that's default value should
     * be provided
     * @return The default value of the parameter, or an empty string if the
     * parameter was not contained in the default input configuration
     */
    public String getDefaultValue(String parameter) {
        String value;
        if ((value = defaultValues.get(parameter)) == null) {
            value = "";
        }
        return value;
    }

    /**
     * Indicates whether all parameters specified in the default input
     * configuration have values set in the current input configuration. This
     * may be used to validate user input when all parameters are required
     * before the input configuration may be constructed.
     *
     * @return True if all parameters in the default model have had their values
     * set in the current configuration, otherwise false
     */
    public boolean allValuesSet() {
        boolean set = true;
        Iterator<String> keyIter = defaultValues.keySet().iterator();
        while (keyIter.hasNext()) {
            String key = keyIter.next();
            if (!inputConfig.containsKey(key)) {
                System.err.println(key + " not found");
                set = false;
                break;
            } else {
                if (inputConfig.get(key) == null
                        || inputConfig.get(key).equals("")) {
                    System.err.println("The value for " + key
                            + " was null or empty!");
                    set = false;
                    break;
                }
            }
        }
        return set;
    }

    /**
     * Provides the current input configuration map
     *
     * @return The input configuration map
     */
    public HashMap<String, String> getMap() {
        return inputConfig;
    }

    /**
     * Copies all of the default input configuration to the current
     * configuration.
     */
    public void setAllToDefault() {
        inputConfig = new HashMap<>(defaultValues);
    }

    /**
     * Resets the current and default configurations to their initial states.
     */
    public void purgeStoredValues() {
        inputConfig = new HashMap<>();
        defaultValues = new HashMap<>();
        fillDefaultParams();
    }
}
