package edu.uwb.braingrid.workbench.model;

import java.util.HashMap;
import java.util.Iterator;

/**
 * Maintains data for an input configuration.
 *
 * @author Del Davis
 */
public class InputConfiguration {

    private HashMap<String, String> inputConfig;
    private HashMap<String, String> defaultValues;

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

    public InputConfiguration() {
        inputConfig = new HashMap<>();
        defaultValues = new HashMap<>();
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
        defaultValues.put(SIM_PARAMS_NUM_SIMS, "600");
        defaultValues.put(SIM_PARAMS_MAX_FIRING_RATE, "200");
        defaultValues.put(SIM_PARAMS_MAX_SYNAPSES_PER_NEURON, "200");
        defaultValues.put(OUTPUT_PARAMS_STATE_OUTPUT_FILENAME, "results/tR_1.9--fE_0.98_historyDump.xml");
        defaultValues.put(SEED_VALUE, "1");
        defaultValues.put(LAYOUT_TYPE, "FixedLayout");
        defaultValues.put(LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME, "Unknown");
        defaultValues.put(LAYOUT_FILES_INH_N_LIST_FILE_NAME, "Unknown");
        defaultValues.put(LAYOUT_FILES_PROBED_N_LIST_FILE_NAME, "Unknown");
    }

    public String setValue(String key, String value) {
        return inputConfig.put(key, value);
    }

    public String getValue(String key) {
        return inputConfig.get(key);
    }

    public boolean isLegalParameter(String parameter) {
        return defaultValues.containsKey(parameter);
    }

    public String getDefaultValue(String parameter) {
        String value;
        if ((value = defaultValues.get(parameter)) == null) {
            value = "";
        }
        return value;
    }

    public boolean allValuesSet() {
        boolean set = true;
        Iterator<String> keyIter = defaultValues.keySet().iterator();
        while (keyIter.hasNext()) {
            String key = keyIter.next();
            if (!inputConfig.containsKey(key)) {
                set = false;
                break;
            } else {
                if (inputConfig.get(key) == null
                        || inputConfig.get(key).equals("")) {
                    set = false;
                    break;
                }
            }
        }
        return set;
    }

    public HashMap<String, String> getMap() {
        return inputConfig;
    }

    public void setAllToDefault() {
        inputConfig = new HashMap<>(defaultValues);
    }
}
