package edu.uwb.braingrid.workbench.model;

import java.util.HashMap;

/**
 * Maintains data for an input configuration.
 *
 * @author Del Davis
 */
public class InputConfiguration {

    private HashMap<String, String> inputConfig;

    /* Config State Data */
    private static final String LSM_FRAC_EXC = "lsmFracExc";
    private static final String LSM_START_NEURONS = "lsmStartNeurons";
    private static final String POOL_SIZE_X = "poolSizeX";
    private static final String POOL_SIZE_Y = "poolSizeY";
    private static final String POOL_SIZE_Z = "poolSizeZ";
    private static final String I_INJECT_MIN = "iInjectMin";
    private static final String I_INJECT_MAX = "iInjectMax";
    private static final String I_NOISE_MIN = "iNoiseMin";
    private static final String I_NOISE_MAX = "iNoiseMax";
    private static final String V_THRESH_MIN = "vThreshMin";
    private static final String V_THRESH_MAX = "vThreshMax";
    private static final String V_RESTING_MIN = "vRestingMin";
    private static final String V_RESTING_MAX = "vRestingMax";
    private static final String V_RESET_MIN = "vResetMin";
    private static final String V_RESET_MAX = "vResetMax";
    private static final String V_INIT_MIN = "vInitMin";
    private static final String V_INIT_MAX = "vInitMax";
    private static final String STARTER_V_THRESH_MIN = "starterVThreshMin";
    private static final String STARTER_V_THRESH_MAX = "starterVThreshMax";
    private static final String STARTER_V_RESET_MIN = "starterVResetMin";
    private static final String STARTER_V_RESET_MAX = "starterVResetMax";
    private static final String GROWTH_PARAMS_EPSILON = "growthParamsEpsilon";
    private static final String GROWTH_BETA = "growthBeta";
    private static final String GROWTH_PARAMS_RHO = "growthParamsRho";
    private static final String GROWTH_PARAMS_TARGET_RATE
            = "growthParamsTargetRate";
    private static final String GROWTH_PARAMS_MIN_RADIUS
            = "growthParamsMinRadius";
    private static final String GROWTH_PARAMS_START_RADIUS
            = "growthParamsStartRadius";
    private static final String SIM_PARAMS_T_SIM = "simParamsTSim";
    private static final String SIM_PARAMS_NUM_SIMS = "simParamsNumSims";
    private static final String SIM_PARAMS_MAX_FIRING_RATE
            = "simParamsMaxFiringRate";
    private static final String SIM_PARAMS_MAX_SYNAPSES_PER_NEURON
            = "simParamsMaxSynapsesPerNeuron";
    private static final String OUTPUT_PARAMS_STATE_OUTPUT_FILENAME
            = "outputParamsStateOutputFileName";
    private static final String SEED_VALUE = "seedValue";
    private static final String LAYOUT_TYPE = "layoutType";
    private static final String LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME
            = "layoutFilesActiveNListFileName";
    private static final String LAYOUT_FILES_INH_N_LIST_FILE_NAME
            = "layoutFilesInhNListFileName";

    public InputConfiguration() {
        inputConfig = new HashMap<>();
    }

    public String setValue(String key, String value) {
        return inputConfig.put(key, value);
    }

    public String getValue(String key) {
        return inputConfig.get(key);
    }
}
