package edu.uwb.braingrid.workbench.data;
/////////////////CLEANED

import edu.uwb.braingrid.workbench.model.InputConfiguration;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Builds the XML document used as input to a simulation
 *
 * @author Del Davis
 */
class InputConfigurationBuilder {

    Document doc;
    Element root;

    private static final String newRootTagName = "BGSimParams";
    //attribute
    private static final String nameAttributeName = "name";
    private static final String classAttributeName = "class";
    
    //tags
    private static final String simInfoParamsTagName = "SimInfoParams";
    private static final String simInfoParamsPoolSizeTagName = "PoolSize";
    private static final String simInfoParamsPoolSizeXTagName = "x";
    private static final String simInfoParamsPoolSizeYTagName = "y";
    private static final String simInfoParamsPoolSizeZTagName = "z";
    private static final String simInfoParamsSimParamsTagName = "SimParams";
    private static final String simInfoParamsSimParamsTsimTagName = "Tsim";
    private static final String simInfoParamsSimParamsNumSimsTagName = "numSims";
    private static final String simInfoParamsSimConfigTagName = "SimConfig";
    private static final String simInfoParamsSimConfigMaxFiringRateTagName = "maxFiringRate";
    private static final String simInfoParamsSimConfigMaxSynapsesPerNeuronTagName
            = "maxSynapsesPerNeuron";
    // <!-- Random seed - set to zero to use /dev/random -->
    private static final String simInfoParamsSeedTagName = "Seed";
    private static final String simInfoParamsSeedValueTagName = "value";
    // <!-- Simulation State Ouptut File -->
    private static final String simInfoParamsOutputParamsTagName = "OutputParams";
    private static final String simInfoParamsOutputParamsStateOutputFileNameTagName
            = "stateOutputFileName";
    
    private static final String modelParamsTagName = "ModelParams";
    private static final String modelParamsNeuronsParamsTagName = "NeuronsParams";
    private static final String modelParamsNeuronsParamsClassAttributeName = "class";
    private static final String modelParamsNeuronsParamsIinjectTagName = "Iinject";
    private static final String modelParamsNeuronsParamsIinjectMinTagName = "min";
    private static final String modelParamsNeuronsParamsIinjectMaxTagName = "max";
    private static final String modelParamsNeuronsParamsInoiseTagName = "Inoise";
    private static final String modelParamsNeuronsParamsInoiseMinTagName = "min";
    private static final String modelParamsNeuronsParamsInoiseMaxTagName = "max";
    private static final String modelParamsNeuronsParamsVthreshTagName = "Vthresh";
    private static final String modelParamsNeuronsParamsVthreshMinTagName = "min";
    private static final String modelParamsNeuronsParamsVthreshMaxTagName = "max";
    private static final String modelParamsNeuronsParamsVrestingTagName = "Vresting";
    private static final String modelParamsNeuronsParamsVrestingMinTagName = "min";
    private static final String modelParamsNeuronsParamsVrestingMaxTagName = "max";
    private static final String modelParamsNeuronsParamsVresetTagName = "Vreset";
    private static final String modelParamsNeuronsParamsVresetMinTagName = "min";
    private static final String modelParamsNeuronsParamsVresetMaxTagName = "max";
    private static final String modelParamsNeuronsParamsVinitTagName = "Vinit";
    private static final String modelParamsNeuronsParamsVinitMinTagName = "min";
    private static final String modelParamsNeuronsParamsVinitMaxTagName = "max";
    private static final String modelParamsNeuronsParamsStarterVthreshTagName = "starter_vthresh";
    private static final String modelParamsNeuronsParamsStarterVthreshMinTagName = "min";
    private static final String modelParamsNeuronsParamsStarterVthreshMaxTagName = "max";
    private static final String modelParamsNeuronsParamsStarterVresetTagName = "starter_vreset";
    private static final String modelParamsNeuronsParamsStarterVresetMinTagName = "min";
    private static final String modelParamsNeuronsParamsStarterVresetMaxTagName = "max";
    
    private static final String modelParamsSynapsesParamsTagName = "SynapsesParams";
    private static final String modelParamsSynapsesParamsClassAttributeName = "class";
    
    private static final String modelParamsConnectionsParamsTagName = "ConnectionsParams";
    private static final String modelParamsConnectionsParamsClassAttributeName = "class";
    private static final String modelParamsConnectionsParamsGrowthParamsTagName = "GrowthParams";
    private static final String modelParamsConnectionsParamsGrowthParamsEpsilonTagName = "epsilon";
    private static final String modelParamsConnectionsParamsGrowthParamsBetaTagName = "beta";
    private static final String modelParamsConnectionsParamsGrowthParamsRhoTagName = "rho";
    private static final String modelParamsConnectionsParamsGrowthParamsTargetRateTagName = "targetRate";
    private static final String modelParamsConnectionsParamsGrowthParamsMinRadiusTagName = "minRadius";
    private static final String modelParamsConnectionsParamsGrowthParamsStartRadiusTagName = "startRadius";
    
    private static final String modelParamsLayoutParamsTagName = "LayoutParams";
    private static final String modelParamsLayoutParamsClassAttributeName = "class";
    private static final String modelParamsLayoutParamsFixedLayoutParamsTagName = "FixedLayoutParams";
    private static final String modelParamsLayoutParamsFixedLayoutParamsLayoutFilesTagName = "LayoutFiles";
    private static final String modelParamsLayoutParamsFixedLayoutParamsLayoutFilesActiveNListFileNameTagName = "activeNListFileName";
    private static final String modelParamsLayoutParamsFixedLayoutParamsLayoutFilesInhNListFileNameTagName = "inhNListFileName";
    private static final String modelParamsLayoutParamsFixedLayoutParamsLayoutFilesPrbNListFileNameTagName = "prbNListFileName";
   
//    /* Will be deleted Later*/
//    /* Tag and Attribute Names */
//    private static final String rootTagName = "SimParams";
////     <!-- Parameters for LSM -->
//    private static final String lsmParamsTagName = "LsmParams";
//    private static final String lsmFracExcAttributeName = "frac_EXC";
//    private static final String lsmStartNeuronsAttributeName = "starter_neurons";
////     <!-- size of pool of neurons [x y z] -->
//    private static final String poolSizeTagName = "PoolSize";
//    private static final String poolSizeXAttributeName = "x";
//    private static final String poolSizeYAttributeName = "y";
//    private static final String poolSizeZAttributeName = "z";
////     <!-- Interval of constant injected current -->
//    private static final String IinjectTagName = "Iinject";
//    private static final String iInjectMinAttributeName = "min";
//    private static final String iInjectMaxAttributeName = "max";
//    // <!-- Interval of STD of (gaussian) noise current -->
//    private static final String iNoiseTagName = "Inoise";
//    private static final String iNoiseMinAttributeName = "min";
//    private static final String iNoiseMaxAttributeName = "max";
//    // <!-- Interval of firing threshold -->
//    private static final String vThreshTagName = "Vthresh";
//    private static final String vThreshMinAttributeName = "min";
//    private static final String vThreshMaxAttributeName = "max";
//    // <!-- Interval of asymptotic voltage -->
//    private static final String vRestingTagName = "Vresting";
//    private static final String vRestingMinAttributeName = "min";
//    private static final String vRestingMaxAttributeName = "max";
//    // <!-- Interval of reset voltage -->
//    private static final String vResetTagName = "Vreset";
//    private static final String vResetMinAttributeName = "min";
//    private static final String vResetMaxAttributeName = "max";
//    // <!-- Interval of initial membrane voltage -->
//    private static final String vInitTagName = "Vinit";
//    private static final String vInitMinAttributeName = "min";
//    private static final String vInitMaxAttributeName = "max";
//    // <!-- Starter firing threshold -->
//    private static final String starterVThreshTagName = "starter_vthresh";
//    private static final String starterVThreshMinAttributeName = "min";
//    private static final String starterVThreshMaxAttributeName = "max";
//    // <!-- Starter reset voltage -->
//    private static final String starterVResetTagName = "starter_vreset";
//    private static final String starterVResetMinAttributeName = "min";
//    private static final String starterVResetMaxAttributeName = "max";
////     <!-- Growth parameters -->
//    private static final String growthParamsTagName
//            = "GrowthParams";
//    private static final String growthParamsEpsilonAttributeName
//            = "epsilon";
//    private static final String growthBetaAttributeName = "beta";
//    private static final String growthParamsRhoAttributeName = "rho";
//    private static final String growthParamsTargetRateAttributeName
//            = "targetRate";
//    private static final String growthParamsMinRadiusAttributeName
//            = "minRadius";
//    private static final String growthParamsStartRadiusAttributeName
//            = "startRadius";
////     <!-- Simulation Parameters -->
//    private static final String simParamsTagName = "SimParams";
//    private static final String simParamsTSimAttributeName = "Tsim";
//    private static final String simParamsNumSimsAttributeName = "numSims";
//    private static final String simParamsMaxFiringRateAttributeName
//            = "maxFiringRate";
//    private static final String simParamsMaxSynapsesPerNeuronAttributeName
//            = "maxSynapsesPerNeuron";
//    // <!-- Simulation State Ouptut File -->
//    private static final String outputParamsTagName = "OutputParams";
//    private static final String outputParamsStateOutputFileNameAttributeName
//            = "stateOutputFileName";
//    // <!-- Random seed - set to zero to use /dev/random -->
//    private static final String seedTagName = "Seed";
//    private static final String seedValueAttributeName = "value";
////     <!-- If FixedLayout is present, the grid will be laid out according to 
////     the positions below, rather than randomly based on LsmParams -->
//    private static final String layoutTypeTagName = "FixedLayout";
//    private static final String layoutFilesTagName = "LayoutFiles";
//    private static final String layoutFilesActiveNListFileNameAttributeName
//            = "activeNListFileName";
//    private static final String layoutFilesInhNListFileNameAttributeName
//            = "inhNListFileName";
//    private static final String layoutFilesPrbNListFileNameAttributeName
//            = "prbNListFileName";
//    /* Will be deleted Later*/

    /**
     * Responsible for initializing members and constructing this builder
     *
     * @throws ParserConfigurationException
     */
    public InputConfigurationBuilder() throws ParserConfigurationException {
        /* Build New XML Document */
        doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();
        //root = doc.createElement(rootTagName);
        //doc.appendChild(root);
    }

    /**
     * Builds the XML from the parameter map.
     *
     * Note: This function does not persist the XML to disk.
     *
     * @param paramMap - A map of keys and values to use as node names and node
     * values, respectively
     */
    public void build(HashMap<String, String> paramMap) {
//        if (root != null) {
//            Element elem;
//            Element parent;
//
//            // <!-- Parameters for LSM -->
//            elem = doc.createElement(lsmParamsTagName);
//            elem.setAttribute(lsmFracExcAttributeName,
//                    paramMap.get(InputConfiguration.LSM_FRAC_EXC));
//            elem.setAttribute(lsmStartNeuronsAttributeName,
//                    paramMap.get(InputConfiguration.LSM_START_NEURONS));
//            root.appendChild(elem); 
//            // <!-- size of pool of neurons [x y z] -->
//            elem = doc.createElement(poolSizeTagName);
//            elem.setAttribute(poolSizeXAttributeName,
//                    paramMap.get(InputConfiguration.POOL_SIZE_X));
//            elem.setAttribute(poolSizeYAttributeName,
//                    paramMap.get(InputConfiguration.POOL_SIZE_Y));
//            elem.setAttribute(poolSizeZAttributeName,
//                    paramMap.get(InputConfiguration.POOL_SIZE_Z));
//            root.appendChild(elem);
//            // <!-- Interval of constant injected current -->
//            elem = doc.createElement(IinjectTagName);
//            elem.setAttribute(iInjectMinAttributeName,
//                    paramMap.get(InputConfiguration.I_INJECT_MIN));
//            elem.setAttribute(iInjectMaxAttributeName,
//                    paramMap.get(InputConfiguration.I_INJECT_MAX));
//            root.appendChild(elem);
//            // <!-- Interval of STD of (gaussian) noise current -->
//            elem = doc.createElement(iNoiseTagName);
//            elem.setAttribute(iNoiseMinAttributeName,
//                    paramMap.get(InputConfiguration.I_NOISE_MIN));
//            elem.setAttribute(iNoiseMaxAttributeName,
//                    paramMap.get(InputConfiguration.I_NOISE_MAX));
//            root.appendChild(elem);
//            // <!-- Interval of firing threshold -->
//            elem = doc.createElement(vThreshTagName);
//            elem.setAttribute(vThreshMinAttributeName,
//                    paramMap.get(InputConfiguration.V_THRESH_MIN));
//            elem.setAttribute(vThreshMaxAttributeName,
//                    paramMap.get(InputConfiguration.V_THRESH_MAX));
//            root.appendChild(elem);
//            // <!-- Interval of asymptotic voltage -->
//            elem = doc.createElement(vRestingTagName);
//            elem.setAttribute(vRestingMinAttributeName,
//                    paramMap.get(InputConfiguration.V_RESTING_MIN));
//            elem.setAttribute(vRestingMaxAttributeName,
//                    paramMap.get(InputConfiguration.V_RESTING_MAX));
//            root.appendChild(elem);
//            // <!-- Interval of reset voltage -->
//            elem = doc.createElement(vResetTagName);
//            elem.setAttribute(vResetMinAttributeName,
//                    paramMap.get(InputConfiguration.V_RESET_MIN));
//            elem.setAttribute(vResetMaxAttributeName,
//                    paramMap.get(InputConfiguration.V_RESET_MAX));
//            root.appendChild(elem);
//            // <!-- Interval of initial membrance voltage -->
//            elem = doc.createElement(vInitTagName);
//            elem.setAttribute(vInitMinAttributeName,
//                    paramMap.get(InputConfiguration.V_INIT_MIN));
//            elem.setAttribute(vInitMaxAttributeName,
//                    paramMap.get(InputConfiguration.V_INIT_MAX));
//            root.appendChild(elem);
//            // <!-- Starter firing threshold -->
//            elem = doc.createElement(starterVThreshTagName);
//            elem.setAttribute(starterVThreshMinAttributeName,
//                    paramMap.get(InputConfiguration.STARTER_V_THRESH_MIN));
//            elem.setAttribute(starterVThreshMaxAttributeName,
//                    paramMap.get(InputConfiguration.STARTER_V_THRESH_MAX));
//            root.appendChild(elem);
//            // <!-- Starter reset voltage -->
//            elem = doc.createElement(starterVResetTagName);
//            elem.setAttribute(starterVResetMinAttributeName,
//                    paramMap.get(InputConfiguration.STARTER_V_RESET_MIN));
//            elem.setAttribute(starterVResetMaxAttributeName,
//                    paramMap.get(InputConfiguration.STARTER_V_RESET_MAX));
//            root.appendChild(elem);
//            // <!-- Growth parameters -->
//            elem = doc.createElement(growthParamsTagName);
//            elem.setAttribute(growthParamsEpsilonAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_PARAMS_EPSILON));
//            elem.setAttribute(growthBetaAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_BETA));
//            elem.setAttribute(growthParamsRhoAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_PARAMS_RHO));
//            elem.setAttribute(growthParamsTargetRateAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_PARAMS_TARGET_RATE));
//            elem.setAttribute(growthParamsMinRadiusAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_PARAMS_MIN_RADIUS));
//            elem.setAttribute(growthParamsStartRadiusAttributeName,
//                    paramMap.get(InputConfiguration.GROWTH_PARAMS_START_RADIUS));
//            root.appendChild(elem);
//            // <!-- Simulation Parameters -->
//            elem = doc.createElement(simParamsTagName);
//            elem.setAttribute(simParamsTSimAttributeName,
//                    paramMap.get(InputConfiguration.SIM_PARAMS_T_SIM));
//            elem.setAttribute(simParamsNumSimsAttributeName,
//                    paramMap.get(InputConfiguration.SIM_PARAMS_NUM_SIMS));
//            elem.setAttribute(simParamsMaxFiringRateAttributeName,
//                    paramMap.get(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE));
//            elem.setAttribute(simParamsMaxSynapsesPerNeuronAttributeName,
//                    paramMap.get(
//                            InputConfiguration.SIM_PARAMS_MAX_SYNAPSES_PER_NEURON));
//            root.appendChild(elem);
//            // <!-- Simulation State Ouptut File -->
//            elem = doc.createElement(outputParamsTagName);
//            elem.setAttribute(
//                    outputParamsStateOutputFileNameAttributeName, paramMap.get(
//                            InputConfiguration.OUTPUT_PARAMS_STATE_OUTPUT_FILENAME));
//            root.appendChild(elem);
//            // <!-- Random seed - set to zero to use /dev/random -->
//            elem = doc.createElement(seedTagName);
//            elem.setAttribute(seedValueAttributeName,
//                    paramMap.get(InputConfiguration.SEED_VALUE));
//            root.appendChild(elem);
//            // <!-- If FixedLayout is present, the grid will be laid out according  
//            // to the positions below, rather than randomly based on LsmParams -->
//            parent = doc.createElement(layoutTypeTagName);
//            elem = doc.createElement(layoutFilesTagName);
//            parent.appendChild(elem);
//            elem.setAttribute(layoutFilesActiveNListFileNameAttributeName,
//                    paramMap.get(InputConfiguration.LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME));
//            elem.setAttribute(layoutFilesInhNListFileNameAttributeName,
//                    paramMap.get(InputConfiguration.LAYOUT_FILES_INH_N_LIST_FILE_NAME));
//            elem.setAttribute(layoutFilesPrbNListFileNameAttributeName,
//                    paramMap.get(InputConfiguration.LAYOUT_FILES_PROBED_N_LIST_FILE_NAME));
//            root.appendChild(parent);
//        }

        //root level
        root = doc.createElement(newRootTagName);
        doc.appendChild(root);
        
        Element elem;
        //Setup SimInfoParams
        Element simInfoParams;
        Element poolSize;
        Element simParams;
        Element simConfig;
        Element seed;
        Element outputParams;
        
        //first level
        simInfoParams = doc.createElement(simInfoParamsTagName);
        simInfoParams.setAttribute(nameAttributeName,simInfoParamsTagName);
        root.appendChild(simInfoParams);
       
        //second level
        poolSize = doc.createElement(simInfoParamsPoolSizeTagName);
        poolSize.setAttribute(nameAttributeName,simInfoParamsPoolSizeTagName);
        simParams = doc.createElement(simInfoParamsSimParamsTagName);
        simParams.setAttribute(nameAttributeName,simInfoParamsSimParamsTagName);
        simConfig = doc.createElement(simInfoParamsSimConfigTagName);
        simConfig.setAttribute(nameAttributeName,simInfoParamsSimConfigTagName);
        seed = doc.createElement(simInfoParamsSeedTagName);
        seed.setAttribute(nameAttributeName,simInfoParamsSeedTagName);
        outputParams = doc.createElement(simInfoParamsOutputParamsTagName);
        outputParams.setAttribute(nameAttributeName,simInfoParamsOutputParamsTagName);
        
        //setup the hierarchy structure
        simInfoParams.appendChild(poolSize);
        simInfoParams.appendChild(simParams);
        simInfoParams.appendChild(simConfig);
        simInfoParams.appendChild(seed);
        simInfoParams.appendChild(outputParams);
        
        //thrid level
        //poolSize
        elem = doc.createElement(simInfoParamsPoolSizeXTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsPoolSizeXTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.POOL_SIZE_X));
        poolSize.appendChild(elem);
        
        elem = doc.createElement(simInfoParamsPoolSizeYTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsPoolSizeYTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.POOL_SIZE_Y));
        poolSize.appendChild(elem);
        
        elem = doc.createElement(simInfoParamsPoolSizeZTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsPoolSizeZTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.POOL_SIZE_Z));
        poolSize.appendChild(elem);
        
        //SimParams
        elem = doc.createElement(simInfoParamsSimParamsTsimTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsSimParamsTsimTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.SIM_PARAMS_T_SIM));
        simParams.appendChild(elem);
        
        elem = doc.createElement(simInfoParamsSimParamsNumSimsTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsSimParamsNumSimsTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.SIM_PARAMS_NUM_SIMS));
        simParams.appendChild(elem);
        
        //SimConfig
        elem = doc.createElement(simInfoParamsSimConfigMaxFiringRateTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsSimConfigMaxFiringRateTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE));
        simConfig.appendChild(elem);
        
        elem = doc.createElement(simInfoParamsSimConfigMaxSynapsesPerNeuronTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsSimConfigMaxSynapsesPerNeuronTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.SIM_PARAMS_MAX_SYNAPSES_PER_NEURON));
        simConfig.appendChild(elem);
        
        //Seed
        elem = doc.createElement(simInfoParamsSeedValueTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsSeedValueTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.SEED_VALUE));
        seed.appendChild(elem);
        
        //OutputParams
        elem = doc.createElement(simInfoParamsOutputParamsStateOutputFileNameTagName);
        elem.setAttribute(nameAttributeName,simInfoParamsOutputParamsStateOutputFileNameTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.OUTPUT_PARAMS_STATE_OUTPUT_FILENAME));
        outputParams.appendChild(elem);
        
        //Setup ModelParams
        Element modelParams;
        Element neuronsParams;
        Element synapsesParams;
        Element connectionsParams;
        Element layoutParams;
        
        Element iinject;
        Element inoise;
        Element vthresh;
        Element vresting;
        Element vreset;
        Element vinit;
        Element starterVthresh;
        Element starterVreset;
        Element growthParams;
        Element fixedLayoutParams;
        
        Element layoutFiles;
        
        //first level
        modelParams = doc.createElement(modelParamsTagName);
        modelParams.setAttribute(nameAttributeName,modelParamsTagName);
        
        root.appendChild(modelParams);
        
        //second level
        neuronsParams = doc.createElement(modelParamsNeuronsParamsTagName);
        neuronsParams.setAttribute(nameAttributeName,modelParamsNeuronsParamsTagName);
        neuronsParams.setAttribute(modelParamsNeuronsParamsClassAttributeName,paramMap.get(InputConfiguration.NEURONS_PARAMS_CLASS));
        synapsesParams = doc.createElement(modelParamsSynapsesParamsTagName);
        synapsesParams.setAttribute(nameAttributeName,modelParamsSynapsesParamsTagName);
        neuronsParams.setAttribute(modelParamsSynapsesParamsClassAttributeName,paramMap.get(InputConfiguration.SYNAPSES_PARAMS_CLASS));
        connectionsParams = doc.createElement(modelParamsConnectionsParamsTagName);
        connectionsParams.setAttribute(nameAttributeName,modelParamsConnectionsParamsTagName);
        neuronsParams.setAttribute(modelParamsConnectionsParamsClassAttributeName,paramMap.get(InputConfiguration.CONNECTIONS_PARAMS_CLASS));
        layoutParams = doc.createElement(modelParamsLayoutParamsTagName);
        layoutParams.setAttribute(nameAttributeName,modelParamsLayoutParamsTagName);
        neuronsParams.setAttribute(modelParamsLayoutParamsClassAttributeName,paramMap.get(InputConfiguration.LAYOUT_PARAMS_CLASS));
        
        modelParams.appendChild(neuronsParams);
        modelParams.appendChild(synapsesParams);
        modelParams.appendChild(connectionsParams);
        modelParams.appendChild(layoutParams);
        
        //third level
        iinject = doc.createElement(modelParamsNeuronsParamsIinjectTagName);
        iinject.setAttribute(nameAttributeName,modelParamsNeuronsParamsIinjectTagName);
        inoise = doc.createElement(modelParamsNeuronsParamsInoiseTagName);
        inoise.setAttribute(nameAttributeName,modelParamsNeuronsParamsInoiseTagName);
        vthresh = doc.createElement(modelParamsNeuronsParamsVthreshTagName);
        vthresh.setAttribute(nameAttributeName,modelParamsNeuronsParamsVthreshTagName);
        vresting = doc.createElement(modelParamsNeuronsParamsVrestingTagName);
        vresting.setAttribute(nameAttributeName,modelParamsNeuronsParamsVrestingTagName);
        vreset = doc.createElement(modelParamsNeuronsParamsVresetTagName);
        vreset.setAttribute(nameAttributeName,modelParamsNeuronsParamsVresetTagName);
        vinit = doc.createElement(modelParamsNeuronsParamsVinitTagName);
        vinit.setAttribute(nameAttributeName,modelParamsNeuronsParamsVinitTagName);
        starterVthresh = doc.createElement(modelParamsNeuronsParamsStarterVthreshTagName);
        starterVthresh.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVthreshTagName);
        starterVreset = doc.createElement(modelParamsNeuronsParamsStarterVresetTagName);
        starterVreset.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVresetTagName);
        
        growthParams = doc.createElement(modelParamsConnectionsParamsGrowthParamsTagName);
        growthParams.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsTagName);
        
        fixedLayoutParams = doc.createElement(modelParamsLayoutParamsFixedLayoutParamsTagName);
        fixedLayoutParams.setAttribute(nameAttributeName,modelParamsLayoutParamsFixedLayoutParamsTagName);
        
        //setup the hierarchy structure     
        neuronsParams.appendChild(iinject);
        neuronsParams.appendChild(inoise);
        neuronsParams.appendChild(vthresh);
        neuronsParams.appendChild(vresting);
        neuronsParams.appendChild(vreset);
        neuronsParams.appendChild(vinit);
        neuronsParams.appendChild(starterVthresh);
        neuronsParams.appendChild(starterVreset);
        
        connectionsParams.appendChild(growthParams);
        
        layoutParams.appendChild(fixedLayoutParams);
        
        //fourth level
        //Iinject
        elem = doc.createElement(modelParamsNeuronsParamsIinjectMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsIinjectMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.I_INJECT_MIN));
        iinject.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsIinjectMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsIinjectMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.I_INJECT_MAX));
        iinject.appendChild(elem);
        //Inoise
        elem = doc.createElement(modelParamsNeuronsParamsInoiseMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsInoiseMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.I_NOISE_MIN));
        inoise.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsInoiseMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsInoiseMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.I_NOISE_MAX));
        inoise.appendChild(elem);
        //Vthresh
        elem = doc.createElement(modelParamsNeuronsParamsVthreshMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVthreshMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_THRESH_MIN));
        vthresh.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsVthreshMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVthreshMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_THRESH_MAX));
        vthresh.appendChild(elem);
        //Vresting
        elem = doc.createElement(modelParamsNeuronsParamsVrestingMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVrestingMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_RESTING_MIN));
        vresting.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsVrestingMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVrestingMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_RESTING_MAX));
        vresting.appendChild(elem);
        //Vreset
        elem = doc.createElement(modelParamsNeuronsParamsVresetMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVresetMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_RESET_MIN));
        vreset.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsVresetMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVresetMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_RESET_MAX));
        vreset.appendChild(elem);
        //Vinit
        elem = doc.createElement(modelParamsNeuronsParamsVinitMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVinitMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_INIT_MIN));
        vinit.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsVinitMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsVinitMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.V_INIT_MAX));
        vinit.appendChild(elem);
        //starter_vthresh
        elem = doc.createElement(modelParamsNeuronsParamsStarterVthreshMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVthreshMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.STARTER_V_THRESH_MIN));
        starterVthresh.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsStarterVthreshMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVthreshMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.STARTER_V_THRESH_MAX));
        starterVthresh.appendChild(elem);
        //starter_vreset 
        elem = doc.createElement(modelParamsNeuronsParamsStarterVresetMinTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVresetMinTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.STARTER_V_RESET_MIN));
        starterVreset.appendChild(elem);
        elem = doc.createElement(modelParamsNeuronsParamsStarterVresetMaxTagName);
        elem.setAttribute(nameAttributeName,modelParamsNeuronsParamsStarterVresetMaxTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.STARTER_V_RESET_MAX));
        starterVreset.appendChild(elem);
        
        //GrowthParams
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsEpsilonTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsEpsilonTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_PARAMS_EPSILON));
        growthParams.appendChild(elem);
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsBetaTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsBetaTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_BETA));
        growthParams.appendChild(elem);
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsRhoTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsRhoTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_PARAMS_RHO));
        growthParams.appendChild(elem);
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsTargetRateTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsTargetRateTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_PARAMS_TARGET_RATE));
        growthParams.appendChild(elem);
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsMinRadiusTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsMinRadiusTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_PARAMS_MIN_RADIUS));
        growthParams.appendChild(elem);
        elem = doc.createElement(modelParamsConnectionsParamsGrowthParamsStartRadiusTagName);
        elem.setAttribute(nameAttributeName,modelParamsConnectionsParamsGrowthParamsStartRadiusTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.GROWTH_PARAMS_START_RADIUS));
        growthParams.appendChild(elem);
        
        //FixedLayoutParams
        layoutFiles = doc.createElement(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesTagName);
        layoutFiles.setAttribute(nameAttributeName,modelParamsLayoutParamsFixedLayoutParamsLayoutFilesTagName);
        fixedLayoutParams.appendChild(layoutFiles);
        
        //fifth level
        //LayoutFiles
        elem = doc.createElement(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesActiveNListFileNameTagName);
        elem.setAttribute(nameAttributeName,modelParamsLayoutParamsFixedLayoutParamsLayoutFilesActiveNListFileNameTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME));
        layoutFiles.appendChild(elem);
        elem = doc.createElement(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesInhNListFileNameTagName);
        elem.setAttribute(nameAttributeName,modelParamsLayoutParamsFixedLayoutParamsLayoutFilesInhNListFileNameTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.LAYOUT_FILES_INH_N_LIST_FILE_NAME));
        layoutFiles.appendChild(elem);
        elem = doc.createElement(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesPrbNListFileNameTagName);
        elem.setAttribute(nameAttributeName,modelParamsLayoutParamsFixedLayoutParamsLayoutFilesPrbNListFileNameTagName);
        elem.setTextContent(paramMap.get(InputConfiguration.LAYOUT_FILES_PROBED_N_LIST_FILE_NAME));
        layoutFiles.appendChild(elem);
    }

    /**
     * Writes the document representing this input configuration XML to disk
     *
     * @return The full path to the file that was persisted
     * @throws TransformerConfigurationException
     * @throws TransformerException
     * @throws java.io.IOException
     */
    public String persist(String filename)
            throws TransformerConfigurationException, TransformerException,
            IOException {
        // create the file we want to save
        File file = new File(filename);
        // create any necessary non-existent directories
        new File(file.getParent()).mkdirs();
        file.createNewFile();
        // write the content into xml file
        Transformer t = TransformerFactory.newInstance().newTransformer();
        t.setOutputProperty(OutputKeys.INDENT, "yes");
        t.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "3");
        t.transform(new DOMSource(doc), new StreamResult(file));
        return filename;
    }

    /**
     * Provides a new input configuration based on the content of an input
     * configuration file.
     *
     * @param filename - Name of the file to read
     * @return Input configuration representing the content of the specified
     * file
     * @throws SAXException
     * @throws IOException
     * @throws ParserConfigurationException
     */
    public InputConfiguration load(String filename) throws SAXException, IOException,
            ParserConfigurationException {
        InputConfiguration ic = new InputConfiguration();

        File file = new File(filename);

//        doc = DocumentBuilderFactory.newInstance().
//                newDocumentBuilder().parse(file);
//        doc.getDocumentElement().normalize();
//        NodeList nl = doc.getElementsByTagName(rootTagName);
//        if (nl.getLength() > 0) {
//            root = (Element) nl.item(0);
//        }
//
//        Element elem;
//        String str;
//
//        // <!-- Parameters for LSM -->
//        elem = getElementFromDom(lsmParamsTagName);
//        str = elem.getAttribute(lsmFracExcAttributeName);
//        ic.setValue(InputConfiguration.LSM_FRAC_EXC, str);
//        str = elem.getAttribute(lsmStartNeuronsAttributeName);
//        ic.setValue(InputConfiguration.LSM_START_NEURONS, str);
//        // <!-- size of pool of neurons [x y z] -->
//        elem = getElementFromDom(poolSizeTagName);
//        str = elem.getAttribute(poolSizeXAttributeName);
//        ic.setValue(InputConfiguration.POOL_SIZE_X, str);
//        str = elem.getAttribute(poolSizeYAttributeName);
//        ic.setValue(InputConfiguration.POOL_SIZE_Y, str);
//        str = elem.getAttribute(poolSizeZAttributeName);
//        ic.setValue(InputConfiguration.POOL_SIZE_Z, str);
//        // <!-- Interval of constant injected current -->
//        elem = getElementFromDom(IinjectTagName);
//        str = elem.getAttribute(iInjectMinAttributeName);
//        ic.setValue(InputConfiguration.I_INJECT_MIN, str);
//        str = elem.getAttribute(iInjectMaxAttributeName);
//        ic.setValue(InputConfiguration.I_INJECT_MAX, str);
//        // <!-- Interval of STD of (gaussian) noise current -->
//        elem = getElementFromDom(iNoiseTagName);
//        str = elem.getAttribute(iNoiseMinAttributeName);
//        ic.setValue(InputConfiguration.I_NOISE_MIN, str);
//        str = elem.getAttribute(iNoiseMaxAttributeName);
//        ic.setValue(InputConfiguration.I_NOISE_MAX, str);
//        // <!-- Interval of firing threshold -->
//        elem = getElementFromDom(vThreshTagName);
//        str = elem.getAttribute(vThreshMinAttributeName);
//        ic.setValue(InputConfiguration.V_THRESH_MIN, str);
//        str = elem.getAttribute(vThreshMaxAttributeName);
//        ic.setValue(InputConfiguration.V_THRESH_MAX, str);
//        // <!-- Interval of asymptotic voltage -->
//        elem = getElementFromDom(vRestingTagName);
//        str = elem.getAttribute(vRestingMinAttributeName);
//        ic.setValue(InputConfiguration.V_RESTING_MIN, str);
//        str = elem.getAttribute(vRestingMaxAttributeName);
//        ic.setValue(InputConfiguration.V_RESTING_MAX, str);
//        // <!-- Interval of reset voltage -->
//        elem = getElementFromDom(vResetTagName);
//        str = elem.getAttribute(vResetMinAttributeName);
//        ic.setValue(InputConfiguration.V_RESET_MIN, str);
//        str = elem.getAttribute(vResetMaxAttributeName);
//        ic.setValue(InputConfiguration.V_RESET_MAX, str);
//        // <!-- Interval of initial membrane voltage -->
//        elem = getElementFromDom(vInitTagName);
//        str = elem.getAttribute(vInitMinAttributeName);
//        ic.setValue(InputConfiguration.V_INIT_MIN, str);
//        str = elem.getAttribute(vInitMaxAttributeName);
//        ic.setValue(InputConfiguration.V_INIT_MAX, str);
//        // <!-- Starter firing threshold -->
//        elem = getElementFromDom(starterVThreshTagName);
//        str = elem.getAttribute(starterVThreshMinAttributeName);
//        ic.setValue(InputConfiguration.STARTER_V_THRESH_MIN, str);
//        str = elem.getAttribute(starterVThreshMaxAttributeName);
//        ic.setValue(InputConfiguration.STARTER_V_THRESH_MAX, str);
//        // <!-- Starter reset voltage -->
//        elem = getElementFromDom(starterVResetTagName);
//        str = elem.getAttribute(starterVResetMinAttributeName);
//        ic.setValue(InputConfiguration.STARTER_V_RESET_MIN, str);
//        str = elem.getAttribute(starterVResetMaxAttributeName);
//        ic.setValue(InputConfiguration.STARTER_V_RESET_MAX, str);
//        // <!-- Growth parameters -->
//        elem = getElementFromDom(growthParamsTagName);
//        str = elem.getAttribute(growthParamsEpsilonAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_PARAMS_EPSILON, str);
//        str = elem.getAttribute(growthBetaAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_BETA, str);
//        str = elem.getAttribute(growthParamsRhoAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_PARAMS_RHO, str);
//        str = elem.getAttribute(growthParamsTargetRateAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_PARAMS_TARGET_RATE, str);
//        str = elem.getAttribute(growthParamsMinRadiusAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_PARAMS_MIN_RADIUS, str);
//        str = elem.getAttribute(growthParamsStartRadiusAttributeName);
//        ic.setValue(InputConfiguration.GROWTH_PARAMS_START_RADIUS, str);
//        // <!-- Simulation Parameters -->
//        elem = getElementFromDom(simParamsTagName);
//        str = elem.getAttribute(simParamsTSimAttributeName);
//        ic.setValue(InputConfiguration.SIM_PARAMS_T_SIM, str);
//        str = elem.getAttribute(simParamsNumSimsAttributeName);
//        ic.setValue(InputConfiguration.SIM_PARAMS_NUM_SIMS, str);
//        str = elem.getAttribute(simParamsMaxFiringRateAttributeName);
//        ic.setValue(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE, str);
//        str = elem.getAttribute(simParamsMaxSynapsesPerNeuronAttributeName);
//        ic.setValue(InputConfiguration.SIM_PARAMS_MAX_SYNAPSES_PER_NEURON, str);
//        // <!-- Simulation State Ouptut File -->
//        elem = getElementFromDom(outputParamsTagName);
//        str = elem.getAttribute(outputParamsStateOutputFileNameAttributeName);
//        ic.setValue(InputConfiguration.OUTPUT_PARAMS_STATE_OUTPUT_FILENAME, str);
//        // <!-- Random seed - set to zero to use /dev/random -->
//        elem = getElementFromDom(seedTagName);
//        str = elem.getAttribute(seedValueAttributeName);
//        ic.setValue(InputConfiguration.SEED_VALUE, str);
//        // <!-- If FixedLayout is present, the grid will be laid out according to 
//        // the positions below, rather than randomly based on LsmParams -->
//        elem = getElementFromDom(layoutTypeTagName);
//        elem = getElementFromDom(layoutFilesTagName);
//        ic.setValue(InputConfiguration.LAYOUT_TYPE, str);
//        str = elem.getAttribute(layoutFilesActiveNListFileNameAttributeName);
//        ic.setValue(InputConfiguration.LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME, str);
//        str = elem.getAttribute(layoutFilesInhNListFileNameAttributeName);
//        ic.setValue(InputConfiguration.LAYOUT_FILES_INH_N_LIST_FILE_NAME, str);
//        str = elem.getAttribute(layoutFilesPrbNListFileNameAttributeName);
//        ic.setValue(InputConfiguration.LAYOUT_FILES_PROBED_N_LIST_FILE_NAME, str);
        
        root = null;

        doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        NodeList nl = doc.getElementsByTagName(newRootTagName);
        if (nl.getLength() > 0) {
            root = (Element) nl.item(0);
        }
        else{
            return ic;
        }
        
        Element elem;
        String str;
        //Elements for SimInfoParams
        Element simInfoParams;
        Element poolSize;
        Element simParams;
        Element simConfig;
        Element seed;
        Element outputParams;
        
        //Elements for ModelParams
        Element modelParams;
        Element neuronsParams;
        Element synapsesParams;
        Element connectionsParams;
        Element layoutParams;
        
        Element iinject;
        Element inoise;
        Element vthresh;
        Element vresting;
        Element vreset;
        Element vinit;
        Element starterVthresh;
        Element starterVreset;
        Element growthParams;
        Element fixedLayoutParams;
        
        Element layoutFiles;

        //SimInfoParams
        simInfoParams = (Element)root.getElementsByTagName(simInfoParamsTagName).item(0);
        //PoolSize
        poolSize = (Element)simInfoParams.getElementsByTagName(simInfoParamsPoolSizeTagName).item(0);
        elem = (Element)poolSize.getElementsByTagName(simInfoParamsPoolSizeXTagName).item(0);
        ic.setValue(InputConfiguration.POOL_SIZE_X, elem.getTextContent());
        
        elem = (Element)poolSize.getElementsByTagName(simInfoParamsPoolSizeYTagName).item(0);
        ic.setValue(InputConfiguration.POOL_SIZE_Y, elem.getTextContent());
        
        elem = (Element)poolSize.getElementsByTagName(simInfoParamsPoolSizeZTagName).item(0);
        ic.setValue(InputConfiguration.POOL_SIZE_Z, elem.getTextContent());
        
        //SimParams
        simParams = (Element)simInfoParams.getElementsByTagName(simInfoParamsSimParamsTagName).item(0);
        elem = (Element)simParams.getElementsByTagName(simInfoParamsSimParamsTsimTagName).item(0);
        ic.setValue(InputConfiguration.SIM_PARAMS_T_SIM, elem.getTextContent());
        
        elem = (Element)simParams.getElementsByTagName(simInfoParamsSimParamsNumSimsTagName).item(0);
        ic.setValue(InputConfiguration.SIM_PARAMS_NUM_SIMS, elem.getTextContent());
        
        //SimConfig
        simConfig = (Element)simInfoParams.getElementsByTagName(simInfoParamsSimConfigTagName).item(0);
        elem = (Element)simConfig.getElementsByTagName(simInfoParamsSimConfigMaxFiringRateTagName).item(0);
        ic.setValue(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE, elem.getTextContent());
        
        elem = (Element)simConfig.getElementsByTagName(simInfoParamsSimConfigMaxSynapsesPerNeuronTagName).item(0);
        ic.setValue(InputConfiguration.SIM_PARAMS_MAX_SYNAPSES_PER_NEURON, elem.getTextContent());
        
        //Seed
        seed = (Element)simInfoParams.getElementsByTagName(simInfoParamsSeedTagName).item(0);
        elem = (Element)seed.getElementsByTagName(simInfoParamsSeedValueTagName).item(0);
        ic.setValue(InputConfiguration.SEED_VALUE, elem.getTextContent());
        
        //OutputParams
        outputParams = (Element)simInfoParams.getElementsByTagName(simInfoParamsOutputParamsTagName).item(0);
        elem = (Element)outputParams.getElementsByTagName(simInfoParamsOutputParamsStateOutputFileNameTagName).item(0);
        ic.setValue(InputConfiguration.OUTPUT_PARAMS_STATE_OUTPUT_FILENAME, elem.getTextContent());
        
        //ModelParams
        modelParams = (Element)root.getElementsByTagName(modelParamsTagName).item(0);
        //NeuronParams
        neuronsParams = (Element)modelParams.getElementsByTagName(modelParamsNeuronsParamsTagName).item(0);
        ic.setValue(InputConfiguration.NEURONS_PARAMS_CLASS, neuronsParams.getAttribute(modelParamsNeuronsParamsClassAttributeName));
        //Iinject
        iinject = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsIinjectTagName).item(0);
        elem = (Element)iinject.getElementsByTagName(modelParamsNeuronsParamsIinjectMinTagName).item(0);
        ic.setValue(InputConfiguration.I_INJECT_MIN, elem.getTextContent());
        
        elem = (Element)iinject.getElementsByTagName(modelParamsNeuronsParamsIinjectMaxTagName).item(0);
        ic.setValue(InputConfiguration.I_INJECT_MAX, elem.getTextContent());
        
        //Inoise
        inoise = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsInoiseTagName).item(0);
        elem = (Element)inoise.getElementsByTagName(modelParamsNeuronsParamsInoiseMinTagName).item(0);
        ic.setValue(InputConfiguration.I_NOISE_MIN, elem.getTextContent());
        
        elem = (Element)inoise.getElementsByTagName(modelParamsNeuronsParamsInoiseMaxTagName).item(0);
        ic.setValue(InputConfiguration.I_NOISE_MAX, elem.getTextContent());
        
        //Vthresh
        vthresh = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsVthreshTagName).item(0);
        elem = (Element)vthresh.getElementsByTagName(modelParamsNeuronsParamsVthreshMinTagName).item(0);
        ic.setValue(InputConfiguration.V_THRESH_MIN, elem.getTextContent());
        
        elem = (Element)vthresh.getElementsByTagName(modelParamsNeuronsParamsVthreshMaxTagName).item(0);
        ic.setValue(InputConfiguration.V_THRESH_MAX, elem.getTextContent());
        
        //Vresting
        vresting = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsVrestingTagName).item(0);
        elem = (Element)vresting.getElementsByTagName(modelParamsNeuronsParamsVrestingMinTagName).item(0);
        ic.setValue(InputConfiguration.V_RESTING_MIN, elem.getTextContent());
        
        elem = (Element)vresting.getElementsByTagName(modelParamsNeuronsParamsVrestingMaxTagName).item(0);
        ic.setValue(InputConfiguration.V_RESTING_MAX, elem.getTextContent());
        
        //Vreset
        vreset = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsVresetTagName).item(0);
        elem = (Element)vreset.getElementsByTagName(modelParamsNeuronsParamsVresetMinTagName).item(0);
        ic.setValue(InputConfiguration.V_RESET_MIN, elem.getTextContent());
        
        elem = (Element)vreset.getElementsByTagName(modelParamsNeuronsParamsVresetMaxTagName).item(0);
        ic.setValue(InputConfiguration.V_RESET_MAX, elem.getTextContent());
        
        //Vinit
        vinit = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsVinitTagName).item(0);
        elem = (Element)vinit.getElementsByTagName(modelParamsNeuronsParamsVinitMinTagName).item(0);
        ic.setValue(InputConfiguration.V_INIT_MIN, elem.getTextContent());
        
        elem = (Element)vinit.getElementsByTagName(modelParamsNeuronsParamsVinitMaxTagName).item(0);
        ic.setValue(InputConfiguration.V_INIT_MAX, elem.getTextContent());
        
        //starter_vthresh
        starterVthresh = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsStarterVthreshTagName).item(0);
        elem = (Element)starterVthresh.getElementsByTagName(modelParamsNeuronsParamsStarterVthreshMinTagName).item(0);
        ic.setValue(InputConfiguration.STARTER_V_THRESH_MIN, elem.getTextContent());
        
        elem = (Element)starterVthresh.getElementsByTagName(modelParamsNeuronsParamsStarterVthreshMaxTagName).item(0);
        ic.setValue(InputConfiguration.STARTER_V_THRESH_MAX, elem.getTextContent());
        
        //starter_vreset 
        starterVreset = (Element)neuronsParams.getElementsByTagName(modelParamsNeuronsParamsStarterVresetTagName).item(0);
        elem = (Element)starterVreset.getElementsByTagName(modelParamsNeuronsParamsStarterVresetMinTagName).item(0);
        ic.setValue(InputConfiguration.STARTER_V_RESET_MIN, elem.getTextContent());
        
        elem = (Element)starterVreset.getElementsByTagName(modelParamsNeuronsParamsStarterVresetMaxTagName).item(0);
        ic.setValue(InputConfiguration.STARTER_V_RESET_MAX, elem.getTextContent());
        
        //SynapsesParams
        synapsesParams = (Element)modelParams.getElementsByTagName(modelParamsSynapsesParamsTagName).item(0);
        ic.setValue(InputConfiguration.SYNAPSES_PARAMS_CLASS, synapsesParams.getAttribute(modelParamsSynapsesParamsClassAttributeName));
        
        //ConnectionsParams
        connectionsParams = (Element)modelParams.getElementsByTagName(modelParamsConnectionsParamsTagName).item(0);
        ic.setValue(InputConfiguration.CONNECTIONS_PARAMS_CLASS, connectionsParams.getAttribute(modelParamsConnectionsParamsClassAttributeName));
        //GrowthParams
        growthParams = (Element)connectionsParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsTagName).item(0);
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsEpsilonTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_PARAMS_EPSILON, elem.getTextContent());
        
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsBetaTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_BETA, elem.getTextContent());
        
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsRhoTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_PARAMS_RHO, elem.getTextContent());
        
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsTargetRateTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_PARAMS_TARGET_RATE, elem.getTextContent());
        
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsMinRadiusTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_PARAMS_MIN_RADIUS, elem.getTextContent());
        
        elem = (Element)growthParams.getElementsByTagName(modelParamsConnectionsParamsGrowthParamsStartRadiusTagName).item(0);
        ic.setValue(InputConfiguration.GROWTH_PARAMS_START_RADIUS, elem.getTextContent());
        
        //LayoutParams
        layoutParams = (Element)modelParams.getElementsByTagName(modelParamsLayoutParamsTagName).item(0);
        ic.setValue(InputConfiguration.LAYOUT_PARAMS_CLASS, layoutParams.getAttribute(modelParamsLayoutParamsClassAttributeName));
        //FixedLayoutParams
        fixedLayoutParams = (Element)layoutParams.getElementsByTagName(modelParamsLayoutParamsFixedLayoutParamsTagName).item(0);
        //LayoutFiles
        layoutFiles = (Element)fixedLayoutParams.getElementsByTagName(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesTagName).item(0);
        elem = (Element)layoutFiles.getElementsByTagName(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesActiveNListFileNameTagName).item(0);
        ic.setValue(InputConfiguration.LAYOUT_FILES_ACTIVE_N_LIST_FILE_NAME, elem.getTextContent());
        
        elem = (Element)layoutFiles.getElementsByTagName(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesInhNListFileNameTagName).item(0);
        ic.setValue(InputConfiguration.LAYOUT_FILES_INH_N_LIST_FILE_NAME, elem.getTextContent());
        
        elem = (Element)layoutFiles.getElementsByTagName(modelParamsLayoutParamsFixedLayoutParamsLayoutFilesPrbNListFileNameTagName).item(0);
        ic.setValue(InputConfiguration.LAYOUT_FILES_PROBED_N_LIST_FILE_NAME, elem.getTextContent());
        
        return ic;
    }

    /**
     * Provides an element from the current document object model of this
     * builder
     *
     * @param tagName - The tagName of the element to retrieve from the document
     * @return The element from the document with the tag name tagName, or null
     * if the element was not found or if the root element has not yet been
     * assigned
     */
    private Element getElementFromDom(String tagName) {
        Element elem = null;
        if (root != null) {
            NodeList nl = root.getElementsByTagName(tagName);
            if (nl.getLength() != 0) {
                elem = (Element) nl.item(0);
            }
        }
        return elem;
    }
}
