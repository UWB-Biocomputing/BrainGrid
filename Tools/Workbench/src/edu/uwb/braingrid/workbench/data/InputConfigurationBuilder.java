package edu.uwb.braingrid.workbench.data;

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

/**
 * Builds the XML document used as input to a simulation
 *
 * @author Del Davis
 */
class InputConfigurationBuilder {

    Document doc;
    Element root;

    /* Tag and Attribute Names */
    private static final String rootTagName = "SimParams";
    // <!-- Parameters for LSM -->
    private static final String lsmParamsTagName = "LsmParams";
    private static final String lsmFracExcAttributeName = "frac_EXC";
    private static final String lsmStartNeuronsAttributeName = "lsmStartNeurons";
    // <!-- size of pool of neurons [x y z] -->
    private static final String poolSizeTagName = "PoolSize";
    private static final String poolSizeXAttributeName = "poolSizeX";
    private static final String poolSizeYAttributeName = "poolSizeY";
    private static final String poolSizeZAttributeName = "poolSizeZ";
    // <!-- Interval of constant injected current -->
    private static final String IinjectTagName = "Iinject";
    private static final String iInjectMinAttributeName = "iInjectMin";
    private static final String iInjectMaxAttributeName = "iInjectMax";
    // <!-- Interval of STD of (gaussian) noise current -->
    private static final String iNoiseTagName = "Inoise";
    private static final String iNoiseMinAttributeName = "min";
    private static final String iNoiseMaxAttributeName = "max";
    // <!-- Interval of firing threshold -->
    private static final String vThreshTagName = "Vthresh";
    private static final String vThreshMinAttributeName = "min";
    private static final String vThreshMaxAttributeName = "max";
    // <!-- Interval of asymptotic voltage -->
    private static final String vRestingTagName = "Vresting";
    private static final String vRestingMinAttributeName = "min";
    private static final String vRestingMaxAttributeName = "max";
    // <!-- Interval of reset voltage -->
    private static final String vResetTagName = "Vreset";
    private static final String vResetMinAttributeName = "min";
    private static final String vResetMaxAttributeName = "max";
    // <!-- Interval of initial membrance voltage -->
    private static final String vInitTagName = "Vinit";
    private static final String vInitMinAttributeName = "min";
    private static final String vInitMaxAttributeName = "max";
    // <!-- Starter firing threshold -->
    private static final String starterVThreshTagName = "starter_vthresh";
    private static final String starterVThreshMinAttributeName = "min";
    private static final String starterVThreshMaxAttributeName = "max";
    // <!-- Starter reset voltage -->
    private static final String starterVResetTagName = "starter_vreset";
    private static final String starterVResetMinAttributeName = "min";
    private static final String starterVResetMaxAttributeName = "max";
    // <!-- Growth parameters -->
    private static final String growthParamsTagName
            = "GrowthParams";
    private static final String growthParamsEpsilonAttributeName
            = "epsilon";
    private static final String growthBetaAttributeName = "beta";
    private static final String growthParamsRhoAttributeName = "rho";
    private static final String growthParamsTargetRateAttributeName
            = "targetRate";
    private static final String growthParamsMinRadiusAttributeName
            = "minRadius";
    private static final String growthParamsStartRadiusAttributeName
            = "startRadius";
    // <!-- Simulation Parameters -->
    private static final String simParamsTagName = "SimParams";
    private static final String simParamsTSimAttributeName = "Tsim";
    private static final String simParamsNumSimsAttributeName = "numSims";
    private static final String simParamsMaxFiringRateAttributeName
            = "maxFiringRate";
    private static final String simParamsMaxSynapsesPerNeuronAttributeName
            = "maxSynapsesPerNeuron";
    // <!-- Simulation State Ouptut File -->
    private static final String outputParamsTagName = "OutputParams";
    private static final String outputParamsStateOutputFileNameAttributeName
            = "stateOutputFileName";
    // <!-- Random seed - set to zero to use /dev/random -->
    private static final String seedTagName = "Seed";
    private static final String seedValueAttributeName = "value";
    // <!-- If FixedLayout is present, the grid will be laid out according to 
    // the positions below, rather than randomly based on LsmParams -->
    private static final String layoutTypeTagName = "FixedLayout";
    private static final String layoutFilesTagName = "LayoutFiles";
    private static final String layoutFilesActiveNListFileNameAttributeName
            = "activeNListFileName";
    private static final String layoutFilesInhNListFileNameAttributeName
            = "inhNListFileName";
    private static final String layoutFilesPrbNListFileNameAttributeName
            = "prbNListFileName";

    public InputConfigurationBuilder() throws ParserConfigurationException {
        /* Build New XML Document */
        doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();
        root = doc.createElement(rootTagName);
        doc.appendChild(root);
    }

    public void build(HashMap<String, String> paramMap) {
        if (root != null) {
            Element elem;
            Element parent;

            // <!-- Parameters for LSM -->
            elem = doc.createElement(lsmParamsTagName);
            elem.setAttribute(lsmFracExcAttributeName,
                    paramMap.get(InputConfiguration.LSM_FRAC_EXC));
            elem.setAttribute(lsmStartNeuronsAttributeName,
                    paramMap.get(InputConfiguration.LSM_START_NEURONS));
            root.appendChild(elem);
            // <!-- size of pool of neurons [x y z] -->
            elem = doc.createElement(poolSizeTagName);
            elem.setAttribute(poolSizeXAttributeName,
                    paramMap.get(InputConfiguration.POOL_SIZE_X));
            elem.setAttribute(poolSizeYAttributeName,
                    paramMap.get(InputConfiguration.POOL_SIZE_Y));
            elem.setAttribute(poolSizeZAttributeName,
                    paramMap.get(InputConfiguration.POOL_SIZE_Z));
            root.appendChild(elem);
            // <!-- Interval of constant injected current -->
            elem = doc.createElement(IinjectTagName);
            elem.setAttribute(iInjectMinAttributeName,
                    paramMap.get(InputConfiguration.I_INJECT_MIN));
            elem.setAttribute(iInjectMaxAttributeName,
                    paramMap.get(InputConfiguration.I_INJECT_MAX));
            root.appendChild(elem);
            // <!-- Interval of STD of (gaussian) noise current -->
            elem = doc.createElement(iNoiseTagName);
            elem.setAttribute(iNoiseMinAttributeName,
                    paramMap.get(InputConfiguration.I_NOISE_MIN));
            elem.setAttribute(iNoiseMaxAttributeName,
                    paramMap.get(InputConfiguration.I_NOISE_MAX));
            root.appendChild(elem);
            // <!-- Interval of firing threshold -->
            elem = doc.createElement(vThreshTagName);
            elem.setAttribute(vThreshMinAttributeName,
                    paramMap.get(InputConfiguration.V_THRESH_MIN));
            elem.setAttribute(vThreshMaxAttributeName,
                    paramMap.get(InputConfiguration.V_THRESH_MAX));
            root.appendChild(elem);
            // <!-- Interval of asymptotic voltage -->
            elem = doc.createElement(vRestingTagName);
            elem.setAttribute(vRestingMinAttributeName,
                    paramMap.get(InputConfiguration.V_RESTING_MIN));
            elem.setAttribute(vRestingMaxAttributeName,
                    paramMap.get(InputConfiguration.V_RESTING_MAX));
            root.appendChild(elem);
            // <!-- Interval of reset voltage -->
            elem = doc.createElement(vResetTagName);
            elem.setAttribute(vResetMinAttributeName,
                    paramMap.get(InputConfiguration.V_RESET_MIN));
            elem.setAttribute(vResetMaxAttributeName,
                    paramMap.get(InputConfiguration.V_RESET_MAX));
            root.appendChild(elem);
            // <!-- Interval of initial membrance voltage -->
            elem = doc.createElement(vInitTagName);
            elem.setAttribute(vInitMinAttributeName,
                    paramMap.get(InputConfiguration.V_INIT_MIN));
            elem.setAttribute(vInitMaxAttributeName,
                    paramMap.get(InputConfiguration.V_INIT_MAX));
            root.appendChild(elem);
            // <!-- Starter firing threshold -->
            elem = doc.createElement(starterVThreshTagName);
            elem.setAttribute(starterVThreshMinAttributeName,
                    paramMap.get(InputConfiguration.STARTER_V_THRESH_MIN));
            elem.setAttribute(starterVThreshMaxAttributeName,
                    paramMap.get(InputConfiguration.STARTER_V_THRESH_MAX));
            root.appendChild(elem);
            // <!-- Starter reset voltage -->
            elem = doc.createElement(starterVResetTagName);
            elem.setAttribute(starterVResetMinAttributeName,
                    paramMap.get(InputConfiguration.STARTER_V_RESET_MIN));
            elem.setAttribute(starterVResetMaxAttributeName,
                    paramMap.get(InputConfiguration.STARTER_V_RESET_MAX));
            root.appendChild(elem);
            // <!-- Growth parameters -->
            elem = doc.createElement(growthParamsTagName);
            elem.setAttribute(growthParamsEpsilonAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_PARAMS_EPSILON));
            elem.setAttribute(growthBetaAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_BETA));
            elem.setAttribute(growthParamsRhoAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_PARAMS_RHO));
            elem.setAttribute(growthParamsTargetRateAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_PARAMS_TARGET_RATE));
            elem.setAttribute(growthParamsMinRadiusAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_PARAMS_MIN_RADIUS));
            elem.setAttribute(growthParamsStartRadiusAttributeName,
                    paramMap.get(InputConfiguration.GROWTH_PARAMS_START_RADIUS));
            root.appendChild(elem);
            // <!-- Simulation Parameters -->
            elem = doc.createElement(simParamsTagName);
            elem.setAttribute(simParamsTSimAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_T_SIM));
            elem.setAttribute(simParamsNumSimsAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_NUM_SIMS));
            elem.setAttribute(simParamsMaxFiringRateAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE));
            elem.setAttribute(simParamsMaxSynapsesPerNeuronAttributeName,
                    paramMap.get(
                            InputConfiguration.SIM_PARAMS_MAX_SYNAPSES_PER_NEURON));
            root.appendChild(elem);
            // <!-- Simulation State Ouptut File -->
            elem = doc.createElement(outputParamsTagName);
            elem.setAttribute(
                    outputParamsStateOutputFileNameAttributeName, paramMap.get(
                            InputConfiguration.OUTPUT_PARAMS_STATE_OUTPUT_FILENAME));
            root.appendChild(elem);
            // <!-- Random seed - set to zero to use /dev/random -->
            elem = doc.createElement(seedTagName);
            elem.setAttribute(seedValueAttributeName,
                    paramMap.get(InputConfiguration.SEED_VALUE));
            root.appendChild(elem);
            // <!-- If FixedLayout is present, the grid will be laid out according  
            // to the positions below, rather than randomly based on LsmParams -->
            parent = doc.createElement(layoutTypeTagName);
            elem = doc.createElement(layoutFilesTagName);
            parent.appendChild(elem);
            elem.setAttribute(layoutFilesActiveNListFileNameAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_T_SIM));
            elem.setAttribute(layoutFilesInhNListFileNameAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_NUM_SIMS));
            elem.setAttribute(layoutFilesPrbNListFileNameAttributeName,
                    paramMap.get(InputConfiguration.SIM_PARAMS_MAX_FIRING_RATE));
            root.appendChild(parent);
        }
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
        System.err.println(filename);
        // create the file we want to save
        File file = new File(filename);
        // create any necessary non-existent directories
        new File(file.getParent()).mkdirs();
        file.createNewFile();
        // write the content into xml file
        Transformer t = TransformerFactory.newInstance().newTransformer();
        t.setOutputProperty(OutputKeys.INDENT, "yes");
        t.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        t.transform(new DOMSource(doc), new StreamResult(file));
        return filename;
    }
}
