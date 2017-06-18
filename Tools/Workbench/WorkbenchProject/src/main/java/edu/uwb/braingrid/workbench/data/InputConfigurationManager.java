package edu.uwb.braingrid.workbench.data;
// CLEANED

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.model.InputConfiguration;
import java.io.IOException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import org.xml.sax.SAXException;

/**
 * Manages the construction of input configuration files.
 *
 * @author Del Davis
 */
public class InputConfigurationManager {

    InputConfiguration inputConfig;
    InputConfigurationBuilder inputConfigBuilder;

    private final boolean load;

    /**
     * Responsible for initializing members. Members may be initialized to a
     * previous state depending on whether a configuration filename is provided
     * (previously constructed configuration file must exist, in this case).
     *
     * @param configFilename - Name of a file for a previously persisted
     * configuration
     * @throws SAXException
     * @throws IOException
     * @throws ParserConfigurationException
     */
    public InputConfigurationManager(String configFilename) throws SAXException,
            IOException, ParserConfigurationException {
        inputConfigBuilder = new InputConfigurationBuilder();
        if (configFilename != null) {
            load = true;
            inputConfig = inputConfigBuilder.load(configFilename);
        } else {
            load = false;
            inputConfig = new InputConfiguration();
        }
    }

    /**
     * Adds a parameter and its value to the input configuration. If the
     * parameter already existed, its value is overwritten.
     *
     * Note: Parameters that are not part of the input configuration model are
     * ignored. For a list of parameters, see default values in the input
     * configuration class.
     *
     * @param parameter - The key for the parameter to add
     * @param value - The value for the parameter to add
     * @return True if the parameter is part of the model, otherwise false.
     */
    public boolean addParameterValue(String parameter, String value) {
        boolean success = inputConfig.isLegalParameter(parameter);
        if (success) {
            inputConfig.setValue(parameter, value);
        }
        return success;
    }

    /**
     * Provides the initial value of the provided parameter. This may also be
     * used to provide the current value. However, it is recommended that this
     * value be obtained from the current text in the respective GUI component.
     *
     * @param parameter - The key for the parameter who's value should be
     * returned
     * @return The value currently set for the provided parameter
     */
    public String getInitValue(String parameter) {
        if (load) {
            return inputConfig.getValue(parameter);
        } else {
            return inputConfig.getDefaultValue(parameter);
        }
    }

    /**
     * Builds the configuration XML and persists it to disk.
     *
     * @param projectName The name of the project, which is part of the path to
     * the directory containing the resulting XML file
     * @param filename - The last name (prefix and extension only, no
     * directories)
     * @return The full path to the constructed file if the operation was
     * successful, otherwise null
     * @throws TransformerException
     * @throws TransformerConfigurationException
     * @throws IOException
     */
    public String buildAndPersist(String projectName, String filename)
            throws TransformerException, TransformerConfigurationException,
            IOException {
        String fullPath = null;
        //boolean success = inputConfig.allValuesSet();
        //if (success) {
            inputConfigBuilder.build(inputConfig.getMap());
            FileManager fm = FileManager.getFileManager();
            fullPath = fm.getSimConfigFilePath(projectName, filename, true);
            inputConfigBuilder.persist(fullPath);
        //}
        return fullPath;
    }

    /**
     * Sets all parameters to their default values.
     *
     * Note: After this is called, it is important to update all values shown in
     * the GUI, as the state of each value may have changed.
     */
    public void setAllToDefault() {
        inputConfig.setAllToDefault();
    }

    /**
     * Purges all parameters and values in the input configuration and
     * associated XML document object. A call to this method results in a new
     * input configuration builder with an empty XML document and an input
     * configuration object with default values for all parameters.
     *
     * @throws ParserConfigurationException
     */
    public void purgeStoredValues() throws ParserConfigurationException {
        inputConfig.purgeStoredValues();
        inputConfigBuilder = new InputConfigurationBuilder();
    }
}
