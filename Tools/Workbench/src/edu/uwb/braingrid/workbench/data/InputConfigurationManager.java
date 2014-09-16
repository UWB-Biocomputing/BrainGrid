package edu.uwb.braingrid.workbench.data;

import edu.uwb.braingrid.workbench.model.InputConfiguration;
import javax.xml.parsers.ParserConfigurationException;

/**
 * Manages the construction of input configuration files.
 *
 * @author Del Davis
 */
public class InputConfigurationManager {

    InputConfiguration inputConfig;
    InputConfigurationBuilder inputConfigBuilder;

    public InputConfigurationManager() throws ParserConfigurationException {
        inputConfig = new InputConfiguration();
        inputConfigBuilder = new InputConfigurationBuilder();
    }

    public boolean addParameterValue(String parameter, String value) {
        boolean success = inputConfig.isLegalParameter(parameter);
        if (success) {
            inputConfig.setValue(parameter, value);
        }
        return success;
    }

    public String getDefaultValue(String parameter) {
        return inputConfig.getDefaultValue(parameter);
    }

    public boolean buildAndPersist() {
        boolean success = inputConfig.allValuesSet();
        if (success) {

        }
        return success;
    }
}
