package edu.uwb.braingrid.workbench.data;

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.model.InputConfiguration;
import java.io.IOException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;

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

    public String buildAndPersist(String projectName, String filename) throws TransformerException, TransformerConfigurationException, IOException {
        String fullPath = null;
        boolean success = inputConfig.allValuesSet();
        if (success) {
            inputConfigBuilder.build(inputConfig.getMap());
            FileManager fm = FileManager.getFileManager();
            fullPath = fm.getInputConfigurationFilePath(projectName, filename);
            inputConfigBuilder.persist(fullPath);
        }
        return fullPath;
    }

    public void setAllToDefault() {
        inputConfig.setAllToDefault();
    }
}
