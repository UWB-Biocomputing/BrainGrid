package edu.uwb.braingrid.workbench.data;

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
     * Adds parameters 
     * @param parameter
     * @param value
     * @return 
     */
    public boolean addParameterValue(String parameter, String value) {
        boolean success = inputConfig.isLegalParameter(parameter);
        if (success) {
            inputConfig.setValue(parameter, value);
        }
        return success;
    }

    public String getInitValue(String parameter) {
        if (load)
            return inputConfig.getValue(parameter);
        else
            return inputConfig.getDefaultValue(parameter);
    }

    public String buildAndPersist(String projectName, String filename)
            throws TransformerException, TransformerConfigurationException,
            IOException {
        String fullPath = null;
        boolean success = inputConfig.allValuesSet();
        if (success) {
            inputConfigBuilder.build(inputConfig.getMap());
            FileManager fm = FileManager.getFileManager();
            fullPath = fm.getSimConfigFilePath(projectName, filename, true);
            inputConfigBuilder.persist(fullPath);
        }
        return fullPath;
    }

    public void setAllToDefault() {
        inputConfig.setAllToDefault();
    }

    public void purgeStoredValues() throws ParserConfigurationException {
        inputConfig.purgeStoredValues();
        inputConfigBuilder = new InputConfigurationBuilder();
    }
}
