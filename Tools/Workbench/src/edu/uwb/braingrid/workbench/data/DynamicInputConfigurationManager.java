package edu.uwb.braingrid.workbench.data;
// CLEANED

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.model.DynamicInputConfiguration;
import java.io.IOException;
import java.util.ArrayList;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.xml.sax.SAXException;

/**
 * Manages the construction of input configuration files.
 *
 * @author Tom Wong
 */
public class DynamicInputConfigurationManager {

    DynamicInputConfiguration inputConfig;
    DynamicInputConfigurationBuilder inputConfigBuilder;

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
    public DynamicInputConfigurationManager(String configFilename) throws SAXException,
            IOException, ParserConfigurationException, Exception {
        inputConfigBuilder = new DynamicInputConfigurationBuilder();
        if (configFilename != null) {
            load = true;
            inputConfig = inputConfigBuilder.load(configFilename);
        } else {
            load = false;
            inputConfig = new DynamicInputConfiguration();
        }
    }

    /**
     * Adds a parameter and its value to the input configuration. If the
     * parameter already existed, its value is overwritten.
     *
     * @param aValues - List of parameter to update
     */
    public void updateParamValues(ArrayList<String> aValues) {
        inputConfig.setValues(aValues);
    }
    
    /**
     * Adds a parameter and its value to the input configuration. If the
     * parameter already existed, its value is overwritten.
     *
     * @param aValues - List of parameter to update
     */
    public void setInputParamElements(ArrayList<Node> aNodes) {
        inputConfig.setElements(aNodes);
    }
    
    
    /**
     * Get the input XML document.
     * @return Document in DynamicInputConfiguration
     */
    public Document getInputConfigDoc() {
        return inputConfig.getDocument();
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
//    public String getInitValue(String parameter) {
//        if (load) {
//            return inputConfig.getValue(parameter);
//        } else {
//            return inputConfig.getDefaultValue(parameter);
//        }
//    }

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
        
        FileManager fm = FileManager.getFileManager();
        fullPath = fm.getSimConfigFilePath(projectName, filename, true);
        inputConfigBuilder.persist(inputConfig.getDocument(),fullPath);
            
        return fullPath;
    }

    /**
     * Sets all parameters to their default values.
     *
     * Note: After this is called, it is important to update all values shown in
     * the GUI, as the state of each value may have changed.
     */
//    public void setAllToDefault() {
//        inputConfig.setAllToDefault();
//    }
}
