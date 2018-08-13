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
import org.w3c.dom.Node;
import org.xml.sax.SAXException;
import java.util.logging.Logger;

/**
 * Manages the construction of input configuration files.
 *
 * @author Tom Wong
 */
public class DynamicInputConfigurationManager {

    DynamicInputConfiguration inputConfig;
    DynamicInputConfigurationBuilder inputConfigBuilder;

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
    	LOG.info("New " + getClass().getName());
        inputConfigBuilder = new DynamicInputConfigurationBuilder();
        if (configFilename != null) {
            inputConfig = inputConfigBuilder.load(configFilename);
        } else {
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
     * Add a list of nodes to the input configuration. 
     *
     * @param aNodes - List of XML Nodes
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
    
    private static final Logger LOG = Logger.getLogger(DynamicInputConfigurationManager.class.getName());
}
