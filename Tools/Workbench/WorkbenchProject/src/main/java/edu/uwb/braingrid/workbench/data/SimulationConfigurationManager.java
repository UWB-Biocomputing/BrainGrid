package edu.uwb.braingrid.workbench.data;

import edu.uwb.braingrid.simconfig.model.SimulationConfiguration;
import edu.uwb.braingrid.workbench.FileManager;
import java.io.IOException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import org.xml.sax.SAXException;

/**
 * Manages the construction of simulation configuration files
 * @author Aaron Conrad and Del Davis
 */
public class SimulationConfigurationManager {
    
    SimulationConfiguration simConfig;
    SimulationConfigurationBuilder simConfigBuilder;
    
    private final boolean load;
    
    public SimulationConfigurationManager(String configFilename) throws SAXException, IOException, ParserConfigurationException {
        simConfigBuilder = new SimulationConfigurationBuilder();
        if (configFilename != null) {
            load = true;
            simConfig = simConfigBuilder.load(configFilename);
        } else {
            load = false;
            //TO DO - Note that there MUST be a template file, so an empty simConfig is unreasonable
            //simConfig = new SimulationConfiguration();
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
        throws TransformerException, TransformerConfigurationException, IOException {
        String fullPath = null;
        
        //TO DO - Add this function to the SimulationConfiguration when working with Dialog's build button
        //boolean success = inputConfig.allValuesSet();
        
        boolean success = true;
        if (success) {
            simConfigBuilder.build(simConfig);
            FileManager fm = FileManager.getFileManager();
            fullPath = fm.getSimConfigFilePath(projectName, filename, true);
            simConfigBuilder.persist(fullPath);
        }
        return fullPath;
    }
}
