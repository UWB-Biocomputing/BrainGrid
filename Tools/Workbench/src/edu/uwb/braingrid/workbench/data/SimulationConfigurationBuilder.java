package edu.uwb.braingrid.workbench.data;

import edu.uwb.braingrid.simconfig.model.SimulationConfiguration;
import java.io.File;
import java.io.IOException;
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
import org.xml.sax.SAXException;

/**
 * Handles all interaction between the XML document used as input to a simulation
 * and the model storing the information in-memory for the Workbench, SimulationConfiguration.
 * This includes loading a file, building a document, and persisting a document.
 * 
 * @author Aaron Conrad and Del Davis
 */
public class SimulationConfigurationBuilder {
    
    Document doc;
    Element root;
    //TO DO - Decide if this is the proper tag name
    private static final String rootTagName = "SimParams";
    
    /**
     * Responsible for initializing members and constructing this builder
     *
     * @throws ParserConfigurationException
     */
    public SimulationConfigurationBuilder() throws ParserConfigurationException {
        /* Build New XML Document */
        doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();
        root = doc.createElement(rootTagName);
        doc.appendChild(root);
    }
    
    /**
     * Builds the XML from the SimulationConfiguration.
     *
     * Note: This function does not persist the XML to disk.
     *
     * @param simConfig - The SimulationConfiguration from which the document
     * is based on
     */
    public void build(SimulationConfiguration simConfig) {
        if (root != null) {
            simConfig.build(doc, root);
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
    public SimulationConfiguration load(String filename) throws SAXException, IOException,
            ParserConfigurationException {
        
        File file = new File(filename);

        doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        
        return new SimulationConfiguration(doc);
    }
}
