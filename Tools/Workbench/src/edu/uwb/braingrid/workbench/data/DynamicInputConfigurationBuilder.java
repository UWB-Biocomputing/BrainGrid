package edu.uwb.braingrid.workbench.data;
/////////////////CLEANED

import edu.uwb.braingrid.workbench.model.DynamicInputConfiguration;
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
import org.xml.sax.SAXException;

/**
 * Builds the XML document used as input to a simulation
 *
 * @author Tom Wong
 */
class DynamicInputConfigurationBuilder {
    /**
     * Responsible for initializing members and constructing this builder
     *
     * @throws ParserConfigurationException
     */
    public DynamicInputConfigurationBuilder() throws ParserConfigurationException {
    }

    /**
     * Writes the document representing this input configuration XML to disk
     *
     * @return The full path to the file that was persisted
     * @throws TransformerConfigurationException
     * @throws TransformerException
     * @throws java.io.IOException
     */
    public String persist(Document aDoc, String filename)
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
        t.transform(new DOMSource(aDoc), new StreamResult(file));
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
    public DynamicInputConfiguration load(String filename) throws SAXException, IOException,
            ParserConfigurationException {
        File file = new File(filename);

        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        
        return new DynamicInputConfiguration(doc);
    }
}
