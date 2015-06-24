package edu.uwb.braingrid.workbench.data;
/////////////////CLEANED
import java.io.File;
import java.io.IOException;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

/**
 * Supports handling input-related data
 *
 * @author Del Davis
 */
public class InputAnalyzer {

    /**
     * Indicates the amount of input file required as arguments for a simulation
     */
    public static final int inputsRequiredForSim = 3;

    /**
     * Provides an input type corresponding to a specific type of neuron list
     */
    public enum InputType {

        /**
         * type of active neuron list
         */
        ACTIVE,
        /**
         * type of inhibitory neuron list
         */
        INHIBITORY,
        /**
         * type of probed neuron list
         */
        PROBED,
        /**
         * error code indicating an invalid type of neuron list (or no input
         * list)
         */
        INVALID
    }

    /**
     * Parses a BrainGrid simulation input XML to determine the input type. The
     * input type is encoded as the root element's tag name: A for active neuron
     * list, I for inhibitory neuron list, P for probed neuron list. If the root
     * node's tag name does not match one of these values the error code INVALID
     * is returned.
     *
     * @param file - The input list file
     * @return - The type of neuron list embedded in the input file
     * @throws org.xml.sax.SAXParseException
     * @throws ParserConfigurationException
     * @throws SAXException
     * @throws IOException
     */
    public static InputType getInputType(File file) throws SAXParseException, ParserConfigurationException, SAXException, IOException {
        InputType inputType = InputType.INVALID;
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        Element root = doc.getDocumentElement();
        String type = root.getTagName();
        switch (type) {
            case "A":
                inputType = InputType.ACTIVE;
                break;
            case "I":
                inputType = InputType.INHIBITORY;
                break;
            case "P":
                inputType = InputType.PROBED;
                break;
        }
        return inputType;
    }
}
