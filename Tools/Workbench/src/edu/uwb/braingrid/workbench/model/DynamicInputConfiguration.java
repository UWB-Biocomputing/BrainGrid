package edu.uwb.braingrid.workbench.model;
// CLEANED

import java.util.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.*;

/**
 * Dynamically maintains data for an input configuration.
 *
 * @author Tom Wong
 */
 public class DynamicInputConfiguration {
    private Document inputConfig;
    private ArrayList<Node> paramValues;

    //Testing
    public static final String NEURONS_PARAMS_CLASS = "neuronsParamsClass";
    public static final String SYNAPSES_PARAMS_CLASS = "synapsesParamsClass";
    public static final String CONNECTIONS_PARAMS_CLASS = "connectionsParamsClass";
    public static final String LAYOUT_PARAMS_CLASS = "layoutParamsClass";

    /**
     * Responsible for initializing containers for parameters/values and their
     * default values, as well as constructing this input configuration object.
     */
    public DynamicInputConfiguration() throws ParserConfigurationException{
        inputConfig = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();
    }
    
    /**
     * 
     * @param aDoc 
     */
    public DynamicInputConfiguration(Document aDoc) {
        inputConfig = aDoc;
    }
    /**
     * Sets the value of all parameters in the input configuration.
     *
     * @param aValues the values of the parameters to update
     */
    public void setValues(ArrayList<String> aValues) {
        for(int i = 0; i < aValues.size(); i++){
            paramValues.get(i).setTextContent(aValues.get(i));
        }
    }

    /**
     * Provides the default value for a specified parameter
     *
     * @param parameter - The key for the parameter that's default value should
     * be provided
     * @return The default value of the parameter, or an empty string if the
     * parameter was not contained in the default input configuration
     */
//    public String getDefaultValue(String parameter) {
//        String value;
//        if ((value = defaultValues.get(parameter)) == null) {
//            value = "";
//        }
//        return value;
//    }

    /**
     * Provides the current input configuration map
     *
     * @return The input configuration map
     */
    public Document getDocument() {
        return inputConfig;
    }
    
    /**
     * set element
     * @param aElements
     */
    public void setElements(ArrayList<Node> aNodes) {
        paramValues = aNodes;
    }

    /**
     * Copies all of the default input configuration to the current
     * configuration.
     */
//    public void setAllToDefault() {
//        inputConfig = new HashMap<>(defaultValues);
//    }
}
