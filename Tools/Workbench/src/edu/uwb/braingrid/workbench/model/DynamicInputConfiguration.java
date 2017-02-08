package edu.uwb.braingrid.workbench.model;
// CLEANED

import edu.uwb.braingrid.workbench.SystemConfig;
import java.util.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.*;
import org.xml.sax.SAXException;

/**
 * Dynamically maintains data for an input configuration.
 *
 * @author Tom Wong
 */
public class DynamicInputConfiguration {
     
    private Document inputConfig;
    private ArrayList<Node> paramValues;

    /**
     * Responsible for initializing containers for parameters/values and their
     * default values, as well as constructing this input configuration object.
     * @throws java.lang.Exception
     */
    public DynamicInputConfiguration() throws Exception{
        Document baseTemplateInfoDoc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(getClass().getResourceAsStream(SystemConfig.BASE_TEMPLATE_INFO_XML_File_URL));
        Node baseTemplateNode = baseTemplateInfoDoc.getFirstChild();
        String templatePath = ((Element)baseTemplateNode).getAttribute(SystemConfig.TEMPLATE_PATH_ATTRIBUTE_NAME);
        inputConfig = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(getClass().getResourceAsStream(templatePath));
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
