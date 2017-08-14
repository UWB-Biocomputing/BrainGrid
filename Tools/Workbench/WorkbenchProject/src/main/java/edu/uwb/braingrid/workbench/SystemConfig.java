package edu.uwb.braingrid.workbench;

import edu.uwb.braingrid.workbench.data.InputAnalyzer;
import java.io.File;
import java.util.HashMap;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * This class defines the constants and functions related to system configuration
 * @author Tom Wong
 */
public class SystemConfig {
    //the base template Config file path
    public static final String BASE_TEMPLATE_INFO_XML_File_URL = "BaseTemplateConfig.xml";
    
    //Attribute names
    public static final String TEMPLATE_PATH_ATTRIBUTE_NAME = "templatePath";
    public static final String TEMPLATE_FILE_NAME_ATTRIBUTE_NAME = "templateFileName";
    public static final String TEMPLATE_DIRECTORY_ATTRIBUTE_NAME = "templateDirectory";
    public static final String NAME_ATTRIBUTE_NAME = "name";
    public static final String CLASS_ATTRIBUTE_NAME = "class";
    public static final String ALL_PARAMS_CLASSES_PATH_ATTRIBUTE_NAME = "allParamsClassesConfigFilePath";
    public static final String NODE_PATH_ATTRIBUTE_NAME = "nodePath";
    
    //Tag names
    public static final String NEURONS_PARAMS_CLASSES_TAG_NAME = "NeuronsParamsClasses";
    public static final String SYNAPSES_PARAMS_CLASSES_TAG_NAME = "SynapsesParamsClasses";
    public static final String CONNECTIONS_PARAMS_CLASSES_TAG_NAME = "ConnectionsParamsClasses";
    public static final String LAYOUT_PARAMS_CLASSES_TAG_NAME = "LayoutParamsClasses";
    public static final String STATE_OUTPUT_FILE_NAME_TAG_NAME = "stateOutputFileName";
    
    //Mapping between Tag Name and Input Type
    public static final HashMap<String,InputAnalyzer.InputType> TAG_NAME_INPUT_TYPE_MAPPING = new HashMap<String,InputAnalyzer.InputType>(){{put("activeNListFileName",InputAnalyzer.InputType.ACTIVE);put("inhNListFileName",InputAnalyzer.InputType.INHIBITORY);put("probedNListFileName",InputAnalyzer.InputType.PROBED);}};

    //Get Base Template Info Document
    public static final Document getBaseTemplateInfoDoc() throws Exception{
        return DocumentBuilderFactory.newInstance().newDocumentBuilder()
                .parse(System.getProperty("user.dir")+ File.separator + SystemConfig.BASE_TEMPLATE_INFO_XML_File_URL);
    }
    
    //get the file path of the all params classes info file stored in Base Template Info Document
    public static final String getAllParamsClassesFilePath() throws Exception{
        Document baseTemplateInfoDoc = getBaseTemplateInfoDoc();
        Node baseTemplateInfoNode = baseTemplateInfoDoc.getFirstChild();
        String allParamsClassesFilePath = ((Element)baseTemplateInfoNode).getAttribute(SystemConfig.ALL_PARAMS_CLASSES_PATH_ATTRIBUTE_NAME);
        
        return System.getProperty("user.dir")+ File.separator + allParamsClassesFilePath;
    }
    
    //get the document of the all params classes info
    public static final Document getAllParamsClassesDoc() throws Exception{
        return DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(getAllParamsClassesFilePath());
    }
}
