package edu.uwb.braingrid.workbench;

import java.io.File;
import java.io.IOException;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.xml.sax.SAXException;

/**
 *
 * @author Tom Wong
 */
public class SystemConfig {
    public static final String BASE_TEMPLATE_INFO_XML_File_URL = "BaseTemplateConfig.xml";
    
    public static final String TEMPLATE_PATH_ATTRIBUTE_NAME = "templatePath";
    public static final String TEMPLATE_FILE_NAME_ATTRIBUTE_NAME = "templateFileName";
    public static final String TEMPLATE_DIRECTORY_ATTRIBUTE_NAME = "templateDirectory";
    public static final String NAME_ATTRIBUTE_NAME = "name";
    public static final String CLASS_ATTRIBUTE_NAME = "class";
    public static final String ALL_PARAMS_CLASSES_PATH_ATTRIBUTE_NAME = "allParamsClassesConfigFilePath";
    public static final String NODE_PATH_ATTRIBUTE_NAME = "nodePath";
    
    public static final String NEURONS_PARAMS_CLASSES_TAG_NAME = "NeuronsParamsClasses";
    public static final String SYNAPSES_PARAMS_CLASSES_TAG_NAME = "SynapsesParamsClasses";
    public static final String CONNECTIONS_PARAMS_CLASSES_TAG_NAME = "ConnectionsParamsClasses";
    public static final String LAYOUT_PARAMS_CLASSES_TAG_NAME = "LayoutParamsClasses";
    
    public static final Document getBaseTemplateInfoDoc() throws Exception{
        return DocumentBuilderFactory.newInstance().newDocumentBuilder()
                .parse(System.getProperty("user.dir")+ File.separator + SystemConfig.BASE_TEMPLATE_INFO_XML_File_URL);
    }
    
    public static final String getAllParamsClassesFilePath() throws Exception{
        Document baseTemplateInfoDoc = getBaseTemplateInfoDoc();
        Node baseTemplateInfoNode = baseTemplateInfoDoc.getFirstChild();
        String allParamsClassesFilePath = ((Element)baseTemplateInfoNode).getAttribute(SystemConfig.ALL_PARAMS_CLASSES_PATH_ATTRIBUTE_NAME);
        
        return System.getProperty("user.dir")+ File.separator + allParamsClassesFilePath;
    }
    
    public static final Document getAllParamsClassesDoc() throws Exception{
        return DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(getAllParamsClassesFilePath());
    }
    
    public static final int getNumOfParamsClassTypes() throws Exception{
        Document baseTemplateInfoDoc = getBaseTemplateInfoDoc();
        Node baseTemplateInfoNode = baseTemplateInfoDoc.getFirstChild();
        return baseTemplateInfoNode.getChildNodes().getLength();
    }
}
