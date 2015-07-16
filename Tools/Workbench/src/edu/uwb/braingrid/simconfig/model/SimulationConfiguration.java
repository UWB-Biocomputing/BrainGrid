package edu.uwb.braingrid.simconfig.model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Maintains the collection of parameters to be used in a simulation once
 * persisted. In practical terms, this class is equivalent to the template, and
 * eventual final, XML document that stores all the information. Important to
 * note is that this class will only record four hierarchical levels of the XML
 * document, including the root. All other levels will not be recorded.
 *
 * @author Aaron
 */
public class SimulationConfiguration {

    private List<ConfigDatum> configData;

    public SimulationConfiguration(String filename) throws
            ParserConfigurationException, SAXException, IOException {
        load(filename);
    }

    /**
     * Loads the file with the given name, parses it, and creates a
     * SimulationConfiguration based off that file.
     * 
     * @param filename
     * @return this SimulationConfiguration
     * @throws ParserConfigurationException
     * @throws SAXException
     * @throws IOException 
     */
    public SimulationConfiguration load(String filename) throws
            ParserConfigurationException, SAXException, IOException {
        File file = new File(filename);
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        Element root = doc.getDocumentElement();
        
        configData = new ArrayList<>();
        configData.add(new ConfigDatum(root, ConfigDatum.NULL_TYPE));

        NodeList tabs = root.getChildNodes();
        Node tab;
        NodeList tabChildren;
        Node tabChild;
        NodeList tabChildrenChildren;
        Node tabChildChild;

        //For each tab
            //Add tab to list
            //For each tabChild
                //If it has children, add subhead datum
                    //For each of those children, add paramtype datum
                //If it has no children, add paramtype datum
        for (int i = 0, im = tabs.getLength(); i < im; i++) {
            tab = tabs.item(i);
            if (tab.getNodeType() == Node.ELEMENT_NODE) {
                configData.add(new ConfigDatum(tab, ConfigDatum.TAB_TYPE));
                tabChildren = tab.getChildNodes();
                for (int j = 0, jm = tabChildren.getLength(); j < jm; j++) {
                    tabChild = tabChildren.item(j);
                    if (tabChild.getNodeType() == Node.ELEMENT_NODE) {
                        if (tabChild.hasChildNodes()) {
                            configData.add(new ConfigDatum(tabChild, ConfigDatum.SUBHEAD_TYPE));
                            tabChildrenChildren = tabChild.getChildNodes();
                            for (int k = 0, km = tabChildrenChildren.getLength(); k < km; k++) {
                                tabChildChild = tabChildrenChildren.item(k);
                                if (tabChildChild.getNodeType() == Node.ELEMENT_NODE) {
                                    configData.add(new ConfigDatum(tabChildChild, ConfigDatum.PARAM_TYPE));
                                }
                            }
                        } else {
                            configData.add(new ConfigDatum(tabChild, ConfigDatum.PARAM_TYPE));
                        }
                        configData.add(new ConfigDatum(null, ConfigDatum.SUBHEAD_END));
                    }
                }
                configData.add(new ConfigDatum(null, ConfigDatum.TAB_END));
            }
        }

        return this;
    }

    // TO DO - INCLUDE CHECKS ON THE FILENAME
    public String persist(String projectFilename) throws ParserConfigurationException, TransformerException {
        // Build New XML Document
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();
        
        //Add the root
        Element root = configData.get(0).getElement(doc);
        doc.appendChild(root);
        
        //Underlying structure to build document
        Element currentTab = null;
        Element currentSubhead = null;
        Element currentElement = null;
        ConfigDatum datum;
        int datumType;
        
        for (int index = 1, im = configData.size(); index < im; index++) {
            datum = configData.get(index);
            datumType = datum.getDatumType();
            currentElement = datum.getElement(doc);
            if (datumType == ConfigDatum.TAB_TYPE) {
                //Append this to root
                root.appendChild(currentElement);
                currentTab = currentElement;
                currentSubhead = null;
            }
            else if (datumType == ConfigDatum.TAB_END) {
                currentTab = null;
                currentSubhead = null;
            }
            else if (datumType == ConfigDatum.SUBHEAD_TYPE) {
                //Append this to most recent tab
                if (currentTab != null) {
                    currentTab.appendChild(currentElement);
                    currentSubhead = currentElement;
                }
                else {
                    root.appendChild(currentElement);
                }
            }
            else if (datumType == ConfigDatum.SUBHEAD_END)  {
                currentSubhead = null;
            }
            else if (datumType == ConfigDatum.PARAM_TYPE) {
                //Append this to most recent subhead if it exists, to the tab otherwise
                if (currentSubhead != null) {
                    currentSubhead.appendChild(currentElement);
                }
                else if (currentTab != null) {
                    currentTab.appendChild(currentElement);
                }
                else {
                    root.appendChild(currentElement);
                }
            }
            else {
                //Append this to root
                root.appendChild(currentElement);
            }
        }
        
        // create the file we want to save
        File projectFile = new File(projectFilename);

        // write the content into xml file
        Transformer t = TransformerFactory.newInstance().newTransformer();
        t.setOutputProperty(OutputKeys.INDENT, "yes");
        t.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        t.transform(new DOMSource(doc), new StreamResult(projectFile));
        return null;
    }

    /**
     * Finds the ConfigDatum at the specified position and returns it
     *
     * @param position of the desired datum
     * @return ConfigDatum if found, null otherwise
     */
    public ConfigDatum getDatum(int position) {
        ConfigDatum retVal = null;
        if (position >= 0 && position < configData.size()) {
            retVal = configData.get(position);
        }
        return retVal;
    }

    public static void main(String args[]) {
        String readFile = "C:\\Users\\Aaron\\Desktop\\SimulationConfigurationTest.xml";
        String persistFile = "C:\\Users\\Aaron\\Desktop\\SimConfigOutput.xml";
        SimulationConfiguration simConfig = null;
        try {
            simConfig = new SimulationConfiguration(readFile);
            //persist(persistFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
