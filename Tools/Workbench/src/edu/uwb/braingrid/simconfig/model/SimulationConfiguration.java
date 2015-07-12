package edu.uwb.braingrid.simconfig.model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
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
        
        configData = new ArrayList<ConfigDatum>();
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
                    }
                }
            }
        }

        return this;
    }

    // TO DO!!!!!
    public String persist() {
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
        String filename = "C:\\Users\\Aaron\\Desktop\\SimulationConfigurationTest.xml";
        SimulationConfiguration simConfig = null;
        try {
            simConfig = new SimulationConfiguration(filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
        simConfig.persist();
    }
}
