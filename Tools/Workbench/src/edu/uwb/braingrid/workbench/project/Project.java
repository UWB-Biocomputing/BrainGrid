package edu.uwb.braingrid.workbench.project;

// CLOSED FOR MODIFICATION

import edu.uwb.braingrid.workbench.project.model.Datum;
import edu.uwb.braingrid.workbench.project.model.ProjectData;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 *
 * @author Aaron Conrad
 */
public class Project {
    
    private HashMap<String, ProjectData> projectData;
    private String projectName;

    public Project(String projName) {
        projectName = projName;
    }

    /**
     *
     * @param filename
     * @return filename
     */
    public String persist(String filename) throws ParserConfigurationException {
        /* Build New XML Document */
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();

        /* Build Root Node */
        Element root = doc.createElement("project");
        doc.appendChild(root);
        // record the project name as an attribute of the root element        
        root.setAttribute("name", projectName);
        
        Set<String> keys = projectData.keySet();
        ProjectData projData = null;
        for (String s : keys) {
            projData = projectData.get(s);
            
        }
        return filename;
    }

    /**
     *
     * @param filename
     * @return this Project
     */
    public Project load(String filename) throws SAXException,
            ParserConfigurationException, IOException {
        File file = new File(filename);
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().parse(file);
        doc.getDocumentElement().normalize();
        Element root = doc.getDocumentElement();

        //Get all data elements
        //For each data element
        //construct the cooresponding object
        //Check tag name to determine which object is corresponding
        //
        NodeList childList = root.getChildNodes();
        NodeList eChildren = null;
        ProjectData projData;
        Element e = null;
        Element eChild = null;
        Datum datum = null;
        NamedNodeMap attributes = null;
        String tagName;
        for (int i = 0, im = childList.getLength(); i < im; i++) {
            try {
                e = (Element) childList.item(i);
                tagName = e.getTagName();
                projData = new ProjectData(tagName);
                eChildren = e.getChildNodes();
                for (int j = 0, jm = eChildren.getLength(); j < jm; j++) {
                    eChild = (Element) eChildren.item(j);
                    datum = new Datum(eChild.getTagName());
                    attributes = eChild.getAttributes();
                    for (int k = 0, km = attributes.getLength(); k < km; k++) {
                        datum.setAttribute(attributes.item(k).getNodeName(),
                                eChild.getAttribute(attributes.item(k).getNodeName()));
                    }
                    datum.setContent(eChild.getTextContent());
                }
                    
            } catch (ClassCastException ex) {}
        }
        return this;
    }

    public String getProjectName() {
        return projectName;
    }
    
    public void addProjectData(ProjectData projData) {
        projectData.put(projData.getName(), projData);
    }
    
    public ProjectData getProjectData(String key) {
        return projectData.get(key);
    }
}
