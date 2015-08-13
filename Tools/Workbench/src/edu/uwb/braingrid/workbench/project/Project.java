package edu.uwb.braingrid.workbench.project;

// CLOSED FOR MODIFICATION
// NOT CLEANED
// FIX THIS!!! (Needs JavaDocs / Line Comments)
import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.project.model.Datum;
import edu.uwb.braingrid.workbench.project.model.ProjectData;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
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
    public String persist() throws ParserConfigurationException,
            TransformerException, IOException {
        // Build New XML Document
        Document doc = DocumentBuilderFactory.newInstance().
                newDocumentBuilder().newDocument();

        //  Build Root Node
        Element root = doc.createElement("project");
        doc.appendChild(root);
        // record the project name as an attribute of the root element        
        root.setAttribute("name", projectName);

        Set<String> keys = projectData.keySet();
        for (String s : keys) {
            projectData.get(s).appendElement(doc, root);
        }

        // calculate the full path to the project file
        String projectFilename = getProjectFilename();

        // create any necessary non-existent directories
        (new File(determineProjectOutputLocation())).mkdirs();

        // create the file we want to save
        File projectFile = new File(projectFilename);

        // write the content into xml file
        Transformer t = TransformerFactory.newInstance().newTransformer();
        t.setOutputProperty(OutputKeys.INDENT, "yes");
        t.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        t.transform(new DOMSource(doc), new StreamResult(projectFile));

        return projectFilename;
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

            } catch (ClassCastException ex) {
            }
        }
        return this;
    }

    public void setProjectName(String projName) {
        if (projName == null) {
            projectName = "None";
        } else {
            projectName = projName;
        }
    }

    public String getProjectName() {
        return projectName;
    }

    public void addProjectData(ProjectData projData) {
        projectData.put(projData.getName(), projData);
    }

    public ProjectData getProjectData(String key) {
        ProjectData data;
        if ((data = projectData.get(key)) == null) {
            projectData.put(key, new ProjectData(key));
        }
        return data;
    }

    /**
     * Provides the full path, including the filename, containing the XML for
     * this project.
     *
     * @return The full path, including the filename, for the file containing
     * the XML for this project
     * @throws IOException
     */
    public String getProjectFilename() throws IOException {
        if (projectName == null) {
            throw new IOException();
        }
        return determineProjectOutputLocation()
                + projectName + ".xml";
    }
    
    public ProjectData remove(String projectDataKey) {
        return projectData.remove(projectDataKey);
    }

    /**
     * Determines the folder location for a project based on the currently
     * loaded configuration
     *
     * @return The path to the project folder for the specified project
     * @throws IOException
     */
    public final String determineProjectOutputLocation()
            throws IOException {
        String workingDirectory = FileManager.getCanonicalWorkingDirectory();
        String ps = FileManager.getFileManager().getFolderDelimiter();
        String projectDirectory = workingDirectory + ps + "projects" + ps
                + projectName + ps;
        return projectDirectory;
    }

}
