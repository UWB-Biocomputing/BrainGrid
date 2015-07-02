package edu.uwb.braingrid.workbench.project;
// NOT CLEANED (Still Implementing / Testing / JavaDocs / Class Header)

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.model.ScriptHistory;
import edu.uwb.braingrid.workbench.model.SimulationSpecification;
import edu.uwb.braingrid.workbench.project.model.Datum;
import edu.uwb.braingrid.workbench.project.model.ProjectData;
import edu.uwb.braingrid.workbench.utils.DateTime;
import java.io.IOException;
import java.util.Date;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import org.w3c.dom.DOMException;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.SAXException;

/**
 *
 * @author Aaron
 */
public class ProjectManager {

    // <editor-fold defaultstate="collapsed" desc="Members">
    private Project project;
//    private Document doc;
//    private String name;
//    private Element root;
//    private Element provElement;
//    private Element scriptVersion;
//    private List<Element> inputs;
//    private Element simulator;
//    private Element simulationConfigurationFile;
//    private Element script;
//    private boolean provEnabled;
//    private static final String projectTagName = "project";
//    private static final String projectNameAttribute = "name";
    private static final int defaultScriptVersion = 0;

    private static final String provTagName = "provenance";

    // FIX THIS : Find where this is used
    private static final String provLocationTagName = "location";

    private static final String provEnabledAttributeName = "enabled";
    private static final String simulatorTagName = "simulator";
    private static final String simulatorExecutionMachine
            = "executionMachine";
    private static final String hostnameTagName = "hostname";
    private static final String simFolderTagName = "simulatorFolder";
    private static final String simulationTypeTagName = "ProcessingType";
    private static final String simulatorSourceCodeUpdatingTagName
            = "sourceCodeUpdating";
    private static final String SHA1KeyTagName = "SHA1Key";
    private static final String buildOptionTagName = "BuildOption";
    private static final String scriptVersionTagName = "scriptVersion";
    private static final String scriptVersionVersionTagName = "version";
    private static final String simulatorVersionAnnotationTagName = "version";
    private static final String simulatorCodeLocationTagName = "repository";
    private static final String scriptTagName = "script";
    private static final String scriptFileTagName = "file";
    private static final String scriptRanRunAttributeName = "ran";
    private static final String scriptRanAtAttributeName = "atMillis";
    private static final String scriptHostnameTagName = "hostname";
    private static final String scriptCompletedAtAttributeName = "completedAt";
    private static final String scriptAnalyzedAttributeName = "outputAnalyzed";
    private static final String simConfigFileTagName = "simConfigFile";
    private static final String simulationConfigurationFileAttributeName
            = "simulationConfigurationFile";

    // </editor-fold>
    // <editor-fold defaultstate="collapsed" desc="Construction">
    /**
     * Constructs a project including the XML document that constitutes the
     * project, as well as project members
     *
     * @param rootNodeName - Name of the project. Name given to the root node
     * @param load - True if the project should be loaded from disk, false
     * otherwise
     * @throws ParserConfigurationException
     * @throws java.io.IOException
     * @throws org.xml.sax.SAXException
     */
    public ProjectManager(String rootNodeName, boolean load)
            throws ParserConfigurationException, IOException, SAXException {
        initState();
        project.setProjectName(rootNodeName);
        if (load) {
            project.load(project.determineProjectOutputLocation()
                    + project.getProjectName() + ".xml");
        }
    }

    private void initState() {
        project = new Project("None");
    }

    //</editor-fold>
    // <editor-fold defaultstate="collapsed" desc="Persistence">
    /**
     * Writes the document representing this project to disk
     *
     * @return The full path to the file that was persisted
     * @throws TransformerConfigurationException
     * @throws TransformerException
     * @throws java.io.IOException
     * @throws javax.xml.parsers.ParserConfigurationException
     */
    public String persist()
            throws TransformerConfigurationException, TransformerException,
            IOException, ParserConfigurationException {
        return project.persist();
    }

    // </editor-fold>
    // <editor-fold defaultstate="collapsed" desc="ProjectMgr Configuration">
    /**
     * Determines the folder location for storing provenance data for a given
     * project
     *
     * @return The path to the provenance folder for the specified project
     * @throws IOException
     */
    public String determineProvOutputLocation()
            throws IOException {
        String projectDirectory = project.determineProjectOutputLocation();
        String provOutputLocation = projectDirectory
                + "provenance"
                + FileManager.getFileManager().getFolderDelimiter();
        return provOutputLocation;
    }

    /**
     * Determines the assumed folder location for a project of a given name
     *
     * @return The path to the project folder for the specified project
     * @throws IOException
     */
    public static final String determineProjectOutputLocation(String name)
            throws IOException {
        String workingDirectory = FileManager.getCanonicalWorkingDirectory();
        String ps = FileManager.getFileManager().getFolderDelimiter();
        String projectDirectory = workingDirectory + ps + "projects" + ps
                + name + ps;
        return projectDirectory;
    }

    // </editor-fold>
    // <editor-fold defaultstate="collapsed" desc="Getters/Setters">
    /**
     * Sets the project's name. This will also modify the name attribute for the
     * project element of the project XML model
     *
     * @param name The name given to the project
     */
    public void setName(String name) {
        project.setProjectName(name);
    }

    /**
     * Provides the name of the project
     *
     * @return The name of the project
     */
    public String getName() {
        return project.getProjectName();
    }

    /**
     * Sets the value that used to determine if provenance support is enabled
     * for this project. Also sets the value of the related attribute for the
     * project element in the project XML
     *
     * @param enabled Whether of not this project should record provenance
     */
    public void setProvenanceEnabled(boolean enabled) {
        ProjectData provData = new ProjectData(provTagName);
        provData.addAttribute(provEnabledAttributeName, String.valueOf(enabled));
        project.addProjectData(provData);
    }

    /**
     * Indicates if provenance support is enabled for this project
     *
     * @return True if provenance support is enabled, otherwise false
     */
    public boolean isProvenanceEnabled() {
        boolean provEnabled = false;
        ProjectData provData = project.getProjectData(provTagName);
        if (provData != null) {
            String provEnabledAttribute = provData.getAttribute(provEnabledAttributeName);
            if (provEnabledAttribute != null) {
                provEnabled = Boolean.getBoolean(provEnabledAttribute);
            }
        }
        return provEnabled;
    }

    /**
     * Determines if prerequisite project data is available in order to generate
     * a script
     *
     * @return True if the prerequisite data is available for generating a
     * script, false if not
     */
    public boolean scriptGenerationAvailable() {
        ProjectData scriptData = project.getProjectData(scriptTagName);
        ProjectData simulatorData = project.getProjectData(simulatorTagName);
        Datum simConfigData = null;
        if (scriptData != null) {
            simConfigData = scriptData.getDatum(scriptFileTagName);
        }
        return simulatorData != null && simConfigData != null;
    }

    // </editor-fold>
    // <editor-fold defaultstate="collapsed" desc="Data Manipulation">
    /**
     * Provides the current simulation specification based on the content of the
     * elements in the project XML document
     *
     * Note: guaranteed to be successful... If the simSpec has yet to be added
     * to the project, it is automatically added here.
     *
     * @return A simulation specification as described by the text of related
     * elements in the project
     */
    public SimulationSpecification getSimulationSpecification() {
        SimulationSpecification simSpec = new SimulationSpecification();
        ProjectData simData = project.getProjectData(simulatorTagName);

        String simType = getSimulationType(simData);
        String codeLocation = getSimulatorCodeLocation(simData);
        String locale = getSimulatorLocale(simData);
        String folder = getSimulatorFolderLocation(simData);
        String hostname = getSimulatorHostname(simData);
        String sha1 = getSHA1Key(simData);
        String buildOption = getBuildOption(simData);
        String updating = getSimulatorSourceCodeUpdatingType(simData);
        String version = getSimulatorVersionAnnotation(simData);
        String executable = null;
        if (simType != null && !simType.isEmpty()) {
            executable = SimulationSpecification.getSimFilename(simType);
        }
        simSpec.setSimulationType(simType);
        simSpec.setCodeLocation(codeLocation);
        simSpec.setSimulatorLocale(locale);
        simSpec.setSimulatorFolder(folder);
        simSpec.setHostAddr(hostname);
        simSpec.setSHA1CheckoutKey(sha1);
        simSpec.setBuildOption(buildOption);
        simSpec.setSourceCodeUpdating(updating);
        simSpec.setVersionAnnotation(version);
        simSpec.setSimExecutable(executable);
        return simSpec;
    }

    /**
     * Provides the locale for the simulator. In other words the relationship
     * between where a simulation should take place and the machine running the
     * workbench. Since the simulator locale is set based on values from
     * SimulationSpecification, the return value is indirectly dependent upon
     * the definitions provided by the SimulationSpecification class.
     *
     * @return The locale for simulations associated with this project
     * @see edu.uwb.braingrid.workbench.model.SimulationSpecification
     */
    private String getSimulatorLocale(ProjectData simData) {
        return getChildDataContent(simData, simulatorExecutionMachine);
    }

    /**
     * Provides the version annotation for the simulation specification
     * associated with this project.
     *
     * @return The version annotation for the simulation
     */
    private String getSimulatorVersionAnnotation(ProjectData simData) {
        return getChildDataContent(simData, simulatorVersionAnnotationTagName);
    }

    /**
     * Provides the repository location (local or otherwise) for the code for
     * compiling the simulator binaries.
     *
     * @return The central location (possibly a repository URL or URI) where the
     * code resides for compiling the simulator binaries
     */
    private String getSimulatorCodeLocation(ProjectData simData) {
        return getChildDataContent(simData, simulatorCodeLocationTagName);
    }

    /**
     * Provides the folder location where the simulator code is moved to and the
     * simulator is built and executed.
     *
     * Note: This may be an absolute or canonical path on the local file system,
     * or it may be a path on a remote machine relative to the starting path of
     * a remote connection.
     *
     * @return The folder location where the simulator code is moved to and the
     * simulator is built and executed.
     */
    private String getSimulatorFolderLocation(ProjectData simData) {
        return getChildDataContent(simData, simFolderTagName);
    }

    /**
     * Provides the host name of the machine where a simulation will be run.
     * This value is not guaranteed to be set for all simulations. If the
     * simulation is set to be run locally, this function will return a null
     * value.
     *
     * @return The host name of the machine where a simulation will be run, or
     * null if the simulation is set to run locally.
     */
    private String getSimulatorHostname(ProjectData simData) {
        return getChildDataContent(simData, hostnameTagName);
    }

    private String getSHA1Key(ProjectData simData) {
        return getChildDataContent(simData, SHA1KeyTagName);
    }

    private String getBuildOption(ProjectData simData) {
        return getChildDataContent(simData, buildOptionTagName);
    }

    /**
     * Provides the source code updating type for the simulation. Possible
     * values are indirectly dependent on the SimulationSpecification model.
     * Generally, this determines whether or not to update the code in the
     * folder location where the simulator will be compiled and run prior to
     * executing the simulation.
     *
     * @return The source code updating type for the simulation
     * @see edu.uwb.braingrid.workbench.model.SimulationSpecification
     */
    private String getSimulatorSourceCodeUpdatingType(ProjectData simData) {
        return getChildDataContent(simData, simulatorSourceCodeUpdatingTagName);
    }

    /**
     * Provides the simulation type associated with the simulation for this
     * project. The possible values are indirectly determined by the Simulation
     * Specification. In general, these values indicate the processing model for
     * the simulation (The threading or core model). This value can be used to
     * determine which executable to invoke in running the simulation.
     *
     * @return The simulation type associated with the simulation for this
     * project
     * @see
     * edu.uwb.braingrid.workbench.model.SimulationSpecification.SimulatorType
     */
    private String getSimulationType(ProjectData simData) {
        return getChildDataContent(simData, simulationTypeTagName);
    }

    private String getChildDataContent(ProjectData parentData, String tagname) {
        String content = null;
        if (parentData != null) {
            content = parentData.getDatum(tagname).getContent();
            if (content.isEmpty()) {
                content = null;
            }
        }
        return content;
    }

    private void setChildDataContent(ProjectData parentData, String tagname, String content) {
        if (parentData != null) {
            parentData.addDatum(tagname, content, null).setContent(content);
        }
    }

    /**
     * Note: This should not be used to determine if the script has completed
     * execution
     *
     * @return
     */
    public ScriptHistory getScriptHistory() {
        ScriptHistory scriptHistory = new ScriptHistory();
        ProjectData scriptData = project.getProjectData(scriptTagName);
        String startedAt = getScriptTimeStarted(scriptData);
        String completedAt = getScriptTimeCompleted(scriptData);
        boolean outputAnalyzed = wasScriptAnalyzed(scriptData);
        boolean ran = hasScriptRun(scriptData);
        String filename = getScriptFilename(scriptData);
        int version = getScriptVersion(scriptData);

        scriptHistory.setStartedAt(startedAt);
        scriptHistory.setCompletedAt(completedAt);
        scriptHistory.setOutputAnalyzed(outputAnalyzed);
        scriptHistory.setRan(ran);
        scriptHistory.setFilename(filename);
        scriptHistory.setVersion(version);
        return scriptHistory;
    }

    private String getScriptTimeStarted(ProjectData scriptData) {
        return scriptData.getAttribute(scriptRanAtAttributeName);
    }

    private String getScriptTimeCompleted(ProjectData scriptData) {
        return scriptData.getAttribute(scriptCompletedAtAttributeName);
    }

    private boolean wasScriptAnalyzed(ProjectData scriptData) {
        return Boolean.valueOf(scriptData.getAttribute(scriptAnalyzedAttributeName));
    }

    private boolean hasScriptRun(ProjectData scriptData) {
        return Boolean.valueOf(scriptData.getAttribute(scriptRanRunAttributeName));
    }

    private String getScriptFilename(ProjectData scriptData) {
        return getChildDataContent(scriptData, scriptFileTagName);
    }

    private int getScriptVersion(ProjectData scriptData) {
        int version;
        try {
            version = Integer.valueOf(getChildDataContent(scriptData,
                    scriptVersionVersionTagName));
        } catch (NumberFormatException e) {
            version = defaultScriptVersion;
        }
        return version;
    }

    /**
     * Provides the version of the script currently associated with the project.
     * This value can be used to determine the base name of the script file name
     *
     * Note: This should not be used to determine if the script has completed
     * execution
     *
     * @return The version of the script currently associated with the project
     */
    public String getScriptVersion() {
        return String.valueOf(getScriptVersion(project.getProjectData(scriptTagName)));
    }

    /**
     * Sets the text content of the script version for the project. This value
     * is only changed in the XML document for the project
     *
     * @param version - The version number of the current script for the project
     */
    public boolean setScriptVersion(String version) {
        boolean success = true;
        String versionCopy = version;
        try {
            int versionNumber = Integer.getInteger(versionCopy);
            if (versionNumber >= 0 && versionNumber <= Integer.MAX_VALUE) {
                setChildDataContent(project.getProjectData(scriptTagName), scriptVersionVersionTagName, version);
            } else {
                success = false;
            }
        } catch (NumberFormatException e) {
            success = false;
        }
        return success;
    }

    /**
     * Sets the value for the attribute used to determine whether the script has
     * run or not.
     *
     * Note: guaranteed to be successful... If the script has yet to be added to
     * the project, it is automatically added here.
     *
     * @param hasRun Whether or not the script has been executed
     */
    public void setScriptRan(boolean hasRun) {
        setChildDataContent(project.getProjectData(scriptTagName), scriptRanRunAttributeName, String.valueOf(hasRun));
    }

    /**
     * Determines whether or not the script has been executed
     *
     *
     * Note: If the script has yet to be added to the project, it is
     * automatically added here.
     *
     * Note: This should not be used to determine if the script has completed
     * execution
     *
     * @return True if the script has been executed, otherwise false
     */
    public boolean getScriptRan() {
        ProjectData script = project.getProjectData(scriptTagName);
        return Boolean.valueOf(script.getAttribute(scriptRanRunAttributeName));
    }

    /**
     *
     * Note: guaranteed to be successful... If the script has yet to be added to
     * the project, it is automatically added here. Sets the attribute used to
     * determine whether or not the script has been executed to "true"
     */
    public void setScriptRanAt() {
        ProjectData script = project.getProjectData(scriptTagName);
        script.addAttribute(scriptRanAtAttributeName,
                String.valueOf(new Date().getTime()));
    }

    /**
     * Sets the attribute used to determine when the script completed execution.
     *
     * Note: guaranteed to be successful... If the script has yet to be added to
     * the project, it is automatically added here.
     *
     * Note: This should be verified through the OutputAnalyzer class first.
     *
     * @param timeCompleted - The number of milliseconds since January 1, 1970,
     * 00:00:00 GMT when execution completed for the script associated with this
     * project
     */
    public void setScriptCompletedAt(long timeCompleted) {
        ProjectData script = project.getProjectData(scriptTagName);
        script.addAttribute(scriptCompletedAtAttributeName,
                String.valueOf(timeCompleted));
    }

    /**
     * Provides the number of milliseconds since January 1, 1970, 00:00:00 GMT
     * when the execution started for the script associated with this project
     *
     * Note: guaranteed to be successful... If the script has yet to be added to
     * the project, it is automatically added here.
     *
     * @return The number of milliseconds since January 1, 1970, 00:00:00 GMT
     * when the execution started for the script associated with this project
     */
    public long getScriptRanAt() {
        String millisText;
        long millis = DateTime.ERROR_TIME;
        ProjectData script = project.getProjectData(scriptTagName);
        millisText = script.getAttribute(scriptRanAtAttributeName);
        if (millisText != null) {
            try {
                millis = Long.parseLong(millisText);
            } catch (NumberFormatException e) {
            }
        }
        return millis;
    }

    /**
     * Provides the number of milliseconds since January 1, 1970, 00:00:00 GMT
     * when execution completed for the script associated with this project
     * 
     * Note: guaranteed to be successful... If the script has yet to be added to
     * the project, it is automatically added here.
     *
     * @return The number of milliseconds since January 1, 1970, 00:00:00 GMT
     * when execution completed for the script associated with this project
     */
    public long getScriptCompletedAt() {
        String millisText;
        long timeCompleted = DateTime.ERROR_TIME;
        ProjectData script = project.getProjectData(scriptTagName);
        millisText = script.getAttribute(scriptCompletedAtAttributeName);
        if (millisText != null) {
            try {
                timeCompleted = Long.parseLong(millisText);
            } catch (NumberFormatException e) {
            }
        }
        return timeCompleted;
    }

    /**
     * Gets the location of the script file
     *
     * @return The location of the script file, or null if no file path is set.
     */
    public String getScriptCanonicalFilePath() {
        String path = null;
        if (script != null) {
            NodeList nl
                    = script.getElementsByTagName(scriptFileTagName);
            if (nl.getLength() > 0) {
                Text scriptPathText
                        = (Text) nl.item(0).getFirstChild();
                if (scriptPathText != null) {
                    path = scriptPathText.getTextContent();
                }
                if (path != null && (path.equals("null")
                        || path.equals(""))) {
                    path = null;
                }
            }
        }
        return path;
    }

    private void removeSimulator() {
        if (null != simulator) {
            simulator.getParentNode().removeChild(simulator);
            simulator = null;
        }
    }

    private void removeSimulationConfigurationFile() {
        if (null != simulationConfigurationFile) {
            simulationConfigurationFile.getParentNode().removeChild(
                    simulationConfigurationFile);
            simulationConfigurationFile = null;
        }
    }

    public void removeScript() {
        if (script != null) {
            script.getParentNode().removeChild(script);
            script = null;
        }
    }

    /**
     * Add a specification for the simulator used in this project to the model.
     * The elements of a specification are used to determine the content of an
     * execution script as well as where it will be executed
     *
     * @param simulatorExecutionLocation - Indicates where the simulator will be
     * executed (e.g. Remote, or Local)
     * @param hostname - The name of the remote host, if the simulator will be
     * executed remotely
     * @param simFolder - The top folder where the script will be deployed to.
     * This also serves as the parent folder of the local copy of the simulator
     * source code.
     * @param simulationType - Indicates which version of the simulator will be
     * executed (e.g. growth, or growth_CUDA
     * @param versionAnnotation - A human interpretable note regarding the
     * version of the simulator that will be executed.
     * @param codeLocation - The location of the repository that contains the
     * code for the simulator
     * @param sourceCodeUpdating - Whether the source code should be updated
     * prior to execution (e.g. Pull, or None). If sourceCodeUpdating is set to
     * first do a pull on the repository, a clone will be attempted first in
     * case the repository has yet to be cloned.
     * @return True if the simulator was added to the model correctly, false if
     * not
     */
    public boolean addSimulator(String simulatorExecutionLocation,
            String hostname, String simFolder, String simulationType,
            String codeLocation, String versionAnnotation,
            String sourceCodeUpdating, String SHA1Key, String buildOption) {
        boolean success = true;
        // remove previously defined simulator
        removeSimulator();
        try {
            /* Create Elements */
            simulator = doc.createElement(simulatorTagName);
            Element simExecLocation
                    = doc.createElement(simulatorExecutionMachine);
            Element versionAnnotationElem
                    = doc.createElement(simulatorVersionAnnotationTagName);
            Element codeLocationElem = doc.createElement(
                    simulatorCodeLocationTagName);
            Element hostnameElem = doc.createElement(hostnameTagName);
            Element simFolderElem = doc.createElement(simFolderTagName);
            Element simulationTypeElem
                    = doc.createElement(simulationTypeTagName);
            Element sourceCodeUpdatingElem
                    = doc.createElement(simulatorSourceCodeUpdatingTagName);
            Element SHA1KeyElem = doc.createElement(SHA1KeyTagName);
            Element buildOptionElem = doc.createElement(buildOptionTagName);

            /* Add Values */
            // create text nodes to add to created elements
            Text simulatorExecutionLocationText
                    = doc.createTextNode(simulatorExecutionLocation);
            Text versionAnnotationText = doc.createTextNode(versionAnnotation);
            Text codeLocationText = doc.createTextNode(codeLocation);
            Text hostnameText = doc.createTextNode(hostname);
            Text simFolderText = doc.createTextNode(simFolder);
            Text simulationTypeText
                    = doc.createTextNode(simulationType);
            Text sourceCodeUpdatingText
                    = doc.createTextNode(sourceCodeUpdating);
            Text sha1keyText = doc.createTextNode(SHA1Key);
            Text buildOptionText = doc.createTextNode(buildOption);

            // attach the text to respective elements
            simExecLocation.appendChild(simulatorExecutionLocationText);
            versionAnnotationElem.appendChild(versionAnnotationText);
            codeLocationElem.appendChild(codeLocationText);
            hostnameElem.appendChild(hostnameText);
            simFolderElem.appendChild(simFolderText);
            simulationTypeElem.appendChild(simulationTypeText);
            sourceCodeUpdatingElem.appendChild(sourceCodeUpdatingText);
            SHA1KeyElem.appendChild(sha1keyText);
            buildOptionElem.appendChild(buildOptionText);

            /* Attach Elements */
            // attach the parameter elements to the input element
            simulator.appendChild(simExecLocation);
            simulator.appendChild(versionAnnotationElem);
            simulator.appendChild(codeLocationElem);
            simulator.appendChild(hostnameElem);
            simulator.appendChild(simFolderElem);
            simulator.appendChild(simulationTypeElem);
            simulator.appendChild(sourceCodeUpdatingElem);
            simulator.appendChild(SHA1KeyElem);
            simulator.appendChild(buildOptionElem);
            // attach the input element to the project element
            root.appendChild(simulator);
        } catch (DOMException e) {
            simulator = null;
            success = false;
        }
        return success;
    }

    public boolean addSimConfigFile(String filename) {
        boolean success = true;
        try {
            removeSimulationConfigurationFile();
            simulationConfigurationFile
                    = doc.createElement(simConfigFileTagName);
            Text configFileText = doc.createTextNode(filename);
            simulationConfigurationFile.appendChild(configFileText);
            root.appendChild(simulationConfigurationFile);
        } catch (DOMException e) {
            simulationConfigurationFile = null;
            success = false;
        }
        return success;
    }

    /**
     * Adds a generated script file to the project
     *
     * @param scriptBasename - The base-name of the file path
     * @param extension - The extension of the file path
     * @return True if the script was added to the model correctly, false if not
     */
    public boolean addScript(String scriptBasename, String extension) {
        boolean success = true;
        try {
            /* Remove Previous Script */
            removeScript();

            /* Create Elements */
            // this will overwrite previously defined script
            script = doc.createElement(scriptTagName);
            // file element of the script element
            Element scriptFile = doc.createElement(scriptFileTagName);

            /* Add Values */
            // text element describing the file location
            Text scriptFileLocation = doc.createTextNode(scriptBasename + "."
                    + extension);

            // 
            /* Attach Elements */
            scriptFile.appendChild(scriptFileLocation);
            script.appendChild(scriptFile);
            root.appendChild(script);

            int version = 0;
            try {
                version = Integer.valueOf(getScriptVersion());
            } catch (NumberFormatException e) { // version not present
                initScriptVersion();
            }
            version++;
            setScriptVersion(String.valueOf(version));
        } catch (DOMException | NumberFormatException e) {
            script = null;
            success = false;
        }
        return success;
    }

    public String getSimConfigFilename() {
        return getFirstChildTextContent(root,
                simConfigFileTagName);
    }

    public boolean scriptGenerated() {
        return script != null;
    }

    public boolean scriptOutputAnalyzed() {
        String analyzedAttributeValue;
        boolean analyzed = false;
        if (script != null) {
            analyzedAttributeValue = script.getAttribute(
                    scriptAnalyzedAttributeName);
            analyzed = Boolean.valueOf(analyzedAttributeValue);
        }
        return analyzed;
    }

    public void setScriptAnalyzed(boolean analyzed) {
        if (script != null) {
            script.setAttribute(scriptAnalyzedAttributeName,
                    String.valueOf(analyzed));
        }
    }

    public void setSimStateOutputFile(String stateOutputFilename) {
        if (simulationConfigurationFile != null) {
            simulationConfigurationFile.setAttribute(
                    simulationConfigurationFileAttributeName,
                    stateOutputFilename);
        }
    }

    public String getSimStateOutputFile() {
        String filename = null;
        if (simulationConfigurationFile != null) {
            filename = simulationConfigurationFile.getAttribute(
                    simulationConfigurationFileAttributeName);
        }
        return filename;
    }
    // </editor-fold>
}
