package edu.uwb.braingrid.workbench;

import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.SftpException;
import edu.uwb.braingrid.data.script.Script;
import edu.uwb.braingrid.data.script.ScriptManager;
import edu.uwb.braingrid.provenance.ProvMgr;
import edu.uwb.braingrid.tools.nledit.ControlFrame;
import edu.uwb.braingrid.workbench.data.InputAnalyzer;
import edu.uwb.braingrid.workbench.project.ProjectMgr;
import edu.uwb.braingrid.workbench.model.SimulationSpecification;
import edu.uwb.braingrid.workbench.ui.InputConfigurationDialog;
import edu.uwb.braingrid.workbench.ui.NewProjectDialog;
import edu.uwb.braingrid.workbench.ui.ProvenanceQueryDialog;
import edu.uwb.braingrid.workbench.ui.SimulatorSpecificationDialog;
import edu.uwb.braingrid.workbench.ui.WorkbenchControlFrame;
import edu.uwb.braingrid.workbench.utils.DateTime;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import org.xml.sax.SAXException;

/**
 * Manages all of the operations for the workbench. In turn, the operations
 * manage instances of the related data.
 *
 * @author Del Davis
 */
public class WorkbenchManager {

    // <editor-fold defaultstate="collapsed" desc="Instance Variables">
    /* Inter-thread Communication */
    // used when running stand-alone NLEdit from the system runtime
    private String msgFromOtherThread;

    /* in-memory file model managers */
    private ProjectMgr project;
    // the provenance manager
    private ProvMgr prov;

    /* Messages for Frame */
    private String messageAccumulator;

    /* Configuration Data */
    private final String folderDelimiter;
    private String NLEditPath;
    private final String rootDir;
    private final String projectsDir;
    private SimulationSpecification simSpec;
    private String simulationConfigurationFile;
    /**
     * Value indicating that an exception occurred during an operation
     */
    public static final int EXCEPTION_OPTION = -2;
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Construction"> 
    /**
     * Responsible for allocating this manager and initializing all members
     */
    public WorkbenchManager() {
        boolean windowsOS = System.getProperty("os.name").
                toLowerCase().startsWith("windows");
        folderDelimiter = windowsOS ? "\\" : "/";
        rootDir = ".";
        messageAccumulator = "";
        msgFromOtherThread = "";
        projectsDir = folderDelimiter + "projects" + folderDelimiter;
        prov = null;
        project = null;
        simSpec = null;
        try {
            String currDir = Paths.get((new java.io.File("."))
                    .getCanonicalPath()).toString();
            String jarName = "NLEdit.jar";
            String ps = folderDelimiter;
            NLEditPath = currDir + ps + "tools" + ps + "NLEdit" + ps + jarName;
        } catch (IOException e) {
            NLEditPath = ": Exception occured during initialization"
                    + " while locating parent of working directory!";
        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Action Helpers">
    /**
     * Creates a new project through the NewProjectDialog
     *
     * @return True if a new project was initialized, otherwise false. Note,
     * failure and cancellation are returned as the same value, with the only
     * difference being the messages that will be delivered through getMsg
     */
    public boolean newProject() {
        boolean success;
        // Ask the user for a new project name (validation in dialogue)
        NewProjectDialog npd = new NewProjectDialog(true);

        if (npd.getSuccess()) {
            success = initProject(npd.getProjectName(), npd.isProvEnabled());
        } else {
            success = false;
            messageAccumulator += "\n" + "New project specification canceled\n";
        }
        return success;
    }

    /**
     * Opens a project from an XML file
     *
     * @return Option from the JFileChooser or EXCEPTION_OPTION from this class
     * indicating that an exception was thrown
     * @see javax.swing.JFileChooser
     */
    public int openProject() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select a Project Specification...");
        File projectsDirectory = getProjectsDirectory();
        chooser.setCurrentDirectory(projectsDirectory);
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML file (*.xml)", "xml");
        chooser.addChoosableFileFilter(filter);
        chooser.setFileFilter(filter);
        int choice = chooser.showOpenDialog(null);
        switch (choice) {
            case JFileChooser.APPROVE_OPTION:
                try {
                    File selectedFile = chooser.getSelectedFile();
                    project = new ProjectMgr(selectedFile.getName(), true);
                    updateSimSpec();
                    if (project.isProvenanceEnabled()) {
                        prov = new ProvMgr(project, true);
                    } else {
                        prov = null;
                    }
                    messageAccumulator += "\n" + "Project: " + project.getName()
                            + " loaded...\n";
                } catch (ParserConfigurationException | IOException |
                        SAXException ex1) {
                    choice = EXCEPTION_OPTION;
                    project = null;
                    prov = null;
                    simSpec = null;
                    messageAccumulator += "\n"
                            + "Project did not load correctly!\n"
                            + ex1.getClass().getSimpleName() + "..."
                            + " occurred\n";
                }
                break;
            // cancel was chosen (can't load project)
            case JFileChooser.CANCEL_OPTION:
                messageAccumulator += "\n"
                        + "Open Project Operation Cancelled\n";
                break;
            // a file system error occurred within the dialog
            case JFileChooser.ERROR_OPTION:
                messageAccumulator += "\n"
                        + "Open project operation encountered an error\n"
                        + "Error occurred within the open file dialog\n";
                break;
        }
        return choice;
    }

    /**
     * Saves the current project to XML. If provenance is enabled, the
     * provenance file is persisted as well.
     *
     * <i>Assumption: This action is unreachable prior to specifying a new
     * project or loading a project from disk</i>
     */
    public void saveProject() {
        String msg = "Unknown";
        if (project != null) {
            try {
                /* Persist ProjectMgr XML */
                String projectFileName = project.persist();
                // part of error-handling message
                msg = projectFileName + project.getName()
                        + ".xml";
                if (project.isProvenanceEnabled()) {
                    persistProvenance();
                }
                messageAccumulator += "\n" + "Project saved to "
                        + projectFileName
                        + "\n";
            } catch (IOException | TransformerException e) {
                messageAccumulator += "\n" + "The project file: " + msg
                        + " could not be created do to: " + "\n"
                        + e.getClass().toString() + "\n";
            }
        }
    }

    /**
     * Starts the neuron list editor. Depending on whether provenance is enabled
     * or not, this may be the workbench version of NLEdit or an external jar
     * located in the tools folder
     *
     * @param parent - The frame to used to simulate modal invocation of NLEdit.
     * In other words the frame that should be disabled while NLEdit is not
     * disposed
     */
    public void launchNLEdit(WorkbenchControlFrame parent) {
        /* RunNLEdit */
        if (project != null && project.isProvenanceEnabled()) {
            runInternalNLEdit(parent);
        } else {
            // reset message for runExternalEdit
            msgFromOtherThread = "";
            runExternalNLEdit();
            /* Display status of operation */
            if (msgFromOtherThread.equals("")) {
                messageAccumulator += "\n"
                        + "External NLEdit tool launched successfully\n";
            } else {
                messageAccumulator += "\n" + msgFromOtherThread + "\n";
            }
        }
    }

    /**
     * Opens input files from any reachable file system location. These files
     * are added to the project and overwrite any previously opened files of the
     * same neuron list type.
     *
     * @return True if at least one input file was added to the project
     * successfully
     */
    public boolean addInputs() {
        boolean inputAdded = false;
        JFileChooser chooser = new JFileChooser(getWorkingDirectory());
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML file (*.xml)", "xml");
        chooser.addChoosableFileFilter(filter);
        chooser.setFileFilter(filter);
        chooser.setMultiSelectionEnabled(true);
        String dialogTitle = "Select Input Files for a Simulation";
        chooser.setDialogTitle(dialogTitle);
        if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            InputAnalyzer.InputType type;
            File[] files = chooser.getSelectedFiles();
            File f;
            for (int i = 0, im = files.length; i < im; i++) {
                f = files[i];
                try {
                    type = InputAnalyzer.getInputType(f.getAbsoluteFile());
                    if (type != InputAnalyzer.InputType.INVALID) {
                        addInputFile(f.getAbsolutePath().toString(), type);
                        inputAdded = true;
                    } else {
                        messageAccumulator += "\n" + f.getAbsolutePath()
                                + " could not be added...\n no supported neuron"
                                + " list detected in file.\n";
                    }
                } catch (ParserConfigurationException | SAXException |
                        IOException e) {
                    messageAccumulator += "\n" + f.getAbsolutePath()
                            + " could not be added due to an error parsing"
                            + " the document...\n";
                }
            }
        } else {
            messageAccumulator += "\n" + "Input Selection Cancelled\n";
        }
        return inputAdded;
    }

    /**
     * Updates the simulation specification for the currently open project based
     * on user inputs entered in a SimulationSpecificationDialog
     *
     * @return True if the user clicked the OkButton in the
     * SimulationSpecificationDialog (which validates required input in order
     * for the action to be performed)
     */
    public boolean specifyScript() {
        String hostAddr;
        SimulatorSpecificationDialog simulator;
        if (simSpec != null) {
            simulator = new SimulatorSpecificationDialog(true, simSpec);
        } else {
            simulator = new SimulatorSpecificationDialog(true);
        }
        boolean success = simulator.getSuccess();
        if (success) {
            simSpec = simulator.toSimulatorSpecification();
            String locale = simSpec.getSimulationLocale();
            String remote = SimulationSpecification.REMOTE_EXECUTION;
            if (locale.equals(remote)) {
                hostAddr = simSpec.getHostAddr();
            } else {
                hostAddr = "";
            }
            project.addSimulator(locale,
                    hostAddr, simSpec.getSimulatorFolder(),
                    simSpec.getSimulationType(),
                    simSpec.getCodeLocation(),
                    simSpec.getVersionAnnotation(),
                    simSpec.getSourceCodeUpdating());
            updateSimSpec();
            messageAccumulator += "\n" + "New simulation specified\n";
        } else {
            messageAccumulator += "\n"
                    + "New simulator specification canceled\n";
        }
        return success;
    }

    /**
     * Analyzes the redirected provenance output from an executed script.
     *
     * @return The time in milliseconds since January 1, 1970, 00:00:00 GMT when
     * the simulator finished execution. DateTime.ERROR_TIME indicates that the
     * simulator has not finished execution
     * @see edu.uwb.braingrid.workbench.utils.DateTime
     */
    public long analyzeScriptOutput() {
        long timeCompleted = DateTime.ERROR_TIME;
        if (project != null && project.isProvenanceEnabled()) {
            try {
                messageAccumulator += "\n"
                        + "Gathering simulation provenance...\n";
                String targetFolder = ScriptManager.getScriptFolder(
                        project.determineProjectOutputLocation());
                ScriptManager scriptMgr = new ScriptManager();
                timeCompleted = scriptMgr.analyzeScriptOutput(simSpec, prov, targetFolder);
                if (timeCompleted != DateTime.ERROR_TIME) {
                    project.setScriptCompletedAt(timeCompleted);
                }
                messageAccumulator += "\n" + "Simulation provenance gathered\n";
            } catch (IOException | JSchException | SftpException e) {
                messageAccumulator += "\n"
                        + "Simulation provenance could not be gathered due to "
                        + e.getClass() + "...\n";
                messageAccumulator += "Exception message: " + e.getMessage();
                //e.printStackTrace();
            }
        } else {
            messageAccumulator += "\n"
                    + "Provenance is not enabled... nothing to analyze.\n";
        }
        return timeCompleted;
    }

    /**
     * Generates a script based on simulator input files and the simulation
     * specification
     *
     * @return True if the script was generated and persisted successfully,
     * otherwise false
     */
    public boolean generateScript() {
        boolean success;
        success = false;
        Script script = ScriptManager.
                generateScript(project.getName(), project.getNextScriptVersion(), simSpec, project.getSimConfigFilename());
        if (script != null) {
            try {
                String projectFolder = project.determineProjectOutputLocation();
                String scriptsFolder = projectFolder + "scripts"
                        + folderDelimiter;
                new File(scriptsFolder).mkdirs();
                String scriptName = getNextScriptName();
                String scriptFilename = scriptsFolder + scriptName;
                script.persist(scriptFilename);
                success = project.addScript(scriptFilename, "sh");
                if (success) {
                    messageAccumulator += "\n" + "Script generated at: "
                            + scriptFilename + ".sh\n";
                } else {
                    throw new Exception();
                }
            } catch (Exception e) {
                success = false;
                messageAccumulator += "\nThe script could not be generated.\n"
                        + e.getClass().toString() + " occurred:" + e.toString()
                        + "\n";
            }
        }
        // if script was constructed
        project.setScriptRan(!success);
        return success;
    }

    /**
     * Runs the last generated script file. This entails moving the script to
     * the directory specified in the last specified simulation specification
     * (which may be to a remote machine). This also entails moving any required
     * files for the successful execution of all commands embedded in the script
     *
     * @return True if all files were uploaded/copied successfully and the
     * script was started, otherwise false
     */
    public boolean runScript() {
        boolean success = false;
        ScriptManager sm = new ScriptManager();
        try {
            String scriptPath = project.getScriptCanonicalFilePath();
            String[] neuronLists = FileManager.getFileManager().getNeuronListFilenames(project.getName());
            success = sm.runScript(simSpec, scriptPath, neuronLists,
                    project.getSimConfigFilename());
            project.setScriptRan(success);
            project.setScriptRanAt();
            messageAccumulator += sm.getOutstandingMessages();
        } catch (JSchException | SftpException |
                IOException e) {
            messageAccumulator += "\n" + "Script did not run do to "
                    + e.getClass() + "...\n";
            messageAccumulator += "Exception message: " + e.getMessage();
        }

        return success;
    }
    //</editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Inter-Frame Communication">
    /**
     * Initializes a new project by setting the name of the current project.
     * Used externally when a new project is specified. Also used when a new
     * project specification is canceled in order to notify the user from the
     * workbench message center.
     *
     * @param name - Name to give the current project (as well as the base name
     * of the file to record the project in.
     * @param provEnabled - True if provenance should be enabled for this
     * project
     * @return
     */
    public boolean initProject(String name, boolean provEnabled) {
        boolean success = true;
        /* Create a new project */
        try {
            // make a new project (with new XML doc model)
            project = new ProjectMgr(name, false);
            messageAccumulator += "\n" + "New project specified\n";

            /* Set Provenance */
            project.setProvenanceEnabled(provEnabled);
            if (provEnabled) {
                try {
                    prov = new ProvMgr(project, false);
                } catch (IOException ex) {
                    messageAccumulator += "\n"
                            + ">Error initializing provenance"
                            + "home directory for this project...\n" + ex
                            + "\n";
                    throw ex;
                }
            } else {
                prov = null;
            }
        } catch (IOException | ParserConfigurationException | SAXException e) {
            success = false;
            messageAccumulator += "\n"
                    + "Exception occurred while constructing project XML"
                    + "\n" + e.toString();
            project = null;
            prov = null;
        }
        return success;
    }

    /**
     * Adds an input file to the project. If provenance support is enabled the
     * provenance of the file's creation is added to the provenance model.
     * InputAnalyzer filenames are also added to the existing file input label.
     *
     * @param uri - The file location
     * @param type - The type of input file
     */
    public void addInputFile(String uri, InputAnalyzer.InputType type) {
        /* add prov */
        if (project.isProvenanceEnabled()) {
            prov.addEntity(uri, type.toString(), false);
        }
        /* add to project */
        String toRemove = project.addInputFile(uri, type);
        messageAccumulator += "\n" + "Adding input: " + uri
                + " to the project...\n";
        if (toRemove != null) {
            messageAccumulator += "\n" + type.toString() + " neuron list input"
                    + toRemove + " replaced: \n" + toRemove
                    + " as a project input" + "\n";
        } else {
            messageAccumulator += "\n" + uri
                    + " successfully added as a project input" + "\n";
        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Data Manipulation">
    private void persistProvenance() {
        if (project != null) {
            try {
                prov.persist(project);
                messageAccumulator += "\n" + "Provenance persisted to: "
                        + prov.getProvFileURI() + "\n";
            } catch (IOException e) {
                messageAccumulator += "\n" + "Unable to persist provenance\n"
                        + e.toString() + "\n";
            }
        } else {
            messageAccumulator += "\n" + "Unable to persist provenance..."
                    + " no project is loaded\n";
        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Private Helpers">
    private void runInternalNLEdit(WorkbenchControlFrame workbench) {
        messageAccumulator += "\n" + "NLEdit launched...";
        ControlFrame.runNLEdit(workbench, this);
    }

    /**
     * Launches the neuron list editor (NLEdit) from an external jar file.
     * NLEdit provides a means to graphically specify Brain Grid simulation
     * input files. InputAnalyzer files represent lists of neurons with regard
     * to their position in a neuron array (e.g. position 12 is x: 1, y: 2 on a
     * 10x10 grid)
     *
     * Note: Launching externally means that NLEdit will not communicate with
     * this frame. Therefore, this method should only be called in the event
     * that provenance support is turned off for the open project.
     */
    private void runExternalNLEdit() {
        (new Runnable() {
            @Override
            public void run() {
                try {
                    String pathToJar = NLEditPath;
                    if (new File(pathToJar).exists()) {
                        msgFromOtherThread = pathToJar;
                        String cmd = "java -jar " + pathToJar;
                        Runtime.getRuntime().exec(cmd);
                        msgFromOtherThread = "";
                    } else {
                        msgFromOtherThread = NLEditPath
                                + " not found";
                    }
                } catch (IOException e) {
                    msgFromOtherThread = e.toString() + "<br>"
                            + msgFromOtherThread;
                }
            }
        }).run();
    }

    /**
     * Delivers a full system-dependent canonical form of the path to the
     * working directory
     *
     * @return A full system-dependent canonical path to the working directory
     */
    public String getWorkingDirectory() {
        String root;
        try {
            root = new File(rootDir).getCanonicalPath();
        } catch (IOException e) {
            root = rootDir;
        }
        return root;
    }

    /**
     * Delivers the full system-dependent canonical form of the path to the
     * projects directory
     *
     * @return The full system-dependent canonical form of the path to the
     * projects directory
     */
    public Path getProjectsDirectoryPath() {
        return Paths.get(getWorkingDirectory() + projectsDir);
    }

    private File getProjectsDirectory() {
        File projectsDirectory = new File(getProjectsDirectoryPath().toString());
        if (!projectsDirectory.exists()) {
            messageAccumulator += "\n"
                    + "The projects directory does not exist.\n"
                    + "Creating projects directory...\n";
            if (!projectsDirectory.mkdirs()) {
                messageAccumulator += "\n"
                        + "The projects directory could not be created at:";
                try {
                    messageAccumulator += "\n"
                            + projectsDirectory.getCanonicalPath() + "\n";
                } catch (IOException e) {
                    messageAccumulator += "\n"
                            + "Unknown path due to an IOException\n";
                }
            }
        }
        return projectsDirectory;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Getters/Setters">
    /**
     * Gets the name of the project that was last specified while opening or
     * creating a new project
     *
     * @return The name of the currently open project
     */
    public String getProjectName() {
        String name;
        if (project != null) {
            name = project.getName();
        } else {
            name = "None";
        }
        return name;
    }

    /**
     * Indicates whether the last simulation specification was set to remote
     * execution
     *
     * @return True if the last simulation specification was set to remote,
     * otherwise false
     */
    public boolean isSimExecutionRemote() {
        boolean remote = false;
        if (project != null) {
            String simulatorExecutionMachine = project.getSimulatorLocale();
            if (simulatorExecutionMachine != null) {
                remote = simulatorExecutionMachine.
                        equals(SimulationSpecification.REMOTE_EXECUTION);
            }
        }
        return remote;
    }

    private String getNextScriptName() {
        String name;
        name = null;
        if (project != null) {
            String version = project.getNextScriptVersion();
            if (version != null) {
                name = "run_v" + version;
            }
        }
        return name;
    }

    /**
     * Retrieves a textual representation of the inputs specified in the
     * currently open project
     *
     * @return An overview of the input files for the project
     */
    public String getSimConfigFileOverview() {
        String labelText = "None";
        if (project != null) {
            String input = project.getSimConfigFilename();
            if (input != null) {
                labelText = input;
            }
        }
        return labelText;
    }

    /**
     * Provides the current simulation specification of the currently open
     * project
     *
     * @return The current simulation specification for the current project
     */
    public SimulationSpecification getSimulationSpecification() {
        return simSpec;
    }

    private void updateSimSpec() {
        simSpec = project.getSimulationSpecification();
    }

    /**
     * Provides the full path, including the filename, to the last script added
     * to the project
     *
     * @return The full path, including the filename, to the last script added
     * to the project
     */
    public String getScriptPath() {
        String path = null;
        if (project != null) {
            path = project.getScriptCanonicalFilePath();
        }
        return path;
    }

    /**
     * Indicates whether a script can be generated based on the presence of a
     * simulation specification and input files required to invoke the
     * simulation
     *
     * @return True if a script can be generated
     */
    public boolean scriptGenerationAvailable() {
        boolean available = false;
        if (project != null) {
            available = project.scriptGenerationAvailable();
        }
        return available;
    }

    /**
     * Indicates whether or not the last script generated has been moved and
     * executed
     *
     * @return True if the last script generated has been moved and executed
     */
    public boolean scriptRan() {
        boolean ran = false;
        if (project != null) {
            ran = project.getScriptRan();
        }
        return ran;
    }

    /**
     * Provides overview text describing the last simulation specified
     *
     * @return Overview text describing the last simulation specified
     */
    public String getSimulationOverview() {
        String overview = "None";
        if (simSpec != null) {
            String simFoldername = simSpec.getSimulatorFolder();
            String simVersionAnnotation = simSpec.getVersionAnnotation();
            String simCodeLocation = simSpec.getCodeLocation();
            overview = "<html>";
            boolean simAttributeAddedToText = false;
            if (simFoldername != null) {
                FileManager fm = FileManager.getFileManager();
                String home = fm.getUserDir();
                overview += "location: " + home + simFoldername;
                simAttributeAddedToText = true;
            }
            if (simVersionAnnotation != null) {
                if (simAttributeAddedToText) {
                    overview += "<br>";
                }
                overview += "version: "
                        + simVersionAnnotation;
                simAttributeAddedToText = true;
            }
            if (simCodeLocation != null) {
                if (simAttributeAddedToText) {
                    overview += "<br>";
                }
                overview += "compiled from: "
                        + simCodeLocation;
            }
            overview += "</html>";
        }
        return overview;
    }

    /**
     * Provides the status of moving and executing the script.
     *
     * @return The status of moving and executing the script in the form of the
     * time when the script was executed, if it was copied and executed
     * successfully, or if not, the default text for this status prior to the
     * attempt to execute
     */
    public String getScriptRunOverview() {
        String scriptRunMsg = "None";
        long runAt;
        if (project != null) {
            runAt = project.getScriptRanAt();
            if (runAt != DateTime.ERROR_TIME) {
                String time = DateTime.getTime(runAt);
                scriptRunMsg = "Script execution started at: " + time;
            }
        }
        return scriptRunMsg;
    }

    /**
     * Provides an overview the analysis process for the output generated by
     * executing a script. In particular, the provenance related output.
     *
     * @return An overview of the analysis process if the output redirected from
     * the script was downloaded/copied successfully and the script finished
     * execution. If the script hasn't finished executing, but the copy/download
     * was successful, then a message indicating that the execution is
     * incomplete is returned. If the script never ran (was not downloaded or
     * uploaded and executed in the first place) the initial text for this
     * status is returned.
     */
    public String getScriptAnalysisOverview() {
        String overview = "None";
        long completedAt;
        if (project != null) {
            completedAt = project.getScriptCompletedAt();
            if (completedAt != DateTime.ERROR_TIME) {
                overview = "Completed at: "
                        + DateTime.getTime(completedAt);
            } else {
                if (project.getScriptRan()) {
                    overview = "Script execution incomplete, try again later.";
                }
            }
        }
        return overview;
    }

    public String getSimConfigFilename() {
        return simulationConfigurationFile;
    }
    // </editor-fold>

    //<editor-fold defaultstate="collapsed" desc="User Communication">
    /**
     * Provides all of the messages that have accumulated since the construction
     * of this manager
     *
     * @return The messages that have accumulated since the construction of this
     * manager
     */
    public String getMessages() {
        return messageAccumulator;
    }
    // </editor-fold>

    public void viewProvenance() {
        ProvenanceQueryDialog pqd = new ProvenanceQueryDialog(true, prov);
    }

    public boolean isProvEnabled() {
        boolean isEnabled = false;

        if (project != null) {
            isEnabled = project.isProvenanceEnabled();
        }

        return isEnabled;
    }

    public boolean configureSimulation() {
        boolean success = true;
        String projectName = getProjectName();
        if (!projectName.equals("None")) {
            String configFilename = project.getSimConfigFilename();
            InputConfigurationDialog icd = new InputConfigurationDialog(projectName, true, configFilename);

            if (success = icd.getSuccess()) {
                simulationConfigurationFile = icd.getBuiltFile();
                project.addSimConfigFile(simulationConfigurationFile);
                if (project.isProvenanceEnabled()) {
                    // add the config file, but it should be remote
                    // this means that there will need to be a trace when copied
                    prov.addEntity(simulationConfigurationFile, null, false);
                }
            } else {
                simulationConfigurationFile = "None";
            }
        }
        return success;
    }

    public boolean scriptGenerated() {
        boolean generated = false;
        if (project != null) {
            generated = project.scriptGenerated();
        }
        return generated;
    }
}
