package edu.uwb.braingrid.workbench;
/////////////////CLEANED

import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.SftpException;
import edu.uwb.braingrid.data.script.Script;
import edu.uwb.braingrid.data.script.ScriptManager;
import edu.uwb.braingrid.provenance.ProvMgr;
import edu.uwb.braingrid.tools.nledit.ControlFrame;
import edu.uwb.braingrid.workbench.data.InputAnalyzer;
import edu.uwb.braingrid.workbench.project.ProjectMgr;
import edu.uwb.braingrid.workbench.model.SimulationSpecification;
import edu.uwb.braingrid.workbench.ui.DynamicInputConfigurationDialog;
import edu.uwb.braingrid.workbench.ui.InputConfigClassSelectionDialog;
import edu.uwb.braingrid.workbench.ui.NewProjectDialog;
import edu.uwb.braingrid.workbench.ui.ProvenanceQueryDialog;
import edu.uwb.braingrid.workbench.ui.ScriptSpecificationDialog;
import edu.uwb.braingrid.workbench.ui.WorkbenchControlFrame;
import edu.uwb.braingrid.workbench.utils.DateTime;
import java.io.File;
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
    private ProjectMgr projectMgr;
    // the provenance manager
    private ProvMgr prov;

    /* Messages for Frame */
    private String messageAccumulator;

    /* Configuration Data */
    private final String folderDelimiter;
    private final String rootDir;
    private final String projectsDir;
    private SimulationSpecification simSpec;

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
        projectMgr = null;
        simSpec = null;
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
     * Allows the user to configure the input for the simulation.
     *
     * @return True if the user followed through on the specification, False if
     * the user canceled the specification.
     */
    public boolean configureSimulation() {
        boolean success = true;
        String projectName = getProjectName();
        if (!projectName.equals("None")) {
            String configFilename = projectMgr.getSimConfigFilename();
            InputConfigClassSelectionDialog iccsd
                    = new InputConfigClassSelectionDialog(projectName, true, configFilename);
            if (success = iccsd.getSuccess()) {
                DynamicInputConfigurationDialog icd
                        = new DynamicInputConfigurationDialog(projectName, true, configFilename, iccsd.getInputConfigMgr());
                String simulationConfigurationFile = null;
                String stateOutputFilename = null;
                if (success = icd.getSuccess()) {
                    simulationConfigurationFile = icd.getBuiltFile();
                    stateOutputFilename = icd.getStateOutputFilename();
                    if (simulationConfigurationFile != null && stateOutputFilename != null) {
                        projectMgr.addSimConfigFile(simulationConfigurationFile);
                        projectMgr.setSimStateOutputFile(stateOutputFilename);
                        if (projectMgr.isProvenanceEnabled()) {
                            prov.addFileGeneration("simulation_input_file_generation",
                                    null, "workbench", null, false,
                                    simulationConfigurationFile, null, false);
                        }
                    } else {
                        success = false;
                    }
                }
            }
        }
        return success;
    }

    /**
     * Allows the user to query the provenance for the currently open project.
     * Note: In order for this action helper to be invoked, there must be a
     * provenance file associated with a project. Implicitly, a project must be
     * loaded, otherwise this code should not be reachable.
     */
    public void viewProvenance() {
        ProvenanceQueryDialog pqd = new ProvenanceQueryDialog(true, prov);
    }

    /**
     * Opens a project from an XML file
     *
     * @return Option from the JFileChooser or EXCEPTION_OPTION from this class
     * indicating that an exception was thrown
     * @see javax.swing.JFileChooser
     */
    public int openProject() {
        Long functionStartTime = System.currentTimeMillis();
        Long accumulatedTime = 0L;
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
                    try {
                        projectMgr = new ProjectMgr(FileManager.getLastNamePrefix(selectedFile.getName()), true);
                        
                    } catch (IOException ex1) {
                        messageAccumulator += "\n"
                                + "Unmanaged project selected.\n"
                                + "Attempting to import project...\n";
                        String destFolder = ProjectMgr.determineProjectOutputLocation(
                                selectedFile.getName().split("\\.")[0]);
                        FileManager.copyFolder(selectedFile.getParent(),
                                destFolder);
                        messageAccumulator += "\n" + "Folder contents copied..."
                                + "\nFrom: " + selectedFile.getParent()
                                + "\nTo:   "
                                + destFolder + "\n";
                        projectMgr = new ProjectMgr(FileManager.getLastNamePrefix(selectedFile.getName()), true);
                    }
                    updateSimSpec();
                    if (projectMgr.isProvenanceEnabled()) {
                        Long startTime = System.currentTimeMillis();
                        prov = new ProvMgr(projectMgr, true);
                        accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
                    } else {
                        prov = null;
                    }
                    messageAccumulator += "\n" + "Project: "
                            + projectMgr.getName()
                            + " loaded...\n";
                } catch (ParserConfigurationException | IOException |
                        SAXException ex1) {
                    choice = EXCEPTION_OPTION;
                    projectMgr = null;
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
        if (projectMgr != null) {
            DateTime.recordFunctionExecutionTime("WorkbenchManager", "openProject",
                    System.currentTimeMillis() - functionStartTime,
                    projectMgr.isProvenanceEnabled());
            if (projectMgr.isProvenanceEnabled()) {
                DateTime.recordAccumulatedProvTiming("WorkbenchManager", "openProject",
                        accumulatedTime);
            }
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
        Long functionStartTime = System.currentTimeMillis();
        Long accumulatedTime = 0L;
        String msg = "Unknown";
        if (projectMgr != null) {
            try {
                /* Persist ProjectMgr XML */
                String projectFileName = projectMgr.persist();
                // part of error-handling message
                msg = projectFileName + projectMgr.getName()
                        + ".xml";
                if (projectMgr.isProvenanceEnabled()) {
                    Long startTime = System.currentTimeMillis();
                    persistProvenance();
                    accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
                }
                messageAccumulator += "\n" + "Project saved to "
                        + projectFileName
                        + "\n";
            } catch (IOException | TransformerException e) {
                messageAccumulator += "\n" + "The project file: " + msg
                        + " could not be created due to: " + "\n"
                        + e.getClass().toString() + "\n";
            }
        }
        DateTime.recordFunctionExecutionTime("WorkbenchManager", "saveProject",
                System.currentTimeMillis() - functionStartTime,
                projectMgr.isProvenanceEnabled());
        if (projectMgr.isProvenanceEnabled()) {
            DateTime.recordAccumulatedProvTiming("WorkbenchManager", "saveProject",
                    accumulatedTime);
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
        runInternalNLEdit(parent);
    }

    /**
     * Opens input files from any reachable file system location. These files
     * are added to the project and overwrite any previously opened files of the
     * same neuron list type.
     * 
     * NOTE: DEAD CODE
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
        ScriptSpecificationDialog spd;
        if (simSpec != null) {
            spd = new ScriptSpecificationDialog(true, simSpec);
        } else {
            spd = new ScriptSpecificationDialog(true);
        }
        boolean success = spd.getSuccess();
        if (success) {
            simSpec = spd.toSimulatorSpecification();
            String locale = simSpec.getSimulationLocale();
            String remote = SimulationSpecification.REMOTE_EXECUTION;
            if (locale.equals(remote)) {
                hostAddr = simSpec.getHostAddr();
            } else {
                hostAddr = "";
            }
            projectMgr.addSimulator(locale,
                    hostAddr, simSpec.getSimulatorFolder(),
                    simSpec.getSimulationType(),
                    simSpec.getCodeLocation(),
                    simSpec.getVersionAnnotation(),
                    simSpec.getSourceCodeUpdating(),
                    simSpec.getSHA1CheckoutKey(),
                    simSpec.getBuildOption());
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
        if (projectMgr != null) {
            if (!projectMgr.scriptOutputAnalyzed()) {
                ScriptManager scriptMgr = new ScriptManager();
                try {
                    messageAccumulator += "\n"
                            + "Gathering simulation provenance...\n";
                    String targetFolder = ScriptManager.getScriptFolder(
                            projectMgr.determineProjectOutputLocation());
                    timeCompleted = scriptMgr.analyzeScriptOutput(simSpec,
                            projectMgr, prov, targetFolder);
                    if (timeCompleted != DateTime.ERROR_TIME) {
                        projectMgr.setScriptCompletedAt(timeCompleted);
                        projectMgr.setScriptAnalyzed(true);
                    }
                    messageAccumulator += scriptMgr.getOutstandingMessages();
                    messageAccumulator += "\n" + "Simulation provenance gathered\n";
                } catch (IOException | JSchException | SftpException e) {
                    messageAccumulator += scriptMgr.getOutstandingMessages();
                    messageAccumulator += "\n"
                            + "Simulation provenance could not be gathered due to "
                            + e.getClass() + "...\n";
                    messageAccumulator += "Exception message: " + e.getMessage();
                    e.printStackTrace();
                }
            } else {
                messageAccumulator += "\n"
                        + "Script output has already been analyzed for this simulation run"
                        + "\nTo analyze another run, please respecify script or input and run again"
                        + "\n";
            }
        } else {
            messageAccumulator += "\n"
                    + "No project loaded... nothing to analyze.\n";
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
                generateScript(projectMgr.getName(), projectMgr.getNextScriptVersion(), simSpec, projectMgr.getSimConfigFilename());
        if (script != null) {
            try {
                String projectFolder
                        = projectMgr.determineProjectOutputLocation();
                String scriptsFolder = projectFolder + "scripts"
                        + folderDelimiter;
                new File(scriptsFolder).mkdirs();
                String scriptName = getNextScriptName();
                String scriptFilename = scriptsFolder + scriptName;
                script.persist(scriptFilename);
                success = projectMgr.addScript(scriptFilename, "sh");
                if (success) {
                    messageAccumulator += "\n" + "Script generated at: "
                            + scriptFilename + ".sh\n";
                    // this is where prov would be if we didn't want to wait till
                    // script execution to record the script's existence
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
        projectMgr.setScriptRan(!success);
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
            String scriptPath = projectMgr.getScriptCanonicalFilePath();
            String[] neuronLists
                    = FileManager.getFileManager().getNeuronListFilenames(projectMgr.getName());
            success = sm.runScript(prov, simSpec, scriptPath,
                    projectMgr.getScriptVersion(), neuronLists,
                    projectMgr.getSimConfigFilename());
            projectMgr.setScriptRan(success);
            projectMgr.setScriptRanAt();
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
        Long functionStartTime = System.currentTimeMillis();
        Long accumulatedTime = 0L;
        boolean success = true;
        /* Create a new project */
        try {
            // make a new project (with new XML doc model)
            projectMgr = new ProjectMgr(name, false);
            messageAccumulator += "\n" + "New project specified\n";

            /* Set Provenance */
            projectMgr.setProvenanceEnabled(provEnabled);
            if (provEnabled) {
                Long startTime = System.currentTimeMillis();
                try {
                    prov = new ProvMgr(projectMgr, false);
                } catch (IOException ex) {
                    messageAccumulator += "\n"
                            + ">Error initializing provenance"
                            + "home directory for this project...\n" + ex
                            + "\n";
                    throw ex;
                }
                accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
            } else {
                prov = null;
            }
        } catch (IOException | ParserConfigurationException | SAXException e) {
            success = false;
            messageAccumulator += "\n"
                    + "Exception occurred while constructing project XML"
                    + "\n" + e.toString();
            projectMgr = null;
            prov = null;
        }
        DateTime.recordFunctionExecutionTime("WorkbenchManager", "initProject",
                System.currentTimeMillis() - functionStartTime,
                projectMgr.isProvenanceEnabled());
        if (projectMgr.isProvenanceEnabled()) {
            DateTime.recordAccumulatedProvTiming("WorkbenchManager", "initProject",
                    accumulatedTime);
        }
        return success;
    }

    /**
     * Adds an input file to the project. If provenance support is enabled the
     * provenance of the file's creation is added to the provenance model.
     * InputAnalyzer filenames are also added to the existing file input label.
     *
     * NOTE: DEAD CODE
     * 
     * @param uri - The file location
     * @param type - The type of input file
     */
    public void addInputFile(String uri, InputAnalyzer.InputType type) {
//        /* add prov */
//        if (projectMgr.isProvenanceEnabled()) {
//            prov.addEntity(uri, type.toString(), false, false);
//        }
//        /* add to project */
//        //String toRemove = projectMgr.addInputFile(uri, type);
//        messageAccumulator += "\n" + "Adding input: " + uri
//                + " to the project...\n";
//        if (toRemove != null) {
//            messageAccumulator += "\n" + type.toString() + " neuron list input"
//                    + toRemove + " replaced: \n" + toRemove
//                    + " as a project input" + "\n";
//        } else {
//            messageAccumulator += "\n" + uri
//                    + " successfully added as a project input" + "\n";
//        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Data Manipulation">
    private void persistProvenance() {
        if (projectMgr != null) {
            try {
                prov.persist(projectMgr);
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

    /**
     * Sets the ScriptRan attribute of the Project to false. Run invalidation
     * should occur whenever the script specification or simulation
     * specification changes. This attribute is used by the view to update
     * workflow state (which buttons are enabled and what text is shown to the
     * user)
     */
    public void invalidateScriptRan() {
        projectMgr.setScriptRan(false);
    }

    /**
     * Removes the script from the project. Invalidation should occur whenever
     * the script specification or simulation specification changes. This is a
     * safety measure meant to protect against utilizing an expired script (i.e.
     * the version doesn't match, but the script gets used anyway)
     */
    public void invalidateScriptGenerated() {
        projectMgr.removeScript();
    }

    /**
     * Sets the time when the script completed execution to an error code.
     * Invalidation should occur whenever script specification or simulation
     * specification occurs. This is a safety measure for the view in updating
     * the overview of script output analysis.
     */
    public void invalidateScriptAnalyzed() {
        projectMgr.setScriptCompletedAt(DateTime.ERROR_TIME);
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
        if (projectMgr != null) {
            name = projectMgr.getName();
        } else {
            name = "None";
        }
        return name;
    }

    /**
     * Provides the status of whether or not the script has been generated.
     *
     * @return True if the script has been generated, otherwise false (not
     * including the script not being generated after changes were made to the
     * simulation configuration or the script execution directives.
     */
    public boolean scriptGenerated() {
        boolean generated = false;
        if (projectMgr != null) {
            generated = projectMgr.scriptGenerated();
        }
        return generated;
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
        if (projectMgr != null) {
            String simulatorExecutionMachine = projectMgr.getSimulatorLocale();
            if (simulatorExecutionMachine != null) {
                remote = simulatorExecutionMachine.
                        equals(SimulationSpecification.REMOTE_EXECUTION);
            }
        }
        return remote;
    }

    /**
     * Answers a query regarding whether or not provenance is enabled for the
     * currently open project. Note: This implicitly means that a project must
     * be loaded for this code to be reachable.
     *
     * @return True if provenance is enabled for the currently open project,
     * otherwise false.
     */
    public boolean isProvEnabled() {
        boolean isEnabled = false;

        if (projectMgr != null) {
            isEnabled = projectMgr.isProvenanceEnabled();
        }

        return isEnabled;
    }

    private String getNextScriptName() {
        String name;
        name = null;
        if (projectMgr != null) {
            String version = projectMgr.getNextScriptVersion();
            if (version != null) {
                name = ScriptManager.getScriptName(projectMgr.getName(), version);
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
        if (projectMgr != null) {
            String input = projectMgr.getSimConfigFilename();
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
        simSpec = projectMgr.getSimulationSpecification();
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
        if (projectMgr != null) {
            path = projectMgr.getScriptCanonicalFilePath();
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
        if (projectMgr != null) {
            available = projectMgr.scriptGenerationAvailable();
        }
        return available;
    }

    /**
     * Indicates whether or not the last script generated has been moved and
     * executed
     *
     * @return True if the last script generated has been moved and executed,
     * otherwise false
     */
    public boolean scriptRan() {
        boolean ran = false;
        if (projectMgr != null) {
            ran = projectMgr.getScriptRan();
        }
        return ran;
    }

    /**
     * Indicates whether or not the output of script execution has been
     * analyzed.
     *
     * Note: An incomplete analysis results in a false return value.
     *
     * @return True if the output of script execution has been analyzed (and the
     * script execution has completed), otherwise false
     */
    public boolean scriptAnalyzed() {
        boolean analyzed = false;
        if (projectMgr != null) {
            analyzed = projectMgr.scriptOutputAnalyzed();
        }
        return analyzed;
    }

    /**
     * Provides overview text describing the last simulation specified
     *
     * @return Overview text describing the last simulation specified
     */
    public String getSimulationOverview() {
        String overview = "<html>None";
        if (simSpec != null) {
            overview = "<html>";
            String simFoldername = simSpec.getSimulatorFolder();
            String simVersionAnnotation = simSpec.getVersionAnnotation();
            String simCodeLocation = simSpec.getCodeLocation();
            boolean simAttributeAddedToText = false;
            if (simFoldername != null) {
                FileManager fm = FileManager.getFileManager();
                String home;
                if (simSpec.isRemote()) {
                    home = "~/";
                } else {
                    home = fm.getUserDir();
                }
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
        if (projectMgr != null) {
            runAt = projectMgr.getScriptRanAt();
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
        if (projectMgr != null) {
            completedAt = projectMgr.getScriptCompletedAt();
            if (completedAt != DateTime.ERROR_TIME) {
                overview = "Completed at: "
                        + DateTime.getTime(completedAt);
            } else {
                if (projectMgr.getScriptRan()) {
                    overview = "Script execution incomplete, try again later.";
                }
            }
        }
        return overview;
    }

    public ProvMgr getProvMgr() {
        return prov;
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

    public boolean configureParamsClasses() {
        //This function will be able to add/modify/delete parameter classes.
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
