package edu.uwb.braingrid.data.script;

import com.hp.hpl.jena.rdf.model.Resource;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.SftpException;
import edu.uwb.braingrid.provenance.ProvMgr;
import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.comm.SecureFileTransfer;
import edu.uwb.braingrid.workbench.data.OutputAnalyzer;
import edu.uwb.braingrid.workbench.model.ExecutedCommand;
import edu.uwb.braingrid.workbench.model.SimulationSpecification;
import edu.uwb.braingrid.workbench.ui.LoginCredentialsDialog;
import edu.uwb.braingrid.workbench.utils.DateTime;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.List;

/**
 * Manages script creation, script execution, and script output analysis.
 *
 * @author Del Davis
 */
public class ScriptManager {

    private String outstandingMessages;

    public ScriptManager() {
        outstandingMessages = "";
    }

    /**
     * Generates a constructed (but not persisted) Script
     *
     * @param version The version of the script (used in tracing output back to
     * the script that printed the output)
     * @param simSpec - The specification for the simulator to execute in the
     * script
     * @return A constructed script or null in the case that the script could
     * not be constructed properly
     */
    public static Script generateScript(String version, SimulationSpecification simSpec) {
        boolean success;
        // create a new script
        Script script = new Script();

        /* Print Header Data */
        script.printf(Script.versionText, version,
                false);
        // determine which simulator file to execute
        String type = simSpec.getSimulationType();
        String simExecutableToInvoke;
        simExecutableToInvoke = SimulationSpecification.getSimFilename(type);
        success = simExecutableToInvoke != null;
        ////////TODO: Modify the code in this function to use real inputs///////
        printfSimSpecToScript(script, simExecutableToInvoke, true);

        /* Prep From ProjectMgr Data */
        // folder to make/cd to, assumption is that containing folder
        // note use mkdir -p to create any intermediate directories
        String simFolder = simSpec.getSimulatorFolder();
        // central repository location
        String repoURI = simSpec.getCodeLocation();
        // git pull or do we assume it exists as is?
        String simulatorSourceCodeUpdatingType
                = simSpec.getSourceCodeUpdating();
        boolean updateRepo = false;
        if (simulatorSourceCodeUpdatingType != null) {
            updateRepo = simulatorSourceCodeUpdatingType.
                    equals(SimulationSpecification.GIT_PULL_AND_CLONE);
        }

        /* Create Script */
        // add a mkdir that will create intermediate directories
        String[] argsForMkdir = {"-p", simSpec.getSimulatorFolder()};
        script.executeProgram("mkdir", argsForMkdir);
        // do a pull?
        if (updateRepo) {
            // first do a clone and maybe fail
            String[] gitCloneArgs = {"clone", repoURI, simFolder};
            script.executeProgram("git", gitCloneArgs);
            // change directory to do a pull 
            // note: unnecessary with git 1.85 or higher, but git hasn't been 
            // updated in quite some time on the UWB root binaries :(
            String[] cdArg = {simFolder};
            script.executeProgram("cd", cdArg);
            // then do a pull and maybe fail (one of the two will work)
            String[] gitPullArgs = {"pull"};
            script.executeProgram("git", gitPullArgs);
            /* checkout master */
            String[] gitCheckoutMasterArgs = {"checkout", "master"};
            script.executeProgram("git", gitCheckoutMasterArgs);
        }
        // else { do other things for local that wouldn't be done for remote? }

        /* Make the Simulator */
        String[] cleanMakeArgs = {"-p", "clean"};
        String[] makeArgs = {"-p", simExecutableToInvoke};
        script.executeProgram("make", cleanMakeArgs);
        script.executeProgram("make", makeArgs);

        /* Run the Simulator */
        // TODO: change to commented code below when integrated
        //------->//String[] inputFiles = project.getInputFiles();
        //------->//String[] simArgs = new String[inputFiles.length + 3];
        //------->//simArgs[0] = "-t";
        //------->//for (int i = 0, im = inputFiles.length; i < im; i++) {
        //--->//simArgs[i + 1] = InputAnalyzer.getSimpleFilename(inputFiles[i]);
        //------->//}
        //------->//simArgs[inputFiles.length + 1] = "-o";
        //------->//simArgs[inputFiles.length + 2] = "output.xml";
        //------->//script.executeProgram(simExecutableToInvoke, simArgs, true);
        // Hard Coded to short growth Test for now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO: replace the code below with the commented code above after
        // new simulator input format has been implemented in the simulator
        // code
        String[] hardCodedTestArgs = {"-t", "test.xml", "-o", "output.xml"};
        script.executeProgram(simExecutableToInvoke, hardCodedTestArgs, true);

        /* Put Script Together and Save */
        if (!success || !script.construct()) {
            script = null; // or indicate unsuccessful operation
        }
        return script;
    }

    /**
     * Provides a string representation of the path to the folder where script
     * related files are stored
     *
     * @param projectFolder - A string representation of the path to the folder
     * where projects are stored (in relation to where the workbench was invoked
     * this is: ./projects/the_project_name/scripts/. However, this is system
     * dependent)
     * @return A string representation of the path to the folder where script
     * related files are stored. Depending on the form of the project folder
     * provided, this may represent a relative path.
     */
    public static String getScriptFolder(String projectFolder) {
        return projectFolder + "scripts"
                + FileManager.getFileManager().getFolderDelimiter();
    }

    /**
     * Runs the script as specified in the associated simulation specification.
     *
     * @param simSpec - Holds information about the simulation
     * @param scriptPath - indicates where the constructed script to execute
     * resides on the file system.
     * @param inputFilenames - A list of inputs to run the script with
     * @return True if all operations associated with running the script were
     * successful, otherwise false. This does not indicate whether the
     * simulation ran successfully
     * @throws com.jcraft.jsch.JSchException
     * @throws java.io.FileNotFoundException
     * @throws com.jcraft.jsch.SftpException
     */
    public boolean runScript(SimulationSpecification simSpec, String scriptPath, String[] inputFilenames) throws JSchException, FileNotFoundException, SftpException {
        boolean success = false;
        String executionMachine = simSpec.getSimulationLocale();
        String remoteExecution
                = SimulationSpecification.REMOTE_EXECUTION;
        // run script remotely?
        if (executionMachine.equals(remoteExecution)) {
            success = runRemoteScript(simSpec, scriptPath, inputFilenames);
        } else { // or run it locally
            if (!FileManager.getFileManager().isWindowsSystem()) {
                success = runLocalScript(simSpec, scriptPath, inputFilenames);
            }
        }
        return success;
    }

    /**
     * Analyzes script output for provenance data. Relays that data to the
     * provenance manager. Note: This class defines the context between
     * provenance data. The provenance manager is used to connect such data as
     * determined by this function. This means that if script generation changes
     * this function may need to change, in turn, and vice-versa.
     *
     * @param simSpec - Specification used to indicate the context in which the
     * simulation was specified when the script was generated
     * @param prov - Provenance manager used to create provenance based on
     * analysis of the printf output
     * @param outputTargetFolder - location to store the redirected printf
     * output.
     * @return time completed in milliseconds since the epoch, or an error code
     * indicating that the script has not completed
     * @throws com.jcraft.jsch.JSchException
     * @throws com.jcraft.jsch.SftpException
     */
    public long analyzeScriptOutput(SimulationSpecification simSpec, ProvMgr prov,
            String outputTargetFolder) throws JSchException, SftpException {
        long timeCompleted = DateTime.ERROR_TIME;
        String localOutputFilename
                = fetchScriptOutputFile(simSpec, outputTargetFolder);
        if (localOutputFilename != null) {
            OutputAnalyzer analyzer = new OutputAnalyzer();
            analyzer.analyzeOutput(localOutputFilename);
            /* Completed */
            long atTime;
            String simExec = simSpec.getSimExecutable();
            atTime = analyzer.completedAt("./" + simExec);
            timeCompleted = atTime;
            if (timeCompleted != DateTime.ERROR_TIME) {
                /* Set Remote Namespace Prefix */
                if (simSpec.isRemote()) {
                    prov.setNsPrefix("remote", simSpec.getHostAddr());
                }
                /* Simulation */
                ExecutedCommand sim = analyzer.getFirstCommand("./" + simExec);

                if (sim != null) {
                    // get agent resource
                    String uri = simSpec.getSimulatorFolder() + "/" + simExec;
                    Resource simAgent
                            = prov.addSoftwareAgent(uri, "simulator",
                                    simSpec.isRemote());
                    // get activity resource
                    Resource simActivity
                            = prov.addActivity("simulation", null,
                                    simSpec.isRemote());
                    // connect the two
                    Resource associatedWith = prov.associateWith(simActivity,
                            simAgent);
                    prov.atTime(associatedWith, new Date(atTime).toString());
                }
                String scriptName = Script.getFilename(
                        analyzer.getScriptVersion());
                System.err.println(scriptName);
                //git log --pretty=format:'%h' -n 1
                SimulationSpecification spec = analyzer.getSimSpec();
                System.err.println(simSpec.getSimExecutable());
                List<ExecutedCommand> allCommandsList = null;
                Collection<ExecutedCommand> allCommands
                        = analyzer.getAllCommands();
                if (allCommands != null) {
                    allCommandsList = new ArrayList(allCommands);
                }
                if (allCommandsList != null) {
                    for (ExecutedCommand ec : allCommandsList) {
                        System.err.println(ec);
                    }
                }
            }
        }
        return timeCompleted;
    }

    /**
     * Gets the messages that have accumulated within a public function call.
     * The accumulated messages are discarded during this call.
     *
     * @return The messages that have accumulated since the last call to this
     * function in the form of a single String.
     */
    public String getOutstandingMessages() {
        String msg = outstandingMessages;
        outstandingMessages = "";
        return msg;
    }

    private static void printfSimSpecToScript(Script script, String simFile, boolean append) {

        script.printf(SimulationSpecification.simExecText, simFile,
                append);
        ///////////////////////TODO change test args to commented code/////////
        /* Hard Coded Test Input - Change this for Production */
        // printf the inputs
//        String[] inputs = project.getInputFiles();
//        String joinedInputs = "";
//        if (inputs.length > 0) {
//            for (int i = 0, im = inputs.length; i < im; i++) {
//                // first \\t is regex second is string. 
//                // So \t is replaced with \\t
//                joinedInputs += Script.printfEscape(inputs[i]);
//                if (i < im - 1) {
//                    joinedInputs += " ";
//                }
//            }
//        }
        String joinedInputs = "test.xml";
        script.printf(SimulationSpecification.simInputsText, joinedInputs, true);
        // printf the outputs
        script.printf(SimulationSpecification.simOutputsText, "output.xml", true);
        // printf the end tag for the sim spec data
        script.printf(SimulationSpecification.endSimSpecText, "", true);
    }

    private boolean runRemoteScript(SimulationSpecification simSpec, String scriptPath, String[] inputFilenames) throws JSchException, FileNotFoundException, SftpException {
        char[] password = null;
        boolean success = false;
        String hostname = simSpec.getHostAddr();
        // get username and password
        LoginCredentialsDialog lcd = new LoginCredentialsDialog(
                hostname, true);
        if (lcd.okClicked()) {
            SecureFileTransfer sft = new SecureFileTransfer();
            password = lcd.getPassword();
            lcd.clearPassword();
            if (sft.uploadFile(scriptPath, "", hostname, lcd.getUsername(),
                    password, null)) {
                outstandingMessages += "\n" + scriptPath + " uploaded to "
                        + hostname + "\n";
                String filename;
                boolean loopSuccess = true;
                for (int i = 0, im = inputFilenames.length; i < im;
                        i++) {
                    filename = FileManager.
                            getSimpleFilename(inputFilenames[i]);
                    outstandingMessages += "\n" + "Uploading "
                            + inputFilenames[i] + " to " + hostname
                            + "\n";
                    loopSuccess = sft.uploadFile(inputFilenames[i], "",
                            hostname, lcd.getUsername(), password,
                            null);
                    if (!loopSuccess) {
                        success = false;
                        outstandingMessages += "\n" + "Problem uploading "
                                + filename + " to " + hostname + "\n";
                        break;
                    } else {
                        outstandingMessages += "\n" + filename
                                + " uploaded to "
                                + hostname + "\n";
                    }
                }
                if (loopSuccess) {
                    String cmd = "nohup sh ./"
                            + FileManager.getSimpleFilename(scriptPath)
                            + " &";
                    outstandingMessages += "\n" + "Executing " + cmd
                            + " at "
                            + hostname + "\n";
                    sft.executeCommand(cmd, hostname, lcd.getUsername(),
                            password, false);
                    success = true;
                }
            } else {
                outstandingMessages += "\n" + "Failed to upload script to "
                        + hostname + "\n";
            }
        } else {
            outstandingMessages += "\n"
                    + "\nRemote Credentials Specification Canceled\n";
        }
        outstandingMessages += "\n" + "Remote Operations Completed: "
                + new Date()
                + "\n";

        if (password != null) {
            Arrays.fill(password, '0');
        }

        return success;
    }

    // TODO: IMPLEMENT
    // THIS IS A STUB
    private boolean runLocalScript(SimulationSpecification simSpec, String scriptPath, String[] inputFilenames) {
        boolean success = false;
        //success = java.nio.file.Files.copy(, null);
        outstandingMessages += "\n" + "Local script execution disabled" + "\n";
        return success;
    }

    private String fetchScriptOutputFile(SimulationSpecification simSpec,
            String outputStorageFolder) throws JSchException, SftpException {
        String filename = null;
        char[] password = null;
        String copyToFilepath = outputStorageFolder
                + Script.printfOutputFilename;
        // run simulation here or on another machine?
        boolean remote = simSpec.isRemote();
        if (remote) {
            // download prov.txt
            SecureFileTransfer sft = new SecureFileTransfer();
            String hostname = simSpec.getHostAddr();
            LoginCredentialsDialog lcd
                    = new LoginCredentialsDialog(hostname, true);
            password = lcd.getPassword();
            lcd.clearPassword();
            if (sft.downloadFile(Script.printfOutputFilename, copyToFilepath,
                    hostname, lcd.getUsername(), password)) {
                filename = copyToFilepath;
            }
        } else {
            FileManager fm = FileManager.getFileManager();
            String simFolder = simSpec.getSimulatorFolder();
            if (FileManager.copyFile(Paths.get(simFolder
                    + fm.getFolderDelimiter()
                    + Script.printfOutputFilename),
                    Paths.get(copyToFilepath))) {
                filename = copyToFilepath;
            }
        }
        if (password != null) {
            Arrays.fill(password, '0');
        }
        return filename;
    }
}
