package edu.uwb.braingrid.workbench.project;

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.model.SimulationSpecification;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.IOException;

public class ProjectMgrTest {

    @Test
    // Test Constructor
    public void constructorTest() {
        //////////////
        // Valid Cases
        //////////////
        // New Project
        ProjectMgr pm = getPmNameFalseLoad();
        Assertions.assertNotEquals(null, pm);
        // Load project
        pm = getPmNameTrueLoadActualProject();
        Assertions.assertNotEquals(null, pm);

        ////////////////
        // Exceptions
        ////////////////
        // Load a nonexistant project
        pm = getPmNameTrueLoadNotAProject();
        Assertions.assertEquals(null, pm);
        // Null string as a name, new project
        pm = getPmNullNameFalseLoad();
        Assertions.assertEquals(null, pm);
        // Null string as a name, load project
        pm = getPmNullNameTrueLoad();
        Assertions.assertEquals(null, pm);
    }

    @Test
    // Test Load
    public void loadTest() {
        //////////////
        // Valid Cases
        //////////////
        // New Project
        ProjectMgr pmNew = getPmNameFalseLoad();
        try {
            pmNew.load("not-gunna-be-a-file");
            Assertions.fail("An io exception should be thrown. This line should not be reached");
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (IOException e) {

        }

        // Load project
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        try {
            pmLoad.load("not-gunna-be-a-file");
            Assertions.fail("An io exception should be thrown. This line should not be reached");
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (IOException e) {

        }

        // TODO: ParserConfigurationExceptin and SAXException not tested
    }

    @Test
    public void loadPersistTest() {
        // New Project
        String newProjectName = "NewProjectToSave";
        ProjectMgr pmNew = null;
        try {
            pmNew = new ProjectMgr(newProjectName, false);
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        }

        try {
            Assertions.assertEquals(pmNew.getProjectFilename(), pmNew.persist());
        } catch (TransformerException e) {
            Assertions.fail("Transformer Exception");
            e.printStackTrace();
        } catch (IOException e) {
            Assertions.fail("IOException");
            e.printStackTrace();
        }

        // Load project
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        try {
            Assertions.assertEquals(pmLoad.getProjectFilename(), pmLoad.persist());
        } catch (TransformerException e) {
            Assertions.fail("Transformer Exception");
            e.printStackTrace();
        } catch (IOException e) {
            Assertions.fail("IOException");
            e.printStackTrace();
        }

        // TODO: ParserConfigurationExceptin and SAXException not tested
    }

    @Test
    public void determinieProjectOutputLocationTest() {
        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();
        this.determinieProjectOutputLocationTestHelper(pmNew);

        // Load Prj
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        this.determinieProjectOutputLocationTestHelper(pmLoad);
    }

    private void determinieProjectOutputLocationTestHelper(ProjectMgr pm) {
        String workingDirectory = null;
        try {
            workingDirectory = FileManager.getCanonicalWorkingDirectory();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String ps = FileManager.getFileManager().getFolderDelimiter();
        String projectDirectory = workingDirectory + ps + "projects" + ps
                + pm.getName() + ps;
        try {
            Assertions.assertEquals(projectDirectory, pm.determineProjectOutputLocation());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Test
    public void determineProvOutputLocationTest() {

        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();
        this.determineProvOutputLocationTestHelper(pmNew);

        // Load Prj
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        this.determineProvOutputLocationTestHelper(pmLoad);


    }

    private void determineProvOutputLocationTestHelper(ProjectMgr pm) {
        String workingDirectory = null;
        try {
            workingDirectory = FileManager.getCanonicalWorkingDirectory();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String ps = FileManager.getFileManager().getFolderDelimiter();
        String projectDirectory = workingDirectory + ps + "projects" + ps
                + pm.getName() + ps;

        try {
            Assertions.assertEquals(projectDirectory, ProjectManager.determineProjectOutputLocation(pm.getName()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void scriptGenerationAvailableTest() {
        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();
        this.addSimulatorTo(pmNew);
        pmNew.removeScript();
        pmNew.addSimConfigFile("Example");
        Assertions.assertTrue(pmNew.scriptGenerationAvailable());
        //Assertions.fail("Sob violently. Ended here for today.");
    }

    @Test
    public void addSimulatorTest() {
        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();
        Assertions.assertTrue(pmNew.addSimulator(
                ProjectMgr.LOCAL_EXECUTION,
                "",
                "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
                        "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
                        "ridRepos", SimulationSpecification.SimulatorType.SEQUENTIAL.toString(), "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
                        "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
                        "ridRepos", "1.0.0",
                SimulationSpecification.GIT_NONE,
                "",
                SimulationSpecification.PRE_BUILT_BUILD_OPTION));


    }

    @Test
    public void removeScriptTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void addScriptTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void scriptGeneratedTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void scriptOutputAnalyzedTest() {
        Assertions.fail("Need to test still");
    }



    // <editor-fold defaultstate="collapsed" desc="Getters/Setter Tests">
    @Test
    public void getProjectFileNameTest() {
        // New Project
        ProjectMgr pmNew = getPmNameFalseLoad();
        try {
            Assertions.assertEquals(pmNew.determineProjectOutputLocation() + pmNew.getName() + ".xml", pmNew.getProjectFilename());
        } catch (IOException e) {

        }

        // Load Project
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        try {
            Assertions.assertEquals(pmLoad.determineProjectOutputLocation() + pmLoad.getName() + ".xml", pmLoad.getProjectFilename());
        } catch (IOException e) {

        }

    }

    @Test
    public void setAndGetNameTest() {
        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();

        String newName = "NanananananananaBatman";
        pmNew.setName(newName);
        Assertions.assertEquals(newName, pmNew.getName());


        // Load Prj
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        pmLoad.setName(newName);
        Assertions.assertEquals(newName, pmLoad.getName());

    }

    @Test
    public void setAndGetProvenanceEnabledTest() {

        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();

        pmNew.setProvenanceEnabled(false);
        Assertions.assertFalse(pmNew.isProvenanceEnabled());
        pmNew.setProvenanceEnabled(true);
        Assertions.assertTrue(pmNew.isProvenanceEnabled());


        // Load Prj
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();
        pmLoad.setProvenanceEnabled(false);
        Assertions.assertFalse(pmLoad.isProvenanceEnabled());
        pmLoad.setProvenanceEnabled(true);
        Assertions.assertTrue(pmLoad.isProvenanceEnabled());
    }

    @Test
    public void getSimulationSpecificationTest() {
        ProjectMgr pmNew = getPmNameFalseLoad();
        pmNew.addSimulator(
                ProjectMgr.LOCAL_EXECUTION,
                "",
                "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
                        "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
                        "ridRepos", SimulationSpecification.SimulatorType.SEQUENTIAL.toString(), "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
                        "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
                        "ridRepos", "1.0.0",
                SimulationSpecification.GIT_NONE,
                "",
                SimulationSpecification.PRE_BUILT_BUILD_OPTION);
        this.getSimulationSpecificationTestHelper(pmNew);

        // No simulator specified
        ProjectMgr pm = getPmNameFalseLoad();
        this.getSimulationSpecificationTestHelper(pm);
    }

    private void getSimulationSpecificationTestHelper(ProjectMgr pm) {
        String simType = pm.getSimulationType();
        String codeLocation = pm.getSimulatorCodeLocation();
        String locale = pm.getSimulatorLocale();
        String folder = pm.getSimulatorFolderLocation();
        String hostname = pm.getSimulatorHostname();
        String sha1 = pm.getSHA1Key();
        String buildOption = pm.getBuildOption();
        String updating = pm.getSimulatorSourceCodeUpdatingType();
        String version = pm.getSimulatorVersionAnnotation();
        String executable = null;
        if (simType != null && !simType.isEmpty()) {
            executable = SimulationSpecification.getSimFilename(simType);
        }

        SimulationSpecification ss = new SimulationSpecification();
        ss.setSimulationType(simType);
        ss.setCodeLocation(codeLocation);
        ss.setSimulatorLocale(locale);
        ss.setSimulatorFolder(folder);
        ss.setHostAddr(hostname);
        ss.setSHA1CheckoutKey(sha1);
        ss.setBuildOption(buildOption);
        ss.setSourceCodeUpdating(updating);
        ss.setVersionAnnotation(version);
        ss.setSimExecutable(executable);

        Assertions.assertTrue(pm.getSimulationSpecification().equals(ss));
    }

    @Test
    public void getSimulatorLocaleTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(locale, pm.getSimulatorLocale());
    }

    @Test
    public void getSimulatorVersionAnnotationTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(version, pm.getSimulatorVersionAnnotation());
    }

    @Test
    public void getSimulatorCodeLocationTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(simfolder, pm.getSimulatorCodeLocation());
    }

    @Test
    public void getSimulatorFolderLocationTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(codeLocation, pm.getSimulatorCodeLocation());
    }

    @Test
    public void getSimulatorHostnameTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(hostname, pm.getSimulatorHostname());
    }

    @Test
    public void getSetSHA1Key() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(sha1, pm.getSHA1Key());

        Assertions.fail("Not tested set yet");
    }

    @Test
    public void getSetBuildOptionTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(buildOption, pm.getBuildOption());

        Assertions.fail("Not tested set yet");
    }

    @Test
    public void getSimulatorSourceCodeUpdatingType() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(sourceCodeOption, pm.getSimulatorSourceCodeUpdatingType());
    }

    @Test
    public void getSimulationTypeTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(simulatorType.toString(), pm.getSimulationType());
    }

    @Test
    public void getScriptVersionTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(Integer.toString(0),pm.getScriptVersion());
    }

    @Test
    public void getNextScriptVersionTest() {
        ProjectMgr pm = getPmNameFalseLoad();
        this.addSimulatorTo(pm);
        Assertions.assertEquals(Integer.toString(1), pm.getNextScriptVersion());
    }

    @Test
    public void setGetScriptRanTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void setScriptRanAtTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void setGetScriptCompletedAtTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void getSetScriptHostnameTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void getScriptCanonicalFilePathTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void getSimConfigFilename() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void setScriptAnalyzedTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void setSimSTateOutputFileTest() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void getSimStateOutputFileTest() {
        Assertions.fail("Need to test still");
    }
    // </editor-fold>



    // <editor-fold defaultstate="collapsed" desc="Simulator"
    private String locale = ProjectMgr.LOCAL_EXECUTION;
    private String hostname = "";
    private String simfolder = "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
            "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
            "ridRepos";
    private String codeLocation = "C:\\Users\\Max\\Documents\\DOCUMENTS\\Braingrid-WD\\" +
            "BrainGrid\\Tools\\Workbench\\WorkbenchProject\\BrainG" +
            "ridRepos";
    private SimulationSpecification.SimulatorType simulatorType = SimulationSpecification.SimulatorType.SEQUENTIAL;
    private String version = "1.0.0";
    private String sourceCodeOption = SimulationSpecification.GIT_NONE;
    private String sha1 = "";
    private String buildOption = SimulationSpecification.PRE_BUILT_BUILD_OPTION;

    private void addSimulatorTo(ProjectMgr pm) {
        pm.addSimulator(
                locale,
                hostname,
                simfolder,
                simulatorType.toString(),
                codeLocation, version,
                sourceCodeOption,
                sha1,
                buildOption);
    }

    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Factories">

    // name, true, invalid project
    private ProjectMgr getPmNameTrueLoadActualProject() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr("Example", true);

        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        }

        IOException e;
        return pm;
    }

    private ProjectMgr getPmNameTrueLoadNotAProject() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr("NotARealProject", true);
            Assertions.fail("IO Exception expected to be thrown. This line should not have been reached.");
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {

        } catch (SAXException e) {
            e.printStackTrace();
        }

        IOException e;
        return pm;
    }

    // name, false
    private ProjectMgr getPmNameFalseLoad() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr("NewProject", false);
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        }
        return pm;
    }

    // null name, true
    private ProjectMgr getPmNullNameTrueLoad() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr(null, true);
            Assertions.fail("NullPointerException expected to be thrown. This line should not have been reached.");
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {

        }
        return pm;
    }


    // null name, false
    private ProjectMgr getPmNullNameFalseLoad() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr(null, false);
            Assertions.fail("NullPointerException expected to be thrown. This line should not have been reached.");
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {

        }
        return pm;
    }

    //</editor-fold>

}
