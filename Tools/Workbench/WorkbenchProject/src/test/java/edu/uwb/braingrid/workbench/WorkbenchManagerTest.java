package edu.uwb.braingrid.workbench;

import edu.uwb.braingrid.workbench.utils.DateTime;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class WorkbenchManagerTest {

    public void test() {
        Assertions.fail("Need to test still");
    }

    @Test
    public void newProjectTest() {
        // UI Dialog, nothing to test
    }

    @Test
    public void configureSimulationTest() {
        // UI Dialog, nothing to test
    }

    @Test
    public void viewProvenanceTest() {
        // UI Dialog, nothing to test
    }

    @Test
    public void openProjectTest() {
        // UI Dialog, nothing to test
    }

    @Test
    public void saveProjectTest() {
        // UI Dialog, nothing to test
    }

    @Test
    public void launchNLEditTest() {
        // UI launch, nothing to test
    }

    @Test
    public void addInputsTest() {
        // Dependent on JFileChooser
    }

    @Test
    public void specifyScriptTest() {
        // UI launch, nothing to test
    }

    @Test
    public void analyzeScriptOutputTest() {
        analyzeScriptOutputTest1();
        analyzeScriptOutputTest2();
        analyzeScriptOutputTest3();
        analyzeScriptOutputTest4();
        analyzeScriptOutputTest5();
        analyzeScriptOutputTest6();
    }

    // Project Manager is null
    private void analyzeScriptOutputTest1() {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        Assertions.assertEquals(DateTime.ERROR_TIME, wm.analyzeScriptOutput());
    }

    // ProjectManager is not null, and script has been analyzed
    private void analyzeScriptOutputTest2() {
        WorkbenchManager wm = initProjectWorkbenchManagerFactory(true);
        test();
    }


    // ProjectManager is not null, script has not been analyzed
    private void analyzeScriptOutputTest3() {
        test();
    }

    // ProjectManager is not null, script has not been analyzed - IOException
    private void analyzeScriptOutputTest4() {
        test();
    }

    // ProjectManager is not null, script has not been analyzed- JSchException
    private void analyzeScriptOutputTest5() {
        test();
    }

    // ProjectManager is not null, script has not been analyzed - SftpException
    private void analyzeScriptOutputTest6() {
        test();
    }

    @Test
    public void generateScriptTest() {
        test();
    }

    @Test
    public void runScriptTest() {
        test();
    }

    private String correctFileName = "correct";
    private String incorrectFileName = "incorrect.?$%^&Y";

    @Test
    // Correct file name, provenance is enabled
    public void initProjectTest1() {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        boolean result = wm.initProject(correctFileName, true);
        Assertions.assertTrue(result);
    }

    @Test
    // Correct file name, provenenance is not enabled
    public void initProjectTest2() {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        boolean result = wm.initProject(correctFileName, false);
        Assertions.assertTrue(result);
    }

    @Test
    // Incorrect file name, provenance is enabled
    public void initProjectTest3() {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        boolean result = wm.initProject(incorrectFileName, true);
        Assertions.assertTrue(result);
    }

    @Test
    // Incorrect file name, provenenance is not enabled
    public void initProjectTest4() {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        boolean result = wm.initProject(incorrectFileName, false);
        Assertions.assertTrue(result);
    }

    @Test
    // IOException
    public void initProjectTest5() {

    }

    @Test
    // ParserConfigurationException
    public void initProjectTest6() {

    }

    @Test
    // SAXException
    public void initProjectTest7() {

    }

    @Test
    public void addInputFileTest() {
        test();
    }


    @Test
    public void getProjectNameTest() {
        test();
    }

    @Test
    public void scriptGeneratedTest() {
        test();
    }

    @Test
    public void isSimExecutionRemoteTest() {
        test();
    }

    @Test
    public void isProvEnabledTest() {
        test();
    }

    @Test
    public void getMextScriptNameTest() {
        test();
    }

    @Test
    public void getSimConfigFileOverviewTest() {
        test();
    }

    @Test
    public void getSimulationSpecificationTest() {
        test();
    }

    @Test
    public void getScriptPathTest() {
        test();
    }

    @Test
    public void scriptGenerationAvailableTest() {
        test();
    }

    @Test
    public void scriptRamTest() {
        test();
    }

    @Test
    public void scriptAnalyzedTest() {
        test();
    }

    @Test
    public void getSimulationOverviewTest() {
        test();
    }

    @Test
    public void getScriptRunOverviewTest() {
        test();
    }

    @Test
    public void getScriptAnaylsisOverviewTest() {
        test();
    }

    @Test
    public void getProvMgrTest() {
        test();
    }

    @Test
    public void getMessagesTest() {
        test();
    }

    @Test
    public void configureParamsClassesTest() {
        test();
    }

    private WorkbenchManager blankWorkbenchManagerFactory() {
        return new WorkbenchManager();
    }

    private WorkbenchManager initProjectWorkbenchManagerFactory(boolean provEnabled) {
        WorkbenchManager wm = blankWorkbenchManagerFactory();
        wm.initProject(correctFileName, provEnabled);
        return wm;
    }

}
