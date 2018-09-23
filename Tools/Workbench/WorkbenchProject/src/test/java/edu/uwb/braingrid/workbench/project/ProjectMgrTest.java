package edu.uwb.braingrid.workbench.project;

import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbenchdashboard.nledit.Project;
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

        // Load Prj
        ProjectMgr pmLoad = getPmNameTrueLoadActualProject();

        pmNew.removeScript();

        Assertions.fail("Sob violently. Ended here for today.");
    }

    // <editor-fold defaultstate="collapsed" desc="Getters/Setter Tests">
    @Test
    public void getProjectFileNameTest() {
        // New Prj
        ProjectMgr pmNew = getPmNameFalseLoad();
        try {
            Assertions.assertEquals(pmNew.determineProjectOutputLocation() + pmNew.getName() + ".xml", pmNew.getProjectFilename());
        } catch (IOException e) {

        }

        // Load Prj
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
