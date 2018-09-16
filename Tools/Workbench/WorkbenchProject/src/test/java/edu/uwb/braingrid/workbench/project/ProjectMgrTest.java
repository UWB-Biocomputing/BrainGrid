package edu.uwb.braingrid.workbench.project;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;

public class ProjectMgrTest {
    @Test
    public void test() {
        Assertions.fail("Need to test still");
    }

    @Test
    // load a project
    public void constructorTest() {


        // Invalid cases
        ProjectMgr pm = getPmNameFalseLoad();
        Assertions.assertNotEquals(null, pm);
        pm = getPmNameTrueLoad();
        Assertions.assertEquals(null, pm);
        pm = getPmNullNameFalseLoad();
        Assertions.assertEquals(null, pm);
        pm = getPmNullNameTrueLoad();
        Assertions.assertEquals(null, pm);
    }


    // name, true, invalid project
    private ProjectMgr getPmNameTrueLoad() {
        ProjectMgr pm = null;
        try {
            pm = new ProjectMgr("LoadProject", true);
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

}
