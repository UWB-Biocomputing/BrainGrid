package edu.uwb.braingrid.general;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import edu.uwb.braingrid.workbenchdashboard.userModel.User;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.ArrayList;

public class FileSelectorDirMgrTest {

    private FileSelectorDirMgr fileSelectorDirMgrFactory() {
        FileSelectorDirMgr fs = new FileSelectorDirMgr();

        File main = new File("./main");
        fs.add(main);
        File resources = new File("./resources");
        fs.add(resources);
        Assertions.assertEquals(resources, fs.getLastDir());

        return fs;
    }

    @Test
    public void getLastDirTestAndAddTestAndDefault() {

        FileSelectorDirMgr fs = new FileSelectorDirMgr();
        Assertions.assertEquals(fs.getDefault(), fs.getLastDir());

        File main = new File("./main");
        fs.add(main);
        Assertions.assertEquals(main, fs.getLastDir());

        File resources = new File("./resources");
        fs.add(resources);
        Assertions.assertEquals(resources, fs.getLastDir());
    }

    @Test
    public void getDirTest() {
        FileSelectorDirMgr fs = new FileSelectorDirMgr();
        Assertions.assertEquals(fs.getDefault(), fs.getDir(3));

        fs = fileSelectorDirMgrFactory();
        Assertions.assertEquals(fs.getDefault(), fs.getDir(3));
        Assertions.assertEquals(new File("./resources"), fs.getDir(1));
    }
}
