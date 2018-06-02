package edu.uwb.braingrid.workbenchdashboard.setup;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.Date;

import javax.swing.JFileChooser;

import edu.uwb.braingrid.workbench.WorkbenchManager;
import edu.uwb.braingrid.workbench.utils.DateTime;
import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.embed.swing.SwingNode;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

public class Selector {

    public Selector(WorkbenchManager workbench_Mgr) {
    	workbenchMgr = workbench_Mgr;
    }
    /**
     * Saves the current project to XML.
     *
     * <i>Assumption: This action is unreachable prior to specifying a new
     * project or loading a project from disk</i>
     *
     * @param evt - The event that triggered this action
     */
    void saveProject() {
       
    }
    
    /**
     * Allows the user to open a previously defined BrainGrid project from an
     * XML file. The information from the project is queried to update the UI
     * and load the provenance model.
     *
     * @param evt - The event that triggered this action
     */
    void openProject() {//GEN-FIRST:event_openProjectMenuItemActionPerformed
        
    }//GEN-LAST:event_openProjectMenuItemActionPerformed






    // End of variables declaration//GEN-END:variables
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Custom Members">
    private WorkbenchManager workbenchMgr;
    // </editor-fold>

    private void initCustomMembers() {
        workbenchMgr = new WorkbenchManager();

        transferProgressBar.setVisible(false);
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="UI Manipulation">
    /**
     * Centers the frame in the operating system's GUI window
     */
    private void center() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension frameSize = getSize();
        if (frameSize.height > screenSize.height) {
            frameSize.height = screenSize.height;
        }
        if (frameSize.width > screenSize.width) {
            frameSize.width = screenSize.width;
        }
        setLocation((screenSize.width - frameSize.width) / 2,
                (screenSize.height - frameSize.height) / 2);
    }

//    @Override
//    /**
//     * Fits the window to the maximum width and height of all the contained
//     * components. The minimum size is reset to the current size after the pack
//     * to make sure that the window can not get any smaller. As in the parent
//     * component's implementation of pack, the window will always be within the
//     * bounds of it's maximum size.
//     */
//    public void pack() {
//        super.pack();
//        setMinimumSize(getSize());
//    }



    /**
     * Disables all of the buttons and menu items related to project attribute
     * specification. Since the default behavior is to leave the previous
     * project open, this should only be called when an error is encountered
     * resulting in a null value for the open project.
     */
    private void disableProjectAttributeRelatedButtons() {
        configureSimulationButton.setEnabled(false);
        specifyScriptButton.setEnabled(false);
        runScriptButton.setEnabled(false);
        scriptGenerateButton.setEnabled(false);
        analyzeOutputButton.setEnabled(false);
        saveProjectMenuItem.setEnabled(false);
    }








//    private void setMacCopyPaste() {
//        if (FileManager.getFileManager().isMacSystem()) {
//            InputMap im = (InputMap) UIManager.get("TextField.focusInputMap");
//            im.put(KeyStroke.getKeyStroke(KeyEvent.VK_C,
//                    KeyEvent.META_DOWN_MASK), DefaultEditorKit.copyAction);
//            im.put(KeyStroke.getKeyStroke(KeyEvent.VK_V,
//                    KeyEvent.META_DOWN_MASK), DefaultEditorKit.pasteAction);
//            im.put(KeyStroke.getKeyStroke(KeyEvent.VK_X,
//                    KeyEvent.META_DOWN_MASK), DefaultEditorKit.cutAction);
//        }
//    }
    //</editor-fold>

    // <editor-fold defaultstate="collapsed" desc="User Communication">
    // </editor-fold>
}
