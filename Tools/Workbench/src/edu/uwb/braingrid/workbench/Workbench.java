package edu.uwb.braingrid.workbench;

import edu.uwb.braingrid.workbench.ui.WorkbenchControlFrame;

/**
 * <h2>Project Manager for the Brain Grid Toolbox.<h2>
 *
 * <p>
 * The workbench is usable from a command line interface through runCLI or
 * through a graphical user interface through runGUI. These two types of
 * invocation are determined based on the amount of command line arguments
 * specified at execution</p>
 *
 * <p>
 * Specifying zero arguments (double clicking the java archive
 * BrainGridWorkbench.jar file) will launch the graphical workbench</p>
 *
 * <p>
 * Specifying the input file names, simulator type, and output file name as
 * arguments will launch the application in simple-mode. Simple-mode results in
 * writing RDF provenance represented as a turtle file. The resulting turtle
 * file is named after the output file, but with the .ttl extension</p>
 *
 * @author Del Davis
 * @version 0.1
 */
public class Workbench {

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info
                    : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(WorkbenchControlFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(WorkbenchControlFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(WorkbenchControlFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(WorkbenchControlFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new WorkbenchControlFrame().setVisible(true);
            }
        });
    }
}
