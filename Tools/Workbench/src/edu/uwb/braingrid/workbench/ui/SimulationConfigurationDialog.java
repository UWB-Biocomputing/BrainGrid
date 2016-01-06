package edu.uwb.braingrid.workbench.ui;

import edu.uwb.braingrid.simconfig.model.ConfigDatum;
import edu.uwb.braingrid.simconfig.model.SimulationConfiguration;
import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.data.SimulationConfigurationManager;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import org.xml.sax.SAXException;

/**
 *
 * @author Aaron Conrad and Del Davis
 */
public class SimulationConfigurationDialog extends javax.swing.JDialog {

    // <editor-fold defaultstate="collapsed" desc="Custom Members"> 
    private SimulationConfigurationManager scm;
    private boolean okClicked = false;
    private String lastBuiltFile = null;
    private String lastStateOutputFileName = null;
    private String projectName = null;

    private JTabbedPane tabs;
    private List<JTextField> fields;
    private JButton cancelButton;
    private JButton okButton;
    private JButton buildButton;
    private JTextField configFilename_textField;
    private JLabel configFilename_label;
    private JSeparator jSeparator1;
    private JLabel messageLabel;
    private JLabel messageLabelText;
    // </editor-fold>

    /**
     * Constructor creates a new dialog box
     *
     * @param projectName
     * @param modal
     * @param configFilename
     * @param simConfig
     */
    public SimulationConfigurationDialog(String projectName, boolean modal,
            String configFilename, SimulationConfiguration simConfig) {
        initComponents(simConfig);
        setModal(modal);
        this.projectName = projectName;

        try {
            scm = new SimulationConfigurationManager(configFilename);
        } catch (Exception e) {
            System.err.println(e.toString());
        }
        if (scm != null) {
            okButton.setEnabled(false);

            // show window center-screen
            pack();
            center();
            //setSize(new Dimension(600, 600));
            //setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
            setVisible(true);
        }
    }

    // TO DO - CHECK IF THE FIRST IF STATEMENT CAN BE IMPROVED
    private void initComponents(SimulationConfiguration simConfig) {
        if (simConfig == null) {
            return;
        }
        fields = new ArrayList<>();
        tabs = new JTabbedPane();
        String nextTabName = "";
        JPanel contentPanel = null;
        JPanel subPanel;
        BoxLayout contentLayout;
        GroupLayout subLayout;
        JTextField field;
        JLabel label;
        String labelText;
        JButton importButton;
        String importButtonText = "Import";
        JScrollPane scrollPane;

        ActionListener al = new ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                importButtonActionPerformed(evt);
            }
        };

        cancelButton = new javax.swing.JButton();
        okButton = new javax.swing.JButton();
        buildButton = new javax.swing.JButton();
        configFilename_textField = new javax.swing.JTextField();
        configFilename_label = new javax.swing.JLabel();
        jSeparator1 = new javax.swing.JSeparator();
        messageLabel = new javax.swing.JLabel();
        messageLabelText = new javax.swing.JLabel();

        int datumType;
        int index = 0;
        ConfigDatum datum = simConfig.getDatum(index);

        //For each datum that the simConfig contains, create its relevant component
        while (datum != null) {
            datumType = datum.getDatumType();
            labelText = datum.getLabel();
            //If there is no label, get the tag name instead
            if (labelText == null) {
                labelText = datum.getName();
            }

            //If a tab type, create new tab
            if (datumType == ConfigDatum.TAB_TYPE) {
                nextTabName = labelText;
                contentPanel = new JPanel();
                contentLayout = new BoxLayout(contentPanel, BoxLayout.PAGE_AXIS);
                contentPanel.setLayout(contentLayout);
            } //If subhead type, create new subhead label
            else if (datumType == ConfigDatum.SUBHEAD_TYPE) {
                subPanel = new JPanel();
                subLayout = new GroupLayout(subPanel);
                subPanel.setLayout(subLayout);
                subLayout.setAutoCreateGaps(true);
                label = new JLabel(labelText);

                subLayout.setHorizontalGroup(
                        subLayout.createSequentialGroup()
                        .addComponent(label)
                );
                subLayout.setVerticalGroup(
                        subLayout.createSequentialGroup()
                        .addGroup(subLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                .addComponent(label))
                );
                if (contentPanel == null) {
                    // TO DO - FIGURE OUT WHAT TO DO HERE
                } else {
                    contentPanel.add(subPanel);
                }
            } //If param type, create new label and text field
            else if (datumType == ConfigDatum.PARAM_TYPE) {
                subPanel = new JPanel();
                subLayout = new GroupLayout(subPanel);
                subPanel.setLayout(subLayout);
                subLayout.setAutoCreateGaps(true);
                label = new JLabel(labelText);
                //TO DO - FIND OUT WHAT HAPPENS IF THE defaultContent IS NULL OR EMPTY
                field = new JTextField(datum.getContent());
                field.setMaximumSize(new Dimension(Integer.MAX_VALUE, field.getPreferredSize().height));

                if (datum.isFileChooser()) {
                    importButton = new JButton(importButtonText);
                    importButton.setActionCommand(fields.size() + "");
                    importButton.addActionListener(al);

                    subLayout.setHorizontalGroup(
                            subLayout.createSequentialGroup()
                            .addComponent(label)
                            .addComponent(field)
                            .addComponent(importButton)
                    );
                    subLayout.setVerticalGroup(
                            subLayout.createSequentialGroup()
                            .addGroup(subLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                    .addComponent(label))
                            .addComponent(field)
                            .addComponent(importButton)
                    );
                } else {
                    subLayout.setHorizontalGroup(
                            subLayout.createSequentialGroup()
                            .addComponent(label)
                            .addComponent(field)
                    );
                    subLayout.setVerticalGroup(
                            subLayout.createSequentialGroup()
                            .addGroup(subLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                    .addComponent(label))
                            .addComponent(field)
                    );
                }

                fields.add(field);
                if (contentPanel == null) {
                    // TO DO - FIGURE OUT WHAT TO DO HERE
                } else {
                    contentPanel.add(subPanel);
                }
            } //If subhead_end type, add a seperation
            else if (datumType == ConfigDatum.SUBHEAD_END) {
                subPanel = new JPanel();
                subLayout = new GroupLayout(subPanel);
                subPanel.setLayout(subLayout);
                label = new JLabel("END_SUBHEAD");
                subLayout.setHorizontalGroup(
                        subLayout.createSequentialGroup()
                        .addComponent(label)
                );
                subLayout.setVerticalGroup(
                        subLayout.createSequentialGroup()
                        .addGroup(subLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                .addComponent(label))
                );
                contentPanel.add(subPanel);
                //contentPanel.add(new JSeparator(SwingConstants.HORIZONTAL));
            } //If tab_end type, finish the tab and add to the dialog
            else if (datumType == ConfigDatum.TAB_END) {
                scrollPane = new JScrollPane(contentPanel);
                tabs.add(nextTabName, scrollPane);
                nextTabName = "";
            }
            //If null type, do nothing

            //Get next datum
            index++;
            datum = simConfig.getDatum(index);
        }
        add(tabs);
        tabs.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));

        //Additional panel at bottom that contains the OK, Cancel, and Build buttons
        cancelButton.setText("Cancel");
        cancelButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cancelButtonActionPerformed(evt);
            }
        });

        okButton.setText("OK");
        okButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                okButtonActionPerformed(evt);
            }
        });

        buildButton.setText("Build");
        buildButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buildButtonActionPerformed(evt);
            }
        });

        configFilename_textField.setEnabled(false);

        configFilename_label.setText("Config Filename:");

        messageLabel.setText("Message:");

        messageLabelText.setText("None");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addComponent(jSeparator1)
                .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                        .addComponent(buildButton)
                                        .addGap(18, 18, 18)
                                        .addComponent(configFilename_label)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(configFilename_textField)
                                        .addGap(18, 18, 18)
                                        .addComponent(okButton)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(cancelButton))
                                .addGroup(layout.createSequentialGroup()
                                        .addComponent(messageLabel)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(messageLabelText)
                                        .addGap(0, 0, Short.MAX_VALUE)))
                        .addContainerGap())
                .addComponent(tabs)
        );
        layout.setVerticalGroup(
                layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addComponent(tabs, javax.swing.GroupLayout.PREFERRED_SIZE, 261, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                .addComponent(messageLabel)
                                .addComponent(messageLabelText))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                .addComponent(cancelButton)
                                .addComponent(okButton)
                                .addComponent(buildButton)
                                .addComponent(configFilename_textField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(configFilename_label))
                        .addContainerGap())
        );
    }

    // <editor-fold defaultstate="collapsed" desc="UI Manipulation">
    /**
     * Forces the dialog box to appear in the center of the screen
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
    // </editor-fold>

    private void buildButtonActionPerformed(java.awt.event.ActionEvent evt) {
        /*try {
         //Purge stored values
         //Store new values

         try {
         String fileName = configFilename_textField.getText();
         if (fileName != null && !fileName.isEmpty()) {
         fileName = icm.buildAndPersist(projectName, fileName);
         if (fileName != null) {
         okButton.setEnabled(true);
         lastBuiltFile = fileName;
         lastStateOutputFileName = stateOutputFilename_textField.getText();
         messageLabelText.setText("<html><span style=\"color:green\">"
         + FileManager.getSimpleFilename(fileName)
         + " successfully persisted..."
         + "</span></html>");
         } else {
         messageLabelText.setText("<html><span style=\"color:red\">*All fields must be filled</span></html>");
         }
         }
         } catch (TransformerException | IOException e) {
         messageLabelText.setText("<html><span style=\"color:red\">"
         + e.getClass()
         + " prevented successful build...</span></html>");
         e.printStackTrace();
         }
         } catch (ParserConfigurationException ex) {
         messageLabelText.setText("<html><span style=\"color:red\">"
         + ex.getClass()
         + " prevented successful build...</span></html>");
         ex.printStackTrace();
         }*/
    }

    //TO DO - FINISH THIS
    private void okButtonActionPerformed(java.awt.event.ActionEvent evt) {
        okClicked = true;
        setVisible(false);
    }

    private void cancelButtonActionPerformed(java.awt.event.ActionEvent evt) {
        setVisible(false);
    }

    private void importButtonActionPerformed(java.awt.event.ActionEvent evt) {
        int fieldIndex = Integer.parseInt(((javax.swing.JButton) (evt.getSource())).getActionCommand());
        importFile(fieldIndex);
    }

    private void importFile(int fieldIndex) {
        FileManager fm = FileManager.getFileManager();
        // get starting folder
        String simConfFilesDir;
        try {
            simConfFilesDir = fm.getSimConfigDirectoryPath(projectName, true);
        } catch (IOException e) {
            messageLabelText.setText(
                    "<html><span style=\"color:red\">"
                    + e.getClass()
                    + "occurred, import failed...</span></html>");
            return;
        }
        JFileChooser dlg = new JFileChooser(simConfFilesDir);

        //Restricts file type to XML
        //TO DO - Is this what Stiber wants?
        /*FileNameExtensionFilter filter = new FileNameExtensionFilter(
         "XML file (*.xml)", "xml");
         dlg.addChoosableFileFilter(filter);
         dlg.setFileFilter(filter);
         dlg.setMultiSelectionEnabled(true);*/
        String dialogTitle = "Select Input Files for a Simulation";
        dlg.setDialogTitle(dialogTitle);
        if (dlg.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            try {
                File file = dlg.getSelectedFile();

                Path sourceFilePath = file.toPath();
                String destPathText = fm.getNeuronListFilePath(projectName,
                        file.getName(), true);
                Path destFilePath = new File(destPathText).toPath();
                if (FileManager.copyFile(sourceFilePath, destFilePath)) {
                    fields.get(fieldIndex).setText("workbenchconfigfiles/NList/"
                            + fm.getSimpleFilename(destFilePath.toString()));
                }
                //TO DO - Check whether I need the if statement anymore. Note
                //that code directly above is copied from below with small modifications.
                //Make sure the setText is the correct text.

                // if type is correct
                /*if (InputAnalyzer.getInputType(file) == type) {
                 Path sourceFilePath = file.toPath();
                 String destPathText = fm.getNeuronListFilePath(projectName,
                 file.getName(), true);
                 Path destFilePath = new File(destPathText).toPath();
                 if (FileManager.copyFile(sourceFilePath, destFilePath)) {
                 field.setText("workbenchconfigfiles/NList/"
                 + fm.getSimpleFilename(destFilePath.toString()));
                 }
                 } else {
                 messageLabelText.setText("<html><span style=\"color:orange\">"
                 + "The selected file did not match the type: "
                 + type.toString() + "</span></html>");
                 }*/
            } catch (IOException ex) {
                messageLabelText.setText(
                        "<html><span style=\"color:red\">"
                        + ex.getClass()
                        + "occurred, import failed...</span></html>");
            }
        } else {
            messageLabelText.setText(
                    "<html><span style=\"color:red\">"
                    + "Import Cancelled...</span></html>");
        }
    }

    public static void main(String[] args) {
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
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(SimulationConfigurationDialog.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        // FILE CAN BE FOUND IN BrainGrid/Tools/Workbench/models
        String filename = "C:\\Users\\Del\\Desktop\\SimulationConfigurationTest.xml";
        try {
            SimulationConfigurationDialog dialog = new SimulationConfigurationDialog("Test", true, filename, new SimulationConfiguration(DocumentBuilderFactory.newInstance().
                    newDocumentBuilder().parse(new File(filename))));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
