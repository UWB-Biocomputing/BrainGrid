package edu.uwb.braingrid.workbench.ui;

import edu.uwb.braingrid.provenance.ProvMgr;
import edu.uwb.braingrid.workbench.FileManager;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

/**
 * ToDo
 *
 * @author Nathan
 */
public class ProvenanceQueryDialog extends javax.swing.JDialog {
    // <editor-fold defaultstate="collapsed" desc="Auto-Generated Code">

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        predicateComboBox = new javax.swing.JComboBox();
        jSeparator1 = new javax.swing.JSeparator();
        jScrollPane1 = new javax.swing.JScrollPane();
        outputTextArea = new javax.swing.JTextArea();
        searchButton = new javax.swing.JButton();
        subjectTextField = new javax.swing.JTextField();
        objectTextField = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Provenance Query");

        predicateComboBox.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "None" }));
        predicateComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                predicateComboBoxActionPerformed(evt);
            }
        });

        outputTextArea.setEditable(false);
        outputTextArea.setColumns(20);
        outputTextArea.setRows(5);
        jScrollPane1.setViewportView(outputTextArea);

        searchButton.setText("Search");
        searchButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                searchButtonActionPerformed(evt);
            }
        });

        subjectTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyReleased(java.awt.event.KeyEvent evt) {
                subjectTextFieldKeyReleased(evt);
            }
        });

        objectTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyReleased(java.awt.event.KeyEvent evt) {
                objectTextFieldKeyReleased(evt);
            }
        });

        jLabel1.setText("Subject:");

        jLabel2.setText("Predicate:");

        jLabel3.setText("Object:");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jSeparator1)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLabel1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(subjectTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jLabel2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(predicateComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, 236, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jLabel3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(objectTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(searchButton)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(9, 9, 9)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(predicateComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(searchButton)
                    .addComponent(subjectTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(objectTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1)
                    .addComponent(jLabel2)
                    .addComponent(jLabel3))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 2, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 131, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void searchButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_searchButtonActionPerformed
        String sbjct = subjectTextField.getText();
        String prdct = predicateFullURIs.get(
                predicateComboBox.getSelectedIndex());
        String objct = objectTextField.getText();
        String result
                = provMgr.queryProvenance(sbjct, prdct, objct, lineDelimiter);
        System.err.println("  Subject: |" + sbjct + "|");
        System.err.println("Predicate: |" + prdct + "|");
        System.err.println("   Object: |" + objct + "|");
        outputTextArea.setText(result);
    }//GEN-LAST:event_searchButtonActionPerformed

    private void subjectTextFieldKeyReleased(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_subjectTextFieldKeyReleased
        enableSearchButton();
    }//GEN-LAST:event_subjectTextFieldKeyReleased

    private void objectTextFieldKeyReleased(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_objectTextFieldKeyReleased
        enableSearchButton();
    }//GEN-LAST:event_objectTextFieldKeyReleased

    private void predicateComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_predicateComboBoxActionPerformed
        enableSearchButton();
    }//GEN-LAST:event_predicateComboBoxActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JTextField objectTextField;
    private javax.swing.JTextArea outputTextArea;
    private javax.swing.JComboBox predicateComboBox;
    private javax.swing.JButton searchButton;
    private javax.swing.JTextField subjectTextField;
    // End of variables declaration//GEN-END:variables
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Custom Members">
    private final ProvMgr provMgr;
    private final String lineDelimiter = "\n";
    List<String> subjectFullURIs = new ArrayList<>();
    List<String> predicateFullURIs = new ArrayList<>();
    List<String> objectFullURIs = new ArrayList<>();
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Construction"> 
    /**
     * Constructs and initializes this provenance query dialog
     *
     * @param modal - True if the parent should be disabled while this dialog is
     * open, otherwise false
     * @param provManager - The provenance manager for the workbench
     */
    public ProvenanceQueryDialog(boolean modal, ProvMgr provManager) {
        provMgr = provManager;
        setModal(modal);
        initComponents();
        //searchButton.setEnabled(false);
        // add in the drop down items
        addItemsToPredicateComboBox();
        // show window center-screen
        pack();
        center();
        setVisible(true);
    }

    private void addItemsToPredicateComboBox() {
        predicateFullURIs.add("");
        for(String predicate : provMgr.getPredicates()){
            predicateFullURIs.add(FileManager.getSimpleFilename(predicate));
            predicateComboBox.addItem(predicate);
        }
    }

    /**
     * Centers this dialog
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

    // <editor-fold defaultstate="collapsed" desc="Utility">
    private void enableSearchButton() {
        searchButton.setEnabled(isSubjectValid() || isPredicateValid()
                || isObjectValid());
    }

    private boolean isSubjectValid() {
        if (subjectTextField.getText() != null && !subjectTextField.getText().equals("")) {
            return true;
        } else {
            return false;
        }
    }

    private boolean isPredicateValid() {
        if (!((String) predicateComboBox.getSelectedItem()).equals("None")) {
            return true;
        } else {
            return false;
        }
    }

    private boolean isObjectValid() {
        if (!objectTextField.getText().equals("")) {
            return true;
        } else {
            return false;
        }
    }
    // </editor-fold>
}
