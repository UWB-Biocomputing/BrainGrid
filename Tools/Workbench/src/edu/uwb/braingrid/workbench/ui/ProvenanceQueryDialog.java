package edu.uwb.braingrid.workbench.ui;

import edu.uwb.braingrid.provenance.ProvMgr;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.List;
import javax.swing.JComboBox;

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

        subjectComboBox = new javax.swing.JComboBox();
        predicateComboBox = new javax.swing.JComboBox();
        objectComboBox = new javax.swing.JComboBox();
        jSeparator1 = new javax.swing.JSeparator();
        jScrollPane1 = new javax.swing.JScrollPane();
        outputTextArea = new javax.swing.JTextArea();
        searchButton = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Provenance Query");

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
                        .addComponent(subjectComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(predicateComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(objectComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(searchButton)
                        .addGap(0, 8, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(9, 9, 9)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(subjectComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(objectComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(predicateComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(searchButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 2, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jScrollPane1)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void searchButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_searchButtonActionPerformed
        String subject = (String) subjectComboBox.getSelectedItem();
        String prdicate = (String) predicateComboBox.getSelectedItem();
        String object = (String) objectComboBox.getSelectedItem();
        String result = provMgr.queryProvenance(subject, prdicate, object);
        outputTextArea.setText(result);
    }//GEN-LAST:event_searchButtonActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JComboBox objectComboBox;
    private javax.swing.JTextArea outputTextArea;
    private javax.swing.JComboBox predicateComboBox;
    private javax.swing.JButton searchButton;
    private javax.swing.JComboBox subjectComboBox;
    // End of variables declaration//GEN-END:variables

    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Custom Members">
    private ProvMgr provMgr;
    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Construction"> 
    /**
     * Constructs and initializes this provenance query dialog
     * @param modal - True if the parent should be disabled while this dialog is
     * open, otherwise false
     */
    public ProvenanceQueryDialog(boolean modal, ProvMgr newProvMgr) {
        provMgr = newProvMgr;
        
        setModal(modal);
        initComponents();
        
        // add in the drop down items
        addItemsToComboBox(subjectComboBox, provMgr.getSubjects());
        addItemsToComboBox(predicateComboBox, provMgr.getPredicates());
        addItemsToComboBox(objectComboBox, provMgr.getObjects());
        
        // show window center-screen
        pack();
        center();
        setVisible(true);
    }
    
    private void addItemsToComboBox(JComboBox comboBox, List<String> theList) {
        for (String item : theList)
            comboBox.addItem(item);
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
}
