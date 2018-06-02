package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

/**
 * The ImportPanel class handles import xml neurons list files dialog window.
 * The window contains four input fields and Browse buttons, each of which
 * corresponds width four different kinds of files, configuration, inhibitory
 * neuron list, active neuron list, and probed neuron list files. When a
 * configuration file is specified, the content of the file is parsed, and
 * extract names of neurons list files, and show them in the corresponding
 * fields.
 *
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
@SuppressWarnings("serial")
public class ImportPanel extends JPanel implements ActionListener {

    static final int nFields = 4; // number of input fields
    static final int idxConfigFile = 0; // field index of configuration file

    /**
     * field index of inhibitory neurons list file
     */
    public static final int idxInhList = 1;
    /**
     * field index of active neurons list file
     */
    public static final int idxActList = 2;
    /**
     * field index of probed neurons list file
     */
    public static final int idxPrbList = 3;

    private JLabel[] labels = new JLabel[nFields];
    public JTextField[] tfields = new JTextField[nFields];
    private JButton[] btns = new JButton[nFields];

    private static String configDir = "."; // directory for configuration file
    public static String nlistDir = "."; // directory for neurons list file

    /**
     * A class constructor, which creates UI components, and registers action
     * listener.
     */
    public ImportPanel() {
        labels[idxConfigFile] = new JLabel("Configuration file:");
        labels[idxInhList] = new JLabel("Inhibitory neurons list:");
        labels[idxActList] = new JLabel("Active neurons list:");
        labels[idxPrbList] = new JLabel("Probed neurons list:");

        for (int i = 0; i < nFields; i++) {
            tfields[i] = new JTextField(20);
            tfields[i].setEditable(true);
            btns[i] = new JButton("Browse...");
            btns[i].addActionListener(this);
        }

        GroupLayout layout = new GroupLayout(this);
        setLayout(layout);

        // create vertical sequential group
        GroupLayout.SequentialGroup vgroup = layout.createSequentialGroup();
        for (int i = 0; i < nFields; i++) {
            GroupLayout.ParallelGroup group2 = layout
                    .createParallelGroup(Alignment.BASELINE);
            group2.addComponent(labels[i]);
            group2.addComponent(tfields[i]);
            group2.addComponent(btns[i]);
            vgroup.addGroup(group2);
        }

        // create horizontal sequential group
        GroupLayout.SequentialGroup hgroup = layout.createSequentialGroup();
        GroupLayout.ParallelGroup groupLabel = layout.createParallelGroup();
        GroupLayout.ParallelGroup groupTField = layout.createParallelGroup();
        GroupLayout.ParallelGroup groupBtn = layout.createParallelGroup();
        for (int i = 0; i < nFields; i++) {
            groupLabel.addComponent(labels[i]);
            groupTField.addComponent(tfields[i]);
            groupBtn.addComponent(btns[i]);
        }
        hgroup.addGroup(groupLabel);
        hgroup.addGroup(groupTField);
        hgroup.addGroup(groupBtn);

        // set horizontal and vertical group
        layout.setVerticalGroup(vgroup);
        layout.setHorizontalGroup(hgroup);
    }

    /*
     * (non-Javadoc)
     * 
     * @see
     * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
     */
    public void actionPerformed(ActionEvent e) {
        int iSource = 0;
        for (int i = 0; i < nFields; i++) {
            if (e.getSource() == btns[i]) {
                iSource = i;
                break;
            }
        }
        // create a file chooser
        String curDir;
        if (iSource == idxConfigFile) {
            curDir = configDir;
        } else {
            curDir = nlistDir;
        }

        JFileChooser chooser = new JFileChooser(curDir);
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "XML file (*.xml)", "xml");
        chooser.addChoosableFileFilter(filter);
        chooser.setMultiSelectionEnabled(false);
        String dialogTitle = "";
        switch (iSource) {
            case idxConfigFile:
                dialogTitle = "Configuration file";
                break;
            case idxInhList:
                dialogTitle = "Inhibitory neurons list";
                break;
            case idxActList:
                dialogTitle = "Active neurons list";
                break;
            case idxPrbList:
                dialogTitle = "Probed neurons list";
                break;
        }
        chooser.setDialogTitle(dialogTitle);
        int option = chooser.showOpenDialog(this);
        if (option == JFileChooser.APPROVE_OPTION) {
            tfields[iSource].setText(chooser.getSelectedFile()
                    .getAbsolutePath());
            if (iSource == idxConfigFile) { // configuration files is specified.
                // parse config file, extract names of neurons list files, and
                // show them in the corresponding fields
                String configFile = chooser.getSelectedFile().getAbsolutePath();
                configDir = chooser.getSelectedFile().getParent();
                nlistDir = configDir;
                try {
                    Document doc = new SAXBuilder().build(new File(configFile));
                    Element root = doc.getRootElement();
                    Element layout = root.getChild("FixedLayout").getChild(
                            "LayoutFiles");
                    org.jdom2.Attribute attr;
                    if ((attr = layout.getAttribute("activeNListFileName"))
                            != null) {
                        tfields[idxActList].setText(configDir + "/"
                                + attr.getValue());
                    }
                    if ((attr = layout.getAttribute("inhNListFileName")) != null) {
                        tfields[idxInhList].setText(configDir + "/"
                                + attr.getValue());
                    }
                    if ((attr = layout.getAttribute("probedNListFileName"))
                            != null) {
                        tfields[idxPrbList].setText(configDir + "/"
                                + attr.getValue());
                    }
                } catch (JDOMException je) {
                    // System.err.println(je);
                } catch (IOException ie) {
                    // System.err.println(ie);
                }
            } else {
                nlistDir = chooser.getSelectedFile().getParent();
            }
        }
    }
}
