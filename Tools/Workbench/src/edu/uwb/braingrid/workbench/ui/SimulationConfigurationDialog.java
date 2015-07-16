package edu.uwb.braingrid.workbench.ui;

import edu.uwb.braingrid.simconfig.model.ConfigDatum;
import edu.uwb.braingrid.simconfig.model.SimulationConfiguration;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;
//extends javax.swing.JDialog
/**
 *
 * @author Aaron
 */
public class SimulationConfigurationDialog {
    
    private JFrame window;  ///////////////////////////////////////////////////////
    private String projectName;
    private JTabbedPane tabs;
    private List<JTextField> fields;
    
    public SimulationConfigurationDialog(String projectName, boolean modal,
            String configFilename, SimulationConfiguration simConfig) {
        initComponents(simConfig);
        //setModal(modal);
        this.projectName = projectName;
        
        // show window center-screen
        //pack();
        //center();
        //setSize(new Dimension(400, 800));
        //setVisible(true);
    }
    
    // TO DO - CHECK IF THE FIRST IF STATEMENT CAN BE IMPROVED
    private void initComponents(SimulationConfiguration simConfig) {
        if (simConfig == null) {
            return;
        }
        window = new JFrame("TESTING"); /////////////////////////////////////////////
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
        JScrollPane scrollPane;
        
        int datumType;
        int index = 0;
        ConfigDatum datum = simConfig.getDatum(index);
        
        while (datum != null) {
            datumType = datum.getDatumType();
            labelText = datum.getLabel();
            if (labelText == null) {
                labelText = datum.getName();
            }
            
            if (datumType == ConfigDatum.TAB_TYPE) {
               nextTabName = labelText;
               contentPanel = new JPanel();
               contentLayout = new BoxLayout(contentPanel, BoxLayout.PAGE_AXIS);
               contentPanel.setLayout(contentLayout);
            }
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
                }
                else {
                    contentPanel.add(subPanel);
                }
            }
            else if (datumType == ConfigDatum.PARAM_TYPE) {
                subPanel = new JPanel();
                subLayout = new GroupLayout(subPanel);
                subPanel.setLayout(subLayout);
                subLayout.setAutoCreateGaps(true);
                label = new JLabel(labelText);
                //TO DO - FIND OUT WHAT HAPPENS IF THE defaultContent IS NULL OR EMPTY
                field = new JTextField(datum.getContent());
                field.setMaximumSize(new Dimension(Integer.MAX_VALUE, field.getPreferredSize().height));
                
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
                
                fields.add(field);
                if (contentPanel == null) {
                    // TO DO - FIGURE OUT WHAT TO DO HERE
                }
                else {
                    contentPanel.add(subPanel);
                }
            }
            else if (datumType == ConfigDatum.SUBHEAD_END) {
                //Add blank space
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
            }
            else if (datumType == ConfigDatum.TAB_END) {
                scrollPane = new JScrollPane(contentPanel);
                tabs.add(nextTabName, scrollPane);
                nextTabName = "";
            }
            
            index++;
            datum = simConfig.getDatum(index);
        }
        //add(tabs);
        window.add(tabs);  ////////////////////////////////////////////////////////
        window.setSize(400, 400);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
    }
    
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
    
    public static void main(String[] args) {
        String filename = "C:\\Users\\Aaron\\Desktop\\SimulationConfigurationTest.xml";
        try {
            SimulationConfigurationDialog dialog = new SimulationConfigurationDialog("Test", true, "Test.xml", new SimulationConfiguration(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
