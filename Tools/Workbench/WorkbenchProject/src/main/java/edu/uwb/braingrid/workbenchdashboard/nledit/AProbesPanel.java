package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.GridLayout;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

/**
 * The AProbesPanel class handles arrange probed neurons dialog window. The
 * dialog window contains one input field to enter number of probes.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
@SuppressWarnings("serial")
public class AProbesPanel extends JPanel {
	private JLabel label = new JLabel("<html>Number of probes:</html>");;
	public JTextField tfield = new JTextField();;

	public AProbesPanel() {
		setLayout(new GridLayout(2, 1));
		add(label);
		add(tfield);
	}
}
