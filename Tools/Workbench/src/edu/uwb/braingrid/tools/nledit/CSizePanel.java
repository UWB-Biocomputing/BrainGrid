package edu.uwb.braingrid.tools.nledit;

import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

/*
 * The CSizePanel class handles modify layout size dialog window.
 * The dialog window contains two fields to enter size of the new layout,
 * and three radio buttons to choose method to fill pattern, new, repeat, or alternate.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
@SuppressWarnings("serial")
public class CSizePanel extends JPanel {
	private JLabel[] labels = new JLabel[2];
	public JTextField[] tfields = new JTextField[2];

	public static final int idxX = 0;
	public static final int idxY = 1;

	public JRadioButton newButton = new JRadioButton("New", true);
	public JRadioButton rptButton = new JRadioButton("Repeat", false);
	public JRadioButton altButton = new JRadioButton("Alternate", false);

	public CSizePanel(Dimension size, boolean init) {
		labels[0] = new JLabel("Size x:");
		labels[1] = new JLabel("Size y:");

		for (int i = 0; i < 2; i++) {
			tfields[i] = new JTextField();
		}
		tfields[0].setText(Integer.toString(size.width));
		tfields[1].setText(Integer.toString(size.height));

		ButtonGroup bgroup = new ButtonGroup();
		bgroup.add(newButton);
		bgroup.add(rptButton);
		bgroup.add(altButton);
		if (init) {
			rptButton.setEnabled(false);
			altButton.setEnabled(false);
		}

		JPanel radioPanel = new JPanel();
		radioPanel.setLayout(new GridLayout(3, 1));
		radioPanel.add(newButton);
		radioPanel.add(rptButton);
		radioPanel.add(altButton);
		radioPanel.setBorder(BorderFactory.createTitledBorder(
				BorderFactory.createEtchedBorder(), "Methods"));

		JPanel sizePanel = new JPanel();
		sizePanel.setLayout(new GridLayout(4, 1));
		sizePanel.add(labels[0]);
		sizePanel.add(tfields[0]);
		sizePanel.add(labels[1]);
		sizePanel.add(tfields[1]);

		setLayout(new GridLayout(1, 2));
		add(sizePanel);
		add(radioPanel);
	}
}
