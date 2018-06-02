package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.GridLayout;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

/*
 * The GPatternPanel class handles generate pattern dialog window.
 * The dialog window contains two radio buttons to choose distribution patern,
 * random or regular, and two input fields to enter ratio of inhibitory, 
 * and active neurons.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
@SuppressWarnings({ "unused", "serial" })
public class GPatternPanel extends JPanel {
	public JRadioButton[] btns = new JRadioButton[2];
	private JLabel[] labels = new JLabel[2];
	public JTextField[] tfields = new JTextField[2];

	public static final int idxREG = 0;
	public static final int idxRND = 1;
	public static final int idxINH = 0;
	public static final int idxACT = 1;

	public GPatternPanel() {
		btns[idxREG] = new JRadioButton("Regular pattern");
		btns[idxRND] = new JRadioButton("Random pattern");

		tfields[idxINH] = new JTextField();
		labels[idxINH] = new JLabel("<html>Inhibitory<br>neurons ratio:</html>");
		tfields[idxACT] = new JTextField();
		labels[idxACT] = new JLabel("<html>Active<br>neurons ratio:</html>");

		JPanel cboxPanel = new JPanel();
		cboxPanel.setLayout(new GridLayout(2, 1));
		cboxPanel.add(btns[idxREG]);
		cboxPanel.add(btns[idxRND]);
		ButtonGroup bgroup = new ButtonGroup();
		bgroup.add(btns[idxREG]);
		bgroup.add(btns[idxRND]);
		btns[idxREG].setSelected(true);
		;

		JPanel ratioPanel = new JPanel();
		ratioPanel.setLayout(new GridLayout(2, 2));
		ratioPanel.add(labels[idxREG]);
		ratioPanel.add(tfields[idxREG]);
		ratioPanel.add(labels[idxRND]);
		ratioPanel.add(tfields[idxRND]);
		ratioPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory
				.createEtchedBorder()));

		setLayout(new GridLayout(1, 2));
		add(cboxPanel);
		add(ratioPanel);
	}
}
