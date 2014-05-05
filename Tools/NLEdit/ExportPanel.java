import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.GroupLayout.Alignment;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 * The ExportPanel class handles export xml neurons list files dialog window.
 * The window contains three input fields and Browse buttons, each of which
 * corresponds width four different kinds of files, inhibitory neuron list,
 * active neuron list, and probed neuron list files.
 * 
 * @author fumik
 * @version 1.1
 */
@SuppressWarnings("serial")
public class ExportPanel extends JPanel implements ActionListener {
	private JLabel[] labels = new JLabel[3];
	public JTextField[] tfields = new JTextField[3];
	private JButton[] btns = new JButton[3];
	private static String nlistDir = "."; // directory for neurons list file

	static final int nFields = 3; // number of input fields
	/** field index of inhibitory neurons list file */
	public static final int idxInhList = 0;
	/** field index of active neurons list file */
	public static final int idxActList = 1;
	/** field index of probed neurons list file */
	public static final int idxPrbList = 2;

	/**
	 * A class constructor, which creates UI components, and registers action
	 * listener.
	 * 
	 * @param dir
	 *            directory for neurons list file
	 */
	public ExportPanel(String dir) {
		nlistDir = dir;

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
		JFileChooser chooser = new JFileChooser(nlistDir);
		FileNameExtensionFilter filter = new FileNameExtensionFilter(
				"XML file (*.xml)", "xml");
		chooser.addChoosableFileFilter(filter);
		String dialogTitle = "";
		switch (iSource) {
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
		int option = chooser.showSaveDialog(this);
		if (option == JFileChooser.APPROVE_OPTION) {
			tfields[iSource].setText(chooser.getSelectedFile()
					.getAbsolutePath());
			nlistDir = chooser.getSelectedFile().getParent();
		}
	}
}
