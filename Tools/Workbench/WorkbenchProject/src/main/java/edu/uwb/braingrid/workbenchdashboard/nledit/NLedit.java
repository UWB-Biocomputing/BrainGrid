package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.Dimension;
import java.awt.Graphics;

import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import java.util.Iterator;

import javax.swing.JOptionPane;

import javax.swing.JScrollPane;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.jdom2.output.Format;
import org.jdom2.output.XMLOutputter;


import edu.uwb.braingrid.workbench.WorkbenchManager;

import edu.uwb.braingrid.workbench.utils.DateTime;
import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.embed.swing.SwingNode;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.RadioButton;
import javafx.scene.control.RadioMenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;

public class NLedit extends WorkbenchApp {
	
	private BorderPane bp_ = new BorderPane();

	public NLedit() {
		workbenchMgr = new WorkbenchManager();
		initSettingsPanel();
		initMenu();
		generateSimulator();
	}

	@Override
	public boolean close() {
		// TODO Auto-generated method stub
		return true;
	}
	
	@Override
	public Node getDisplay() {
		return  bp_;
	}
	
	
	public BorderPane getBP() {
		return bp_;
	}
	private void generateSimulator() {
		NeuronsLayout neurons_layout = new NeuronsLayout();
		neurons_layout_ =  neurons_layout;
		
		layoutPanel = new LayoutPanel(this, new Dimension(LayoutPanel.defXCells, LayoutPanel.defYCells), neurons_layout);
		JScrollPane scrollpane = new JScrollPane(layoutPanel, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
				JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		layoutPanel.setScrollPane(scrollpane);

		SwingNode scroll_pane_node = new SwingNode();
		scroll_pane_node.setContent(scrollpane);

		bp_.setCenter(scroll_pane_node);
		nl_sim_util_ = new NL_Sim_Util(layoutPanel, neurons_layout);
	}

	/**
	 * The public function getNeuronType returns the current edit mode, which is
	 * called from LayoutPanel.
	 *
	 * @return Current edit mode: LayoutPanel.INH - inhibitory neurons edit mode.
	 *         LayoutPanel.ACT - active neurons edit mode. LayoutPanel.PRB - probed
	 *         neurons edit mode.
	 */
	public int getNeuronType() {
		if (inhNItem.isSelected()) {
			return LayoutPanel.INH;
		} else if (activeNItem.isSelected()) {
			return LayoutPanel.ACT;
		} else if (probedNItem.isSelected()) {
			return LayoutPanel.PRB;
		}
		return LayoutPanel.OTR;
	}

	private void initTools() {
		
	}
	private void initMenu() {

		// build menu items
		// File menu
		menuBar.getMenus().add(fileMenu);
		fileMenu.getItems().add(clearItem);

		clearItem.setOnAction(event -> {
			actionClear();
		});

		fileMenu.getItems().add(new SeparatorMenuItem());
		fileMenu.getItems().add(importItem);

		importItem.setOnAction(event -> {
			actionImport();
		});
		fileMenu.getItems().add(exportItem);

		exportItem.setOnAction(event -> {
			actionExport();
		});
		fileMenu.getItems().add(new SeparatorMenuItem());
		fileMenu.getItems().add(printItem);
		printItem.setOnAction(event -> {
			actionPrint();
		});
		fileMenu.getItems().add(new SeparatorMenuItem());
		fileMenu.getItems().add(exitItem);
		exitItem.setOnAction(event -> {
		});

		// Edit menu
		menuBar.getMenus().add(editMenu);
		inhNItem.setToggleGroup(editGroup);
		editMenu.getItems().add(inhNItem);
		activeNItem.setToggleGroup(editGroup);
		editMenu.getItems().add(activeNItem);
		probedNItem.setToggleGroup(editGroup);
		editMenu.getItems().add(probedNItem);
		probedNItem.setOnAction(event -> {
		});
		inhNItem.setSelected(true);

		// Layout menu
		menuBar.getMenus().add(layoutMenu);
		layoutMenu.getItems().add(bcellItem);
		bcellItem.setOnAction(event -> {
			actionBiggerCells();
		});
		layoutMenu.getItems().add(scellItem);
		scellItem.setOnAction(event -> {
			actionSmallerCells();
		});
		layoutMenu.getItems().add(new SeparatorMenuItem());
		layoutMenu.getItems().add(new SeparatorMenuItem());
		layoutMenu.getItems().add(gpatItem);
		gpatItem.setOnAction(event -> {
			actionGeneratePattern();
		});
		layoutMenu.getItems().add(aprbItem);
		aprbItem.setOnAction(event -> {
			nl_sim_util_.actionArrangeProbes();
		});
		layoutMenu.getItems().add(new SeparatorMenuItem());
		layoutMenu.getItems().add(sdatItem);
		sdatItem.setOnAction(event -> {
			actionStatisticalData();
		});

		bp_.setTop(menuBar);
	}

	private void initSettingsPanel() {
		Label lbl_sizeX = new Label("Size x:");
		Label lbl_sizeY = new Label("Size y:");

		TextField txtfld_x = new TextField("10");
		TextField txtfld_y = new TextField("10");
		Button btn_submit = new Button("Submit");
		btn_submit.setOnAction(event -> {
			int sizeX = 0, sizeY = 0;

			try {
				sizeX = Integer.parseInt(txtfld_x.getText());
				txtfld_x.setStyle("-fx-text-fill: black;");
			} catch (NumberFormatException e) {
				txtfld_x.setStyle("-fx-text-fill: red;");
			}

			try {
				sizeY = Integer.parseInt(txtfld_y.getText());
				txtfld_y.setStyle("-fx-text-fill: black;");
			} catch (NumberFormatException e) {
				txtfld_y.setStyle("-fx-text-fill: red;");
			}

			boolean inbounds_x = sizeX >= LayoutPanel.minXCells && sizeX <= LayoutPanel.maxXCells;
			boolean inbounds_y = sizeY >= LayoutPanel.minYCells && sizeY <= LayoutPanel.maxYCells;

			if (inbounds_x && inbounds_y) {
				actionModifySize(sizeX, sizeY);
				
				JScrollPane scrollpane = new JScrollPane(layoutPanel, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
						JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
				layoutPanel.setScrollPane(scrollpane);

				SwingNode scroll_pane_node = new SwingNode();
				scroll_pane_node.setContent(scrollpane);

				bp_.setCenter(scroll_pane_node);
			} else {
				if (!inbounds_x) {
					txtfld_x.setStyle("-fx-text-fill: red;");
				}
				if (!inbounds_y) {
					txtfld_y.setStyle("-fx-text-fill: red;");
				}
			}

		});


		toggle_group = new ToggleGroup();

		newButton.setToggleGroup(toggle_group);
		rptButton.setToggleGroup(toggle_group);
		altButton.setToggleGroup(toggle_group);

		HBox hbox_bottom = new HBox(lbl_sizeX, txtfld_x, lbl_sizeY, txtfld_y, btn_submit, newButton, rptButton,
				altButton);
		bp_.setBottom(hbox_bottom);
	}

	/**
	 * The 'Clear' menu handler.
	 */
	private void actionClear() {
		neurons_layout_.inhNList.clear();
		neurons_layout_.activeNList.clear();
		neurons_layout_.probedNList.clear();

		Graphics g = layoutPanel.getGraphics();
		layoutPanel.writeToGraphics(g);
	}

	/**
	 * The 'Import...' menu handler.
	 */
	private void actionImport() {
		ImportPanel myPanel = new ImportPanel();

		// int result = JOptionPane.showConfirmDialog(this, myPanel, "Import",
		// JOptionPane.OK_CANCEL_OPTION);
		int result = JOptionPane.showConfirmDialog(layoutPanel, myPanel, "Import", JOptionPane.OK_CANCEL_OPTION);
		if (result == JOptionPane.OK_OPTION) { // Afirmative
			readNeuronListFromFile(myPanel.tfields[ImportPanel.idxInhList].getText(), neurons_layout_.inhNList,
					LayoutPanel.INH);
			readNeuronListFromFile(myPanel.tfields[ImportPanel.idxActList].getText(), neurons_layout_.activeNList,
					LayoutPanel.ACT);
			readNeuronListFromFile(myPanel.tfields[ImportPanel.idxPrbList].getText(), neurons_layout_.probedNList,
					LayoutPanel.PRB);

			Graphics g = layoutPanel.getGraphics();
			layoutPanel.writeToGraphics(g);
		}
	}

	/**
	 * The function readNeuronListFromFile reads neurons list from the file
	 * specified by nameNListFile and stores neurons index in list.
	 *
	 * @param nameNListFile
	 *            file path of the neurons list (xml format).
	 * @param list
	 *            array list to store neurons index.
	 * @param type
	 *            type of neurons.
	 */
	private void readNeuronListFromFile(String nameNListFile, ArrayList<Integer> list, int type) {
		if (nameNListFile == null || nameNListFile.length() == 0) {
			return;
		}

		try {
			// read a xml file
			Document doc = new SAXBuilder().build(new File(nameNListFile));
			Element root = doc.getRootElement();
			if ((root != null) && ((root.getName() == "A" && type == LayoutPanel.ACT)
					|| (root.getName() == "I" && type == LayoutPanel.INH)
					|| (root.getName() == "P" && type == LayoutPanel.PRB))) {
				list.clear();
				String[] parts = root.getValue().split("[ \n\r]");

				Dimension size = layoutPanel.getLayoutSize();
				int numNeurons = size.height * size.width;
				for (String part : parts) {
					try {
						int index = Integer.parseInt(part);
						if (index < numNeurons) { // ignore indexes greater than
							// numNeurons
							list.add(index);
						}
					} catch (NumberFormatException e) {
						System.err.println("Illegal number :" + part);
					}
				}
			}
		} catch (JDOMException je) {
			System.err.println(je);
		} catch (IOException ie) {
			System.err.println(ie);
		}
	}

	/**
	 * The 'Export...' menu handler.
	 */
	private void actionExport() {
		ExportPanel myPanel = new ExportPanel(ImportPanel.nlistDir);
		// int result = JOptionPane.showConfirmDialog(this, myPanel, "Export",
		// JOptionPane.OK_CANCEL_OPTION);
		int result = JOptionPane.showConfirmDialog(layoutPanel, myPanel, "Export", JOptionPane.OK_CANCEL_OPTION);
		Long functionStartTime = System.currentTimeMillis();
		Long accumulatedTime = 0L;

		if (result == JOptionPane.OK_OPTION) { // Afirmative
			writeNeuronListToFile(myPanel.tfields[ExportPanel.idxInhList].getText(), neurons_layout_.inhNList,
					LayoutPanel.INH);
			// add to workbench project
			if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
				Long startTime = System.currentTimeMillis();
				workbenchMgr.getProvMgr().addFileGeneration("InhibitoryNeuronListExport" + java.util.UUID.randomUUID(),
						"neuronListExport", "NLEdit", null, false, myPanel.tfields[ExportPanel.idxInhList].getText(),
						null, null);

				accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
			}

			writeNeuronListToFile(myPanel.tfields[ExportPanel.idxActList].getText(), neurons_layout_.activeNList,
					LayoutPanel.ACT);
			// add to workbench project
			if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
				Long startTime = System.currentTimeMillis();
				workbenchMgr.getProvMgr().addFileGeneration("ActiveNeuronListExport" + java.util.UUID.randomUUID(),
						"neuronListExport", "NLEdit", null, false, myPanel.tfields[ExportPanel.idxActList].getText(),
						null, null);
				accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
			}

			writeNeuronListToFile(myPanel.tfields[ExportPanel.idxPrbList].getText(), neurons_layout_.probedNList,
					LayoutPanel.PRB);
			// add to workbench project
			if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
				Long startTime = System.currentTimeMillis();
				workbenchMgr.getProvMgr().addFileGeneration("ProbedNeuronListExport" + java.util.UUID.randomUUID(),
						"neuronListExport", "NLEdit", null, false, myPanel.tfields[ExportPanel.idxPrbList].getText(),
						null, null);
				accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
			}
		}
		DateTime.recordFunctionExecutionTime("ControlFrame", "actionExport",
				System.currentTimeMillis() - functionStartTime, workbenchMgr.isProvEnabled());
		if (workbenchMgr.isProvEnabled()) {
			DateTime.recordAccumulatedProvTiming("ControlFrame", "actionExport", accumulatedTime);
		}
	}

	/**
	 * The function writeNeuronListToFile creates a neurons list file specified by
	 * list and type.
	 *
	 * @param nameNListFile
	 *            file path of the neurons list.
	 * @param list
	 *            array list of neurons index.
	 * @param type
	 *            type of neurons.
	 */
	private void writeNeuronListToFile(String nameNListFile, ArrayList<Integer> list, int type) {
		if (nameNListFile == null || nameNListFile.length() == 0) {
			return;
		}

		try {
			Element root = null;
			if (type == LayoutPanel.INH) { // inhibitory neurons
				root = new Element("I");
			} else if (type == LayoutPanel.ACT) { // active neurons
				root = new Element("A");
			} else if (type == LayoutPanel.PRB) { // probed neurons
				root = new Element("P");
			}

			// create a xml file
			String sList = "";
			Iterator<Integer> iter = list.iterator();
			while (iter.hasNext()) {
				sList += " " + iter.next();
			}
			root.setText(sList);

			Document doc = new Document();
			doc.setRootElement(root);

			XMLOutputter xmlOutput = new XMLOutputter(Format.getPrettyFormat());
			xmlOutput.output(doc, new FileOutputStream(nameNListFile));
		} catch (IOException ie) {
			System.err.println(ie);
		}
	}

	/**
	 * The 'Bigger cells' menu handler.
	 */
	private void actionBiggerCells() {
		layoutPanel.changeCellSize(true);
	}

	/**
	 * The 'Smaller cells' menu handler.
	 */
	private void actionSmallerCells() {
		layoutPanel.changeCellSize(false);
	}

	/**
	 * The 'Print...' menu handler.
	 */
	private void actionPrint() {
		// get PrinterJob
		PrinterJob job = PrinterJob.getPrinterJob();
		MyPrintable printable = new MyPrintable(job.defaultPage(), layoutPanel, nl_sim_util_);

		// setup Printable, Pageable
		job.setPrintable(printable);
		job.setPageable(printable);

		// display print dialog and print
		if (job.printDialog()) {
			try {
				job.print();
			} catch (PrinterException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * The 'Modify size...' menu handler.
	 */
	private void actionModifySize(int sizeX, int sizeY) {
		RepType rtype = RepType.CLEAR;
		if (newButton.isSelected()) {
			rtype = RepType.CLEAR;
			//System.out.println("Clear");
		} else if (rptButton.isSelected()) {
			rtype = RepType.REPEAT;
			//System.out.println("Repeat");
		} else if (altButton.isSelected()) {
			rtype = RepType.ALT;
			//System.out.println("Alternate");
		}
		//System.out.println("Default");
		changeLayoutSize(new Dimension(sizeX, sizeY), rtype);

	}

	/**
	 * The function changeLayoutSize generates new neurons lists of inhNList,
	 * activeNList, and activeNList, and changes the size of the layout panel.
	 *
	 * @param newSize
	 *            size for the new layout panel.
	 * @param rtype
	 *            repeat type, CLEAR, REPEAT, or ALT.
	 */
	private void changeLayoutSize(Dimension newSize, RepType rtype) {
		neurons_layout_.inhNList = nl_sim_util_.repPattern(newSize, neurons_layout_.inhNList, rtype);
		neurons_layout_.activeNList = nl_sim_util_.repPattern(newSize, neurons_layout_.activeNList, rtype);
		neurons_layout_.probedNList = nl_sim_util_.repPattern(newSize, neurons_layout_.activeNList, rtype);

		layoutPanel.changeLayoutSize(newSize);
	}

	/**
	 * The 'Generate pattern' menu handler.
	 */
	private void actionGeneratePattern() {
		GPatternPanel gpatPanel = new GPatternPanel();
		// int result = JOptionPane.showConfirmDialog(this, gpatPanel,
		// "Generate pattern", JOptionPane.OK_CANCEL_OPTION);
		int result = JOptionPane.showConfirmDialog(layoutPanel, gpatPanel, "Generate pattern",
				JOptionPane.OK_CANCEL_OPTION);
		if (result == JOptionPane.OK_OPTION) { // Afirmative
			try {
				float ratioInh = Float.parseFloat(gpatPanel.tfields[GPatternPanel.idxINH].getText());
				float ratioAct = Float.parseFloat(gpatPanel.tfields[GPatternPanel.idxACT].getText());

				// validate ratios
				if ((ratioInh < 0 || ratioInh > 1.0) || (ratioAct < 0 || ratioAct > 1.0)
						|| (ratioInh + ratioAct > 1.0)) {
					throw new NumberFormatException();
				}

				if (gpatPanel.btns[GPatternPanel.idxREG].isSelected()) {
					nl_sim_util_.genRegularPattern(ratioInh, ratioAct);
				} else if (gpatPanel.btns[GPatternPanel.idxRND].isSelected()) {
					nl_sim_util_.genRandomPattern(ratioInh, ratioAct);
				}

				Graphics g = layoutPanel.getGraphics();
				layoutPanel.writeToGraphics(g);
			} catch (NumberFormatException ne) {
				JOptionPane.showMessageDialog(null, "Invalid ratio.");
			}
		}
	}

	/**
	 * The 'Statistical data...' menu handler.
	 */
	private void actionStatisticalData() {
		String message = nl_sim_util_.getStatisticalMsg(true);

		JOptionPane.showMessageDialog(null, message, "Statistical data", JOptionPane.PLAIN_MESSAGE);
	}

	private LayoutPanel layoutPanel; // reference to the layout panel
	NL_Sim_Util nl_sim_util_;

	// menus
	private MenuBar menuBar = new MenuBar();

	// File menu
	private Menu fileMenu = new Menu("File");
	private MenuItem importItem = new MenuItem("_Import...");
	private MenuItem exportItem = new MenuItem("_Export...");
	private MenuItem clearItem = new MenuItem("_Clear");
	private MenuItem printItem = new MenuItem("_Print...");
	private MenuItem exitItem = new MenuItem("E_xit");

	// Edit menu
	private Menu editMenu = new Menu("Edit");
	private ToggleGroup editGroup = new ToggleGroup();
	private RadioMenuItem inhNItem = new RadioMenuItem("Inhibitory neurons");
	private RadioMenuItem activeNItem = new RadioMenuItem("Active neurons");
	private RadioMenuItem probedNItem = new RadioMenuItem("Probed neurons");

	// Layout menu
	private Menu layoutMenu = new Menu("Layout");
	private MenuItem bcellItem = new MenuItem("_Bigger cells");
	private MenuItem scellItem = new MenuItem("_Smaller cells");
	private MenuItem csizeItem = new MenuItem("_Modify size...");
	private MenuItem gpatItem = new MenuItem("_Generate pattern...");
	private MenuItem aprbItem = new MenuItem("_Arrange probes...");
	private MenuItem sdatItem = new MenuItem("Statistical _data...");

	// Reference to workbench (or other frame code launching NLEdit)
	private WorkbenchManager workbenchMgr;

	private RadioButton newButton = new RadioButton("New");
	private RadioButton rptButton = new RadioButton("Repeat");
	private RadioButton altButton = new RadioButton("Alternate");
	
	ToggleGroup toggle_group;
	
	private NeuronsLayout neurons_layout_;

	// repeat type for modify size
	public enum RepType {
		REPEAT, ALT, CLEAR
	}

}