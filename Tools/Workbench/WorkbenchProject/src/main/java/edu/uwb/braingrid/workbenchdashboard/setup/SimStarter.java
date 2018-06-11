package edu.uwb.braingrid.workbenchdashboard.setup;

import java.util.Date;

import javax.swing.JFileChooser;

import edu.uwb.braingrid.workbench.WorkbenchManager;
import edu.uwb.braingrid.workbench.utils.DateTime;
import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

public class SimStarter extends WorkbenchApp {

	private BorderPane bp_ = new BorderPane();

	/*
	 * Menu Vars
	 */
	private MenuBar menu_bar_ = new MenuBar();
	private Menu file_menu_ = new Menu("File");
	private MenuItem new_menu_item_ = new MenuItem("New");
	private MenuItem open_proj_menu_item_ = new MenuItem("Open");
	private MenuItem save_proj_menu_item_ = new MenuItem("Save");

	private Menu view_menu_ = new Menu("View");
	private MenuItem view_provenance_menu_item_ = new MenuItem("Provenance");

	private Menu tools_menu_ = new Menu("Tools");
	private MenuItem mng_params_classes_ = new MenuItem("Manage Params Classes");
	/*
	 * End Menu Vars
	 */

	/*
	 * Center Vars
	 */
	private Label project_title_label_ = new Label();
	private Label current_proj_lbl_ = new Label();

	private Button configure_sim_btn_ = new Button("Configure");
	private Button specify_script_btn_ = new Button("Specify");
	private Button script_gen_btn_ = new Button("Generate");
	private Button run_script_btn_ = new Button("Run Script");
	private Button analyze_output_btn_ = new Button("Analyze");

	private Label simulation_lbl_ = new Label("Simulation: ");
	private Label script_specify_lbl_ = new Label("Script: ");
	private Label script_generated_lbl_ = new Label("Script: ");
	private Label run_script_lbl_ = new Label("Status: ");
	private Label analyze_output_sts_lbl_ = new Label("Analyze: ");

	private Label simulation_out_lbl_ = new Label();
	private Label script_specify_out_lbl_ = new Label();
	private Label script_generated_out_lbl_ = new Label();
	private Label run_script_out_lbl_ = new Label();
	private Label analyze_output_sts_out_lbl_ = new Label();

	private ProgressBar progress_bar_ = new ProgressBar(0);
	/*
	 * End Center Vars
	 */

	private WorkbenchManager workbenchMgr = new WorkbenchManager();
	// private Selector sim_starter_helper_ = new Selector(workbenchMgr);

	private TextArea msgText = new TextArea("");

	public SimStarter() {
		initMenu();
		initCenter();
	}

	@Override
	public boolean close() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public Node getDisplay() {
		return bp_;
	}

	private void initMenu() {
		new_menu_item_.setOnAction(event -> {
			newProjectMenuItemActionPerformed();
		});

		open_proj_menu_item_.setOnAction(event -> {
			openProjectMenuItemActionPerformed();
		});

		save_proj_menu_item_.setOnAction(event -> {
			saveProjectMenuItemActionPerformed();
		});

		file_menu_.getItems().add(new_menu_item_);
		file_menu_.getItems().add(open_proj_menu_item_);
		file_menu_.getItems().add(save_proj_menu_item_);

		view_provenance_menu_item_.setOnAction(event -> {
			viewProvenanceMenuItemActionPerformed();
		});

		view_menu_.getItems().add(view_provenance_menu_item_);

		mng_params_classes_.setOnAction(event -> {
			ManageParamsClassesActionPerformed();
		});
		tools_menu_.getItems().add(mng_params_classes_);

		menu_bar_.getMenus().add(file_menu_);
		menu_bar_.getMenus().add(view_menu_);
		menu_bar_.getMenus().add(tools_menu_);

		bp_.setTop(menu_bar_);
	}

	private void newProjectMenuItemActionPerformed() {
		if (workbenchMgr.newProject()) {
			resetUILabelText();
			progress_bar_.setVisible(false);
			current_proj_lbl_.setText(workbenchMgr.getProjectName());
			view_provenance_menu_item_.setDisable(!workbenchMgr.isProvEnabled());
			configure_sim_btn_.setDisable(false);
			specify_script_btn_.setDisable(false);
			save_proj_menu_item_.setDisable(false);
		}
		setMsg();
	}

	private void openProjectMenuItemActionPerformed() {
		int code = workbenchMgr.openProject();
		switch (code) {
		case JFileChooser.APPROVE_OPTION:
			updateProjectOverview();
			break;
		case JFileChooser.CANCEL_OPTION:
			break;
		case JFileChooser.ERROR_OPTION:
			break;
		case WorkbenchManager.EXCEPTION_OPTION:
		default:
			resetUILabelText();
			break;
		}
	}

	private void saveProjectMenuItemActionPerformed() {
		workbenchMgr.saveProject();
		setMsg();
	}

	private void viewProvenanceMenuItemActionPerformed() {
		setMsg();
		workbenchMgr.viewProvenance();
	}

	private void ManageParamsClassesActionPerformed() {
		if (workbenchMgr.configureParamsClasses()) {
			updateProjectOverview();
		}
		setMsg();
	}

	private void initCenter() {
		project_title_label_.setText("Current Project: ");
		current_proj_lbl_.setText("None");
		HBox proj = new HBox(project_title_label_, current_proj_lbl_);
		VBox center = new VBox(proj, initAttributes(), msgText);
		bp_.setCenter(center);
	}

	private HBox initAttributes() {
		configure_sim_btn_.setOnAction(event -> {
			configureSimulationButtonActionPerformed();
		});
		specify_script_btn_.setOnAction(event -> {
			specifyScriptButtonActionPerformed();
		});
		script_gen_btn_.setOnAction(event -> {
			scriptGenerateButtonActionPerformed();
		});
		run_script_btn_.setOnAction(event -> {
			runScriptButtonActionPerformed();
		});
		analyze_output_btn_.setOnAction(event -> {
			analyzeOutputButtonActionPerformed();
		});
		resetUILabelText();
		
		disableProjectAttributeRelatedButtons();

		VBox btns = new VBox(configure_sim_btn_, specify_script_btn_, script_gen_btn_, run_script_btn_,
				analyze_output_btn_);
		VBox lbls = new VBox(simulation_lbl_, script_specify_lbl_, script_generated_lbl_, run_script_lbl_,
				analyze_output_sts_lbl_);
		VBox txt = new VBox(simulation_out_lbl_, script_specify_out_lbl_, script_generated_out_lbl_,
				run_script_out_lbl_, analyze_output_sts_out_lbl_);
		HBox overview = new HBox(btns, lbls, txt);
		return overview;
		// <editor-fold defaultstate="collapsed" desc="Auto-Generated">
		/**
		 * This method is called from within the constructor to initialize the form.
		 * WARNING: Do NOT modify this code. The content of this method is always
		 * regenerated by the Form Editor.
		 */
		// <editor-fold defaultstate="collapsed" desc="Generated
		// Code">//GEN-BEGIN:initComponents
		// private void initComponents() {
		//

		// projectTitleTextLabel = new Label();
		// simulationConfigurationLabel = new Label();
		// scriptSpecificationLabel = new Label();
		// outputFilenameLabel = new Label();
		// projectOverviewLabel = new Label();
		// msgLabel = new Label();

		// generatedScriptFilenameLabel = new Label();
		// scriptGeneratedLabel = new Label();

		// runScriptStatusLabel = new Label();
		// scriptStatusMsgLabel = new Label();
		// jScrollPane1 = new javax.swing.JScrollPane();
		// transferProgressBar = new javax.swing.JProgressBar();

		//
		// projectTitleTextLabel.setText("None");
		//
		// simulationConfigurationLabel.setText("None");
		// simulationConfigurationLabel.setAlignment(Pos.TOP_CENTER);
		//
		// scriptSpecificationLabel.setText("None");
		// scriptSpecificationLabel.setAlignment(Pos.TOP_CENTER);
		//
		// outputFilenameLabel.setText("None");
		// outputFilenameLabel.setAlignment(Pos.TOP_CENTER);
		//
		// projectOverviewLabel.setText("Project Overview");
		//
		// msgLabel.setText("<html><i>Workbench Message: </i><b><p
		// style=\"color:green\"></p></html>");
		// msgLabel.setAlignment(Pos.TOP_CENTER);
		//

		//
		// generatedScriptFilenameLabel.setText("None");
		// generatedScriptFilenameLabel.setAlignment(Pos.TOP_CENTER);
		//

		// //scriptGeneratedLabel.setPreferredSize(null);
		//
		// runScriptButton.setText("Run Script");
		// runScriptButton.setEnabled(false);

		//
		// //runScriptStatusLabel.setPreferredSize(null);
		//
		// scriptStatusMsgLabel.setText("None");
		//
		// msgText.setBackground(new java.awt.Color(225, 225, 225));
		// msgText.setColumns(20);
		// msgText.setRows(5);
		// msgText.setAutoscrolls(false);
		// msgText.setFocusable(false);
		// jScrollPane1.setViewportView(msgText);
		//
		// transferProgressBar.setAlignmentX(-0.5F);
		// transferProgressBar.setFocusable(false);
		//
		//

		// SwingNode pane = new SwingNode();
		// pane.setContent(MainMenuBar);
		// bp_.setTop(pane);
		//
		// HBox hbox = new HBox();
		// VBox vbox = new VBox();
		//
		// }
	}

	/**
	 * Prompts the user to specify the simulator used. This should be the file that
	 * was invoked, which used the input files specified, in order to write the
	 * output file that was specified.
	 *
	 * @param evt
	 *            - The event that triggered this action
	 */
	private void specifyScriptButtonActionPerformed() {// GEN-FIRST:event_specifyScriptButtonActionPerformed
		if (workbenchMgr.specifyScript()) {
			workbenchMgr.invalidateScriptGenerated();
			workbenchMgr.invalidateScriptRan();
			workbenchMgr.invalidateScriptAnalyzed();
			updateProjectOverview();
			script_gen_btn_.setDisable(false);
		}
		setMsg();
	}// GEN-LAST:event_specifyScriptButtonActionPerformed

	/**
	 * Allows the user to select a simulation output file from the file system. This
	 * file then validated and added to the project XML and provenance model.
	 *
	 * @param evt
	 *            - The event that triggered this action
	 */
	private void analyzeOutputButtonActionPerformed() {
		long timeCompleted = workbenchMgr.analyzeScriptOutput();
		if (timeCompleted != DateTime.ERROR_TIME) {
			analyze_output_sts_out_lbl_.setText("Completed at: " + DateTime.getTime(timeCompleted));
		} else {
			analyze_output_sts_out_lbl_.setText("Script execution incomplete, try again later.");
		}
		setMsg();
	}

	private void scriptGenerateButtonActionPerformed() {// GEN-FIRST:event_scriptGenerateButtonActionPerformed
		if (workbenchMgr.generateScript()) {
			updateProjectOverview();
			script_gen_btn_.setDisable(true);
			run_script_btn_.setDisable(false);
		}
		setMsg();
		// pack();
	}// GEN-LAST:event_scriptGenerateButtonActionPerformed

	/**
	 * Runs the script on the remote host.
	 *
	 * Connection information is entered in a SSHConnectionDialog
	 *
	 * @param evt
	 */
	private void runScriptButtonActionPerformed() {// GEN-FIRST:event_runScriptButtonActionPerformed
		if (workbenchMgr.runScript()) {
			String time = DateTime.getTime(new Date().getTime());
			String msg = "Script execution started at: " + time;
			simulation_out_lbl_.setText(msg);
			analyze_output_btn_.setDisable(false);
			run_script_btn_.setDisable(true);
		}
		setMsg();
	}// GEN-LAST:event_runScriptButtonActionPerformed

	/**
	 * Prompts the user to select files for the simulation input. InputAnalyzer
	 * files are created with NLEdit or by hand in XML. InputAnalyzer files
	 * represent lists of neurons with regard to their position in a neuron array
	 * (e.g. position 12 is x: 1, y: 2 on a 10x10 grid)
	 *
	 * @param evt
	 *            - The event that triggered this action
	 */
	private void configureSimulationButtonActionPerformed() {// GEN-FIRST:event_configureSimulationButtonActionPerformed
		if (workbenchMgr.configureSimulation()) {
			workbenchMgr.invalidateScriptGenerated();
			workbenchMgr.invalidateScriptRan();
			workbenchMgr.invalidateScriptAnalyzed();
			updateProjectOverview();
		}
		setMsg();
		// pack();
	}// GEN-LAST:event_configureSimulationButtonActionPerformed

	/**
	 * Resets the UI text
	 */
	private void resetUILabelText() {
		simulation_out_lbl_.setText("None");
		script_specify_out_lbl_.setText("None");
		script_generated_out_lbl_.setText("None");
		run_script_out_lbl_.setText("None");
		analyze_output_sts_out_lbl_.setText("None");

		progress_bar_.setProgress(0);
	}

	void updateProjectOverview() {
		resetUILabelText();
		current_proj_lbl_.setText(workbenchMgr.getProjectName());
		displaySimConfigFile();
		updateSimOverview();
		// transferProgressBar.setVisible(workbenchMgr.isSimExecutionRemote());
		displayScriptGenerationOverview();
		displayScriptRunOverview();
		displayScriptAnalysisOverview();
		// enableInitialButtons();
	}

	private void updateSimOverview() {
		String overview = workbenchMgr.getSimulationOverview();
		script_specify_out_lbl_.setText(overview);
	}

	private void displaySimConfigFile() {
		String labelText = workbenchMgr.getSimConfigFileOverview();
		simulation_out_lbl_.setText(labelText);
	}

	private void displayScriptGenerationOverview() {
		String filename = workbenchMgr.getScriptPath();
		if (filename != null) {
			script_generated_out_lbl_.setText(filename);
		}
	}

	private void displayScriptRunOverview() {
		String runAtMsg = workbenchMgr.getScriptRunOverview();
		if (runAtMsg != null) {
			run_script_out_lbl_.setText(runAtMsg);
		}
	}

	private void displayScriptAnalysisOverview() {
		String overview = workbenchMgr.getScriptAnalysisOverview();
		if (overview != null) {
			analyze_output_sts_out_lbl_.setText(overview);
		}
	}

	private void disableProjectAttributeRelatedButtons() {
		configure_sim_btn_.setDisable(true);
		specify_script_btn_.setDisable(true);
		script_gen_btn_.setDisable(true);
		run_script_btn_.setDisable(true);
		analyze_output_btn_.setDisable(true);
		save_proj_menu_item_.setDisable(true);
	}

	/**
	 * Sets the workbench message content. The content of this message is based on
	 * the accumulated messages of produced by the functions of the workbench
	 * manager.
	 *
	 */
	public void setMsg() {
		msgText.setText(workbenchMgr.getMessages());
	}
}
