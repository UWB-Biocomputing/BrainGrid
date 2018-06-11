package edu.uwb.braingrid.workbenchdashboard.setup;

import java.util.Date;

import javax.swing.JFileChooser;

import edu.uwb.braingrid.workbench.WorkbenchManager;
import edu.uwb.braingrid.workbench.utils.DateTime;
import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

public class SimStarter extends WorkbenchApp {

	private BorderPane bp_ = new BorderPane();

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
	private SimStarterToolBar sim_starter_tool_bar_;
	// private Selector sim_starter_helper_ = new Selector(workbenchMgr);

	private TextArea msgText = new TextArea("");

	public SimStarter() {
		// initMenu();
		sim_starter_tool_bar_ = new SimStarterToolBar(this);
		bp_.setTop(sim_starter_tool_bar_);
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


	public void newProject() {
		if (workbenchMgr.newProject()) {
			resetUILabelText();
			progress_bar_.setVisible(false);
			current_proj_lbl_.setText(workbenchMgr.getProjectName());
			sim_starter_tool_bar_.disableProvidence(!workbenchMgr.isProvEnabled());
			configure_sim_btn_.setDisable(false);
			specify_script_btn_.setDisable(false);
			sim_starter_tool_bar_.disableSave(false);
		}
		setMsg();
	}

	public void openProject() {
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

	public void saveProject() {
		workbenchMgr.saveProject();
		setMsg();
	}

	public void viewProvenance() {
		setMsg();
		workbenchMgr.viewProvenance();
	}

	public void manageParamsClasses() {
		if (workbenchMgr.configureParamsClasses()) {
			updateProjectOverview();
		}
		setMsg();
	}

	private void initCenter() {
		project_title_label_.setText("Current Project: ");
		current_proj_lbl_.setText("None");
		HBox proj = new HBox(project_title_label_, current_proj_lbl_);
		initAttributes();
		//VBox center = new VBox(proj, initAttributeDisplay(), msgText);
		VBox center = new VBox(proj, initAttributeDisplayGridPane(), msgText);
		bp_.setCenter(center);
	}
	
	private void initAttributes() {
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
	}
	
	private GridPane initAttributeDisplayGridPane() {
		GridPane gp = new GridPane();
		gp.getStyleClass().add("gridpane-initattributes");
		
		gp.getChildren().addAll(configure_sim_btn_, specify_script_btn_, script_gen_btn_, run_script_btn_,
				analyze_output_btn_);
		gp.getChildren().addAll(simulation_lbl_, script_specify_lbl_, script_generated_lbl_, run_script_lbl_,
				analyze_output_sts_lbl_);
		gp.getChildren().addAll(simulation_out_lbl_, script_specify_out_lbl_, script_generated_out_lbl_,
				run_script_out_lbl_, analyze_output_sts_out_lbl_);
		
		GridPane.setConstraints(configure_sim_btn_, 0, 0);
		GridPane.setConstraints(specify_script_btn_, 0, 1);
		GridPane.setConstraints(script_gen_btn_, 0, 2);
		GridPane.setConstraints(run_script_btn_, 0, 3);
		GridPane.setConstraints(analyze_output_btn_, 0, 4);
		
		GridPane.setConstraints(simulation_lbl_, 1, 0);
		GridPane.setConstraints(script_specify_lbl_, 1, 1);
		GridPane.setConstraints(script_generated_lbl_, 1, 2);
		GridPane.setConstraints(run_script_lbl_, 1, 3);
		GridPane.setConstraints(analyze_output_sts_lbl_, 1, 4);
		
		GridPane.setConstraints(simulation_out_lbl_, 2, 0);
		GridPane.setConstraints(script_specify_out_lbl_, 2, 1);
		GridPane.setConstraints(script_generated_out_lbl_, 2, 2);
		GridPane.setConstraints(run_script_out_lbl_, 2, 3);
		GridPane.setConstraints(analyze_output_sts_out_lbl_, 2, 4);
		
		return gp;
		
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
		sim_starter_tool_bar_.disableSave(true);
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
