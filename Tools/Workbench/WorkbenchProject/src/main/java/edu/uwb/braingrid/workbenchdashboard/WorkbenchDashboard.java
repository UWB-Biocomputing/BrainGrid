package edu.uwb.braingrid.workbenchdashboard;

import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;

import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.uwb.braingrid.workbenchdashboard.utils.SystemProperties;

public class WorkbenchDashboard extends Application {
	/**
	 * GSLE Growth Simulation Layout Editor
	 */
	private WorkbenchDisplay workbench_display_;

	private static final Logger LOG = Logger.getLogger(WorkbenchDashboard.class.getName());
	public static Level MIN_LOG_LEVEL = Level.ALL;

	public static Stage primaryStage_; // Not good, needs refactoring to send the stage everywhere private

	public static void main(String[] args) {
		launch(args);
	}

	boolean ctrl = false;

	@Override
	public void start(Stage primaryStage) throws Exception {
		FileHandler handler = new FileHandler("WD-log.%u");
		
		LOG.setLevel(MIN_LOG_LEVEL);
		LOG.getParent().getHandlers()[0].setLevel(MIN_LOG_LEVEL);
		LOG.getParent().addHandler(handler);
		
		LOG.info("Starting Application");
		workbench_display_ = new WorkbenchDisplay(primaryStage);
		Scene scene = new Scene(workbench_display_, 900, 600);

		scene.getStylesheets().add("resources/simstarter/css/temp.css");
		scene.getStylesheets().add("resources/simstarter/css/tempII.css");
		scene.getStylesheets().add("resources/nledit/css/design.css");

		scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent arg0) {
				if (arg0.getCode() == KeyCode.CONTROL) {
					ctrl = true;
				}
				if (arg0.getCode() == KeyCode.G && ctrl) {
					workbench_display_.pushGSLEPane();
				}
				if (arg0.getCode() == KeyCode.W && ctrl) {
					workbench_display_.pushWeclomePage();
				}
				if (arg0.getCode() == KeyCode.S && ctrl) {
					workbench_display_.pushSimStarterPage();
				}
				if (arg0.getCode() == KeyCode.P && ctrl) {
					workbench_display_.pushProVisStarterPage();
				}
			}
		});

		scene.setOnKeyReleased(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent arg0) {
				if (arg0.getCode() == KeyCode.CONTROL) {
					ctrl = false;
				}
			}

		});

		primaryStage.setScene(scene);
		primaryStage.setMaximized(true);

		primaryStage.show();
		
		SystemProperties.getSysProperties();
	}
}
