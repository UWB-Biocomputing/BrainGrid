package edu.uwb.braingrid.workbenchdashboard;

import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import jdk.internal.loader.Resource;
import jdk.nashorn.internal.parser.JSONParser;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.ObjectMapper;

import edu.uwb.braingrid.general.LoggerHelper;
import edu.uwb.braingrid.workbenchdashboard.threads.RunInit;
import edu.uwb.braingrid.workbenchdashboard.userModel.User;
import edu.uwb.braingrid.workbenchdashboard.utils.Init;
import edu.uwb.braingrid.workbenchdashboard.utils.RepoManager;
import edu.uwb.braingrid.workbenchdashboard.utils.SystemProperties;

/**
 * Workbench Dashboard Application, contains main
 * @author Max Wright
 *
 */
public class WorkbenchDashboard extends Application {
	/**
	 * GSLE Growth Simulation Layout Editor
	 */
	private WorkbenchDisplay workbench_display_;

	private static final Logger LOG = Logger.getLogger(WorkbenchDashboard.class.getName());

	
	/**
	 * Probably shouldn't exist
	 * TODO: Refactor to to send the stage everywhere private
	 */
	public static Stage primaryStage_;
	
	/**
	 * WorkbenchDashboard main executable.
	 * @param args
	 */
	public static void main(String[] args) {
		
		launch(args);
	}
	
	private void initFileOutput() {
		FileHandler handler = null;
		try {
			handler = new FileHandler("WD-log.%u");
		} catch (SecurityException | IOException e) {
			LOG.severe("");
			e.printStackTrace();
		}
		if(handler != null) {
			LOG.getParent().getHandlers()[0].setLevel(LoggerHelper.MIN_LOG_LEVEL);
			LOG.getParent().addHandler(handler);
		}
	}
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		// Setup Logger
		LOG.setLevel(LoggerHelper.MIN_LOG_LEVEL);
		initFileOutput();
		
		// Get system
		SystemProperties.getSysProperties();
		
		//Load User
		User.load();
		
		// Start Application
		LOG.info("Starting Application");
		workbench_display_ = new WorkbenchDisplay(primaryStage);  // Create main display
		Scene scene = new Scene(workbench_display_, 900, 600); // Create a scene out of the display.

		// Add CSS files
		scene.getStylesheets().add("resources/simstarter/css/temp.css");
		scene.getStylesheets().add("resources/simstarter/css/tempII.css");
		scene.getStylesheets().add("resources/nledit/css/design.css");
		scene.getStylesheets().add("resources/workbenchdisplay/css/design.css");
		scene.getStylesheets().add("resources/provvisualizer/css/design.css");
		// Create Events
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
				if (arg0.getCode() == KeyCode.U && ctrl) {
					workbench_display_.pushUserViewPage();
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

		Image image = new Image(this.getClass().getClassLoader().getResourceAsStream("resources/braingrid/color-logo.png"));
		primaryStage.getIcons().add(image);
		
		primaryStage.setScene(scene);
		primaryStage.setMaximized(true);
		primaryStage.show();
		
		// Init
		RunInit runInit = new RunInit();
		runInit.start();
		
	}
	
	/**
	 * True only if ctrl has been pressed.
	 */
	private boolean ctrl = false;
}
