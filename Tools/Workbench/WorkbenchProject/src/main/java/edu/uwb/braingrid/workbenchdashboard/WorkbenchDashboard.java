package edu.uwb.braingrid.workbenchdashboard;

import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ToggleGroup;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import java.awt.Color;
import java.awt.Dimension;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

import edu.uwb.braingrid.tools.nledit.LayoutPanel;
import edu.uwb.braingrid.workbenchdashboard.nledit.NLedit;
import javafx.application.Application;
import javafx.embed.swing.SwingNode;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Paint;
import javafx.stage.Stage;

public class WorkbenchDashboard extends Application {
	/**
	 * GSLE Growth Simulation Layout Editor
	 */
	private WorkbenchDisplay workbench_display_;

	public static void main(String[] args) {
		launch(args);
	}
	
	boolean ctrl = false;
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		primaryStage.setTitle("Workbench Dashboard");
		
		workbench_display_ = new WorkbenchDisplay(primaryStage);
		
		Scene scene = new Scene(workbench_display_, 900, 600);
		scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent arg0) {
				if(arg0.getCode() == KeyCode.CONTROL) {
					ctrl = true;
				}
				if(arg0.getCode() == KeyCode.G && ctrl ) {
					workbench_display_.pushGSLEPane();
				}
				if(arg0.getCode() == KeyCode.W && ctrl) {
					workbench_display_.pushWeclomePage();
				}
				if(arg0.getCode() == KeyCode.S && ctrl) {
					workbench_display_.pushSimStarterPage();
				}
				if(arg0.getCode() == KeyCode.P && ctrl) {
					workbench_display_.pushProVisStarterPage();
				}
			}
		});
		
		scene.setOnKeyReleased(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent arg0) {
				if(arg0.getCode() == KeyCode.CONTROL) {
					ctrl = false;
				}
			}
			
		});
		
		primaryStage.setScene(scene);
		primaryStage.setMaximized(true);
		
		primaryStage.show();

	}
}
