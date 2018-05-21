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
import edu.uwb.braingrid.workbenchdashboard.welcome.Welcome;
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

public class WorkbenchDisplay extends BorderPane {
	/**
	 * GSLE Growth Simulation Layout Editor
	 */
	private BorderPane main_border_pane_;

	private MenuBar menu_bar_;
	private Stage primaryStage_;

	public WorkbenchDisplay(Stage primary_stage) {
		main_border_pane_ = new BorderPane();
		primaryStage_ = primary_stage;
		generateMenuBar(primaryStage_);
		pushWeclomePage();
	}

	public MenuBar getMenuBar() {
		
		return menu_bar_;
	}

	private MenuBar generateMenuBar(Stage primary_stage) {
		menu_bar_ = new MenuBar();
		menu_bar_.getMenus().add(generateMenuFile(primary_stage));
		return menu_bar_;
	}

	private Menu generateMenuFile(Stage primary_stage) {
		Menu file_menu = new Menu("_File");
		file_menu.getItems().add(generateMenuNew(primary_stage));
		return file_menu;
	}

	private Menu generateMenuNew(Stage primary_stage) {
		Menu new_menu = new Menu("_New");

		// Generate Items
		MenuItem gsle = new MenuItem("_Growth Simulation Layout Editor");

		// Define Functionality
		gsle.setOnAction(event -> {
			pushGSLEPane();
		});
		
		// Add
		new_menu.getItems().add(gsle);
		return new_menu;
	}
	
	void pushGSLEPane() {
		new WorkbenchTab("GSLE", new NLedit() , this);
	}
	
	void pushWeclomePage() {
		new WorkbenchTab("Welcome!", new Welcome(), this);
	}
	
	void pushWorkbenchControlFrame() {

	}
}