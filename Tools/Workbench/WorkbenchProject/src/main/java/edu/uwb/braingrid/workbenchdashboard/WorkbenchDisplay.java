package edu.uwb.braingrid.workbenchdashboard;

import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SingleSelectionModel;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.layout.BorderPane;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

import edu.uwb.braingrid.workbench.provvisualizer.ProvVisGlobal;
import edu.uwb.braingrid.workbenchdashboard.nledit.NLedit;
import edu.uwb.braingrid.workbenchdashboard.provis.ProVis;
import edu.uwb.braingrid.workbenchdashboard.simstarter.SimStarter;
import edu.uwb.braingrid.workbenchdashboard.userModel.User;
import edu.uwb.braingrid.workbenchdashboard.userView.UserView;
import edu.uwb.braingrid.workbenchdashboard.utils.RepoManager;
import edu.uwb.braingrid.workbenchdashboard.welcome.Welcome;
import javafx.stage.Stage;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.api.errors.InvalidRemoteException;
import org.eclipse.jgit.api.errors.TransportException;

/**
 * Defines the main display of the screen along with global functionality.
 * @author Max Wright
 */
public class WorkbenchDisplay extends BorderPane {	
	private static final Logger LOG = Logger.getLogger(WorkbenchDisplay.class.getName());
	
	/**
	 * The top menu bar of the screen.
	 */
	private MenuBar menu_bar_;
	private Stage primaryStage_;
	
	/**
	 * The main content of the screen
	 */
	private TabPane tp_ = new TabPane();

	/**
	 * 
	 * @param primary_stage The Stage object of the fx instance.
	 */
	public WorkbenchDisplay(Stage primary_stage) {
		LOG.info("new " + getClass().getName());
		primaryStage_ = primary_stage;
		setTop(generateMenuBar(primaryStage_));
		setBottom(new WorkbenchStatusBar());
		pushWeclomePage();
		setCenter(tp_);
	}

	public MenuBar getMenuBar() {
		return menu_bar_;
	}

	private MenuBar generateMenuBar(Stage primary_stage) {
		menu_bar_ = new MenuBar();
		menu_bar_.getMenus().add(generateMenuFile(primary_stage));
		menu_bar_.getMenus().add(generateMenuRepo());
		return menu_bar_;
	}
	
	/**
	 * Generates all functionality associated with the "File" tab of the menu bar.
	 * @param primary_stage The Stage of the FX program
	 * @return A complete menu.
	 */
	private Menu generateMenuFile(Stage primary_stage) {
		Menu file_menu = new Menu("_File");
		file_menu.getItems().add(generateMenuNew(primary_stage));
		file_menu.getItems().add(new MenuItem("Open"));
		file_menu.getItems().add(generateMenuRecentProjects());
		return file_menu;
	}

	private Menu generateMenuRecentProjects() {
		Menu recent_proj_menu = new Menu("Recent Projects");
		return recent_proj_menu;
	}

	private Menu generateMenuNew(Stage primary_stage) {
		Menu new_menu = new Menu("_New");

		// Generate Items
		MenuItem gsle = new MenuItem("_Growth Simulation Layout Editor");

		// Define Functionality
		gsle.setOnAction(event -> {
			pushGSLEPane();
		});

		// Generate Items
		MenuItem simstarter = new MenuItem("_Simulation Starter");

		// Define Functionality
		simstarter.setOnAction(event -> {
			pushSimStarterPage();
		});

		// Generate Items
		MenuItem provis = new MenuItem("_ProVis");

		// Define Functionality
		provis.setOnAction(event -> {
			pushProVisStarterPage();
		});

		// Add
		new_menu.getItems().add(gsle);
		new_menu.getItems().add(simstarter);
		new_menu.getItems().add(provis);
		return new_menu;
	}
	
	private Menu generateMenuRepo() {
		Menu repo_menu = new Menu("_Repo");
		MenuItem updateMain = new MenuItem("Update Main");
		
		updateMain.setOnAction(event -> {
			RepoManager.getMasterBranch();
		});
		
		repo_menu.getItems().add(updateMain);
		
		return repo_menu;
	}
	
	

	/**
	 * Adds a new Growth Simulator Layout Editor tab
	 */
	void pushGSLEPane() {
		Tab tab = new Tab();
		NLedit pv = new NLedit(tab);
		tab.setContent(pv.getDisplay());
		tp_.getTabs().add(tab);
		SingleSelectionModel<Tab> selectionModel = tp_.getSelectionModel();
		selectionModel.select(tab);
	}
	
	/**
	 * Adds a new Welcome tab
	 */
	void pushWeclomePage() {
		Tab tab = new Tab();
		Welcome pv = new Welcome(tab);
		tab.setContent(pv.getDisplay());
		tp_.getTabs().add(tab);
		SingleSelectionModel<Tab> selectionModel = tp_.getSelectionModel();
		selectionModel.select(tab);
	}

	/**
	 * Adds a new Simulator Starter tab
	 */
	void pushSimStarterPage() {
		Tab tab = new Tab();
		SimStarter pv = new SimStarter(tab);
		tab.setContent(pv.getDisplay());
		tp_.getTabs().add(tab);
		SingleSelectionModel<Tab> selectionModel = tp_.getSelectionModel();
		selectionModel.select(tab);
	}

	/**
	 * Add a new Providence Visualizer tab
	 */
	void pushProVisStarterPage() {
		Tab tab = new Tab();
		ProVis pv = new ProVis(tab);
		tab.setContent(pv.getDisplay());
		tp_.getTabs().add(tab);
		SingleSelectionModel<Tab> selectionModel = tp_.getSelectionModel();
		selectionModel.select(tab);
	}
	
	void pushUserViewPage() {
		Tab tab = new Tab();
		UserView uv = new UserView(tab);
		tab.setContent(uv.getDisplay());
		tp_.getTabs().add(tab);
		SingleSelectionModel<Tab> selectionModel = tp_.getSelectionModel();
		selectionModel.select(tab);
	}
}
