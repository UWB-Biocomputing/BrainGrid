package edu.uwb.braingrid.workbenchdashboard.simstarter;

import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;

public class SimStarterToolBar extends MenuBar {

	public SimStarterToolBar(SimStarter simstarter) {
		simstarter_ = simstarter;
		initMenu();
		disableProvidence(true);
	}

	public void disableSave(boolean val) {
		save_proj_menu_item_.setDisable(val);
	}

	public void disableProvidence(boolean val) {
		view_provenance_menu_item_.setDisable(val);
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

		getMenus().add(file_menu_);
		getMenus().add(view_menu_);
		getMenus().add(tools_menu_);
	}

	private void newProjectMenuItemActionPerformed() {
		simstarter_.newProject();
	}

	private void openProjectMenuItemActionPerformed() {
		simstarter_.openProject();
	}

	private void saveProjectMenuItemActionPerformed() {
		simstarter_.saveProject();
	}

	private void viewProvenanceMenuItemActionPerformed() {
		simstarter_.viewProvenance();
	}

	private void ManageParamsClassesActionPerformed() {
		simstarter_.manageParamsClasses();
	}

	private Menu file_menu_ = new Menu("File");
	private MenuItem new_menu_item_ = new MenuItem("New");
	private MenuItem open_proj_menu_item_ = new MenuItem("Open");
	private MenuItem save_proj_menu_item_ = new MenuItem("Save");

	private Menu view_menu_ = new Menu("View");
	private MenuItem view_provenance_menu_item_ = new MenuItem("Provenance");

	private Menu tools_menu_ = new Menu("Tools");
	private MenuItem mng_params_classes_ = new MenuItem("Manage Params Classes");

	private SimStarter simstarter_;
}
