package edu.uwb.braingrid.workbenchdashboard.userView;

import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import edu.uwb.braingrid.workbenchdashboard.userModel.User;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;

public class UserView extends WorkbenchApp {
	
	private VBox display = new VBox();
	
	private Label userDir = new Label("Main Directory: ");
	private TextField userDirField = new TextField(User.user.getRootDir());
	
	private Label bgRepoDir = new Label("Brain Grid Repos Directory: ");
	private TextField bgRepoDirField = new TextField(User.user.getBrainGridRepoDirectory());
	
	public UserView(Tab tab) {
		super(tab);
		tab.setText("User View");
		this.initAttribues();
	}
	
	private void initAttribues() {
		
		display.getChildren().addAll(userDir, userDirField, bgRepoDir,bgRepoDirField);
	}

	@Override
	public boolean close() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Node getDisplay() {
		// TODO Auto-generated method stub
		return display;
	}
	
}
