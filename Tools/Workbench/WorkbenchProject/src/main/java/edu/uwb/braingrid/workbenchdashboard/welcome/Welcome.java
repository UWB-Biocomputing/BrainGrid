package edu.uwb.braingrid.workbenchdashboard.welcome;

import edu.uwb.braingrid.workbenchdashboard.TabToPane;
import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;

public class Welcome extends WorkbenchApp {
	private HBox display_ = new HBox();
	public Welcome() {
		Label proof = new Label("Welcome!");
		proof.setAlignment(Pos.CENTER);
		display_.getChildren().add(proof);
		display_.setAlignment(Pos.CENTER);
	}

	@Override
	public boolean close() {
		return true;
	}

	@Override
	public Node getDisplay() {
		// TODO Auto-generated method stub
		return display_;
	}

}