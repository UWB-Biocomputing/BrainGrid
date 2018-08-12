package edu.uwb.braingrid.workbenchdashboard.welcome;

import java.util.Timer;
import java.util.TimerTask;
import java.util.logging.Logger;

import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import edu.uwb.braingrid.workbenchdashboard.utils.ThreadManager;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.layout.HBox;
import javafx.util.Duration;

public class Welcome extends WorkbenchApp {
	private static final Logger LOG = Logger.getLogger(Welcome.class.getName());
	
	private HBox display_ = new HBox();
	
	
	public Welcome(Tab tab) {
		
		super(tab);
		LOG.info("new " + getClass().getName());
		Label proof = new Label("Welcome!");
		proof.setAlignment(Pos.CENTER);

		display_.getChildren().add(proof);
		display_.setAlignment(Pos.CENTER);
		super.setTitle("Welcome!");
		
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
