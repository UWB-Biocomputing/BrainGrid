package edu.uwb.braingrid.workbenchdashboard;

import edu.uwb.braingrid.workbenchdashboard.utils.ThreadManager;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.util.Duration;

public class WorkbenchStatusBar extends HBox {
	
	public WorkbenchStatusBar() {
		WorkbenchStatusBar.updateUpdateMessage();
		this.setAlignment(Pos.BOTTOM_RIGHT);
		statusLabel.setAlignment(Pos.BOTTOM_RIGHT);
		this.getChildren().add(statusLabel);
		
		
		Timeline timeline = new Timeline(
			    new KeyFrame(Duration.seconds(5), e -> {
			    	WorkbenchStatusBar.updateUpdateMessage();
			    })
		);
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.play();
		getStyleClass().add("updates-bar");
	}


	public static void updateUpdateMessage() {
		statusLabel.setText(getUpdateMessage());
	}
	
	private static String getUpdateMessage() {
		String out = ThreadManager.getStatus();
		if(ThreadManager.getProcessesRunning() > 0) {
			out += ", " + ThreadManager.getProcessesRunning() + " Process Running";
		}
		return out;
	}
	
	private static Label statusLabel = new Label(ThreadManager.getStatus());
}
