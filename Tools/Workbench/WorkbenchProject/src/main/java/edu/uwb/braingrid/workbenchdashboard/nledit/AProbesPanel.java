package edu.uwb.braingrid.workbenchdashboard.nledit;

import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;

/**
 * The AProbesPanel class handles arrange probed neurons dialog window. The
 * dialog window contains one input field to enter number of probes.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */

public class AProbesPanel extends Pane {
	private Label label = new Label("<html>Number of probes:</html>");;
	public TextField tfield = new TextField();;

	public AProbesPanel() {
		HBox hbox = new HBox(label, tfield);
		getChildren().add(hbox);
	}
}
