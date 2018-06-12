package edu.uwb.braingrid.workbenchdashboard.provis;

import java.io.IOException;

import edu.uwb.braingrid.workbenchdashboard.WorkbenchApp;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.TextArea;

public class ProVisOld extends WorkbenchApp {

	Parent root_;

	public ProVisOld() {
		 //Parent root;
	        try {
	            root_ = FXMLLoader.load(getClass().getResource("/provvisualizer/view/ProvenanceVisualizerView.fxml"));
	            
	        } catch (IOException e) {
	        	root_ = new TextArea();
	        	((TextArea) root_).setText(e.toString());
	            //e.printStackTrace();
	        }
	}
	
	@Override
	public boolean close() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public Node getDisplay() {
		// TODO Auto-generated method stub
		return root_;
	}

}
