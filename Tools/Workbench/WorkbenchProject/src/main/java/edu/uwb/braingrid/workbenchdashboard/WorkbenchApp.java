package edu.uwb.braingrid.workbenchdashboard;

import javafx.scene.Node;
import javafx.scene.control.Tab;

public abstract class WorkbenchApp {
	private Tab tab_;
	public WorkbenchApp(Tab tab) {
		tab_ = tab;
	}
	public abstract boolean close();
	public abstract Node getDisplay();
	public void setTitle(String title) {
		tab_.setText(title);
	}
}
