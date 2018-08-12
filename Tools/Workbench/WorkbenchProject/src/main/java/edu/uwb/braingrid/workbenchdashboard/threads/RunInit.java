package edu.uwb.braingrid.workbenchdashboard.threads;

import java.util.logging.Logger;
import edu.uwb.braingrid.workbenchdashboard.utils.Init;
import edu.uwb.braingrid.workbenchdashboard.utils.ThreadManager;

public class RunInit extends Thread implements Runnable {

	public RunInit() {

	}
	
	public void run() {
		ThreadManager.addThread("Init");
		Init.init();
		ThreadManager.removeThread("Init");
	}
	
	private static final Logger LOG = Logger.getLogger(RunUpdateRepo.class.getName());
}
