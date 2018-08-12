package edu.uwb.braingrid.workbenchdashboard.utils;

import java.util.ArrayList;
import java.util.logging.Logger;

public class ThreadManager {
	private static ArrayList<String> activeThreads = new ArrayList<String>();
	private static final String NO_PROCESSES = "No Processes";
	
	public static void addThread(String thread) {
		LOG.info("Watching Thread: " + thread.toString());
		activeThreads.add(thread);
	}
	
	public static boolean removeThread(String thread) {
		LOG.info("Removing Thread: " + thread.toString());
		activeThreads.remove(thread);
		return false;
	}
	
	public static String getStatus() {
		if(activeThreads.size() > 0) {
			return activeThreads.get(0);
		}
		return NO_PROCESSES;
	}
	
	public static int getProcessesRunning() {
		return activeThreads.size();
	}

	private static final Logger LOG = Logger.getLogger(ThreadManager.class.getName());
}
