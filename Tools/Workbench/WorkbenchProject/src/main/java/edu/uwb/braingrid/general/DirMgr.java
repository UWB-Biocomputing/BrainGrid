package edu.uwb.braingrid.general;

import java.io.File;
import java.util.logging.Logger;

import edu.uwb.braingrid.workbench.provvisualizer.ProvVisGlobal;

public class DirMgr {
	private static Logger LOG = Logger.getLogger(DirMgr.class.getName());
	
	public static String getRootPath() {
		LOG.info("Root Path: " + System.getProperty("user.dir"));
		return System.getProperty("user.dir");
	}
	
	public static String getBrainGridRepoDirectory() {
		
		String bgReposPath = DirMgr.getRootPath() + File.separator + ProvVisGlobal.BG_REPOSITORY_LOCAL;
	
		LOG.info("BrainGrid Repo Path: " + bgReposPath);
		return bgReposPath;
	}
	
}
