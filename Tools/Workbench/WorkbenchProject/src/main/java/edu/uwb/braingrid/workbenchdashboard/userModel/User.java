/**
 * 
 */
package edu.uwb.braingrid.workbenchdashboard.userModel;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import edu.uwb.braingrid.workbench.provvisualizer.ProvVisGlobal;

/**
 * @author Max
 *
 */
public class User {
	
	public static User user = new User();

	
	private User() {
		
	}
	
	private static void initUser() {
		LOG.info("Initializing User Data");
		User.user.setRootDir(System.getProperty("user.dir"));
		User.user.setBrainGridRepoDirectory(User.user.getRootDir() + File.separator + ProvVisGlobal.BG_REPOSITORY_LOCAL);
		save();
	}
	
	public static boolean load() {
		LOG.info("Loading User Information");
		ObjectMapper mapper = new ObjectMapper();
		File file = new File(USER_DATA_PATH);
		JsonNode json = null;
		if(file.exists()) {
			try {
				json = mapper.readTree(file);
				User.user = mapper.readValue(new File(User.USER_DATA_PATH), User.class);
			} catch (IOException e) {
				LOG.severe(e.getMessage());
				return false;
			}
		} else {
			LOG.info("No User Info Found");
			initUser();
			return load();
		}
		LOG.info("User info loaded: "  + json.toString());
		return true;
	}
	
	public static boolean save() {
		LOG.info("Saving User Data");
		ObjectMapper mapper = new ObjectMapper();
		
		try {
			mapper.writeValue(new File(User.USER_DATA_PATH), User.user);
		} catch (IOException e) {
			LOG.severe(e.getMessage());
			return false;
		}
		return true;
	}
	
	public String getRootDir() {
		return this.rootDir;
	}

	public void setRootDir(String rootDir) {
		this.rootDir = rootDir;
	}

	public String getBrainGridRepoDirectory() {
		return brainGridRepoDirectory;
	}

	public void setBrainGridRepoDirectory(String brainGridRepoDirectory) {
		this.brainGridRepoDirectory = brainGridRepoDirectory;
	}
	
	

	private String brainGridRepoDirectory = "";
	private String rootDir = "";
	private final static Logger LOG = Logger.getLogger(User.class.getName());
	private static final String USER_DATA_PATH = "./user.json";
}
