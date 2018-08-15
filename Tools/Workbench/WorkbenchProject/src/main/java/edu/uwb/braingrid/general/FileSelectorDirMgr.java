package edu.uwb.braingrid.general;

import java.io.File;
import java.util.ArrayList;

import edu.uwb.braingrid.workbenchdashboard.userModel.User;

public class FileSelectorDirMgr {
	
	ArrayList<File> dirs = new ArrayList<File>();
	
	public FileSelectorDirMgr() {
		
	}
	
	public File getLastDir() {
		if(dirs.isEmpty()) {
			return getDefault();
		}
		return dirs.get(dirs.size() - 1);
	}
	
	public File getDir(int index) {
		if(dirs.isEmpty() || index >= dirs.size()) {
			return getDefault();
		}
		return dirs.get(index);
	}
	
	public void add(File newdir) {
		dirs.add(newdir);
	}
	
	public File getDefault() {
		return new File(User.user.getRootDir());
	}
	

}
