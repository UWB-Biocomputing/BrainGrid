package edu.uwb.braingrid.workbenchdashboard.utils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.api.errors.InvalidRemoteException;
import org.eclipse.jgit.api.errors.TransportException;
import org.eclipse.jgit.lib.Ref;

import edu.uwb.braingrid.workbench.provvisualizer.ProvVisGlobal;
import edu.uwb.braingrid.workbenchdashboard.threads.RunUpdateRepo;
import edu.uwb.braingrid.workbenchdashboard.userModel.User;

public class RepoManager {
	public static final String MASTER_BRANCH_NAME = "master";
	private static boolean updatingBranch = false;

	
	public static void updateMaster()  throws InvalidRemoteException, TransportException, GitAPIException, IOException{
		LOG.info("Updating Master Branch");
		updatingBranch = true;
		Git git;
		String masterBranchPath = RepoManager.getMasterBranchDirectory();
		if(Files.exists(Paths.get(masterBranchPath))) {
			LOG.info("Pulling Fresh Repo");
			git = Git.open(new File(masterBranchPath));
            git.pull().call();
		} else {
			LOG.info("Updating Repo");
			git = Git.cloneRepository()
	                .setURI(ProvVisGlobal.BG_REPOSITORY_URI)
	                .setDirectory(new File(masterBranchPath))
	                .call();
		}
		updatingBranch = false;
	}
	
	public static boolean isUpdating() {
		return updatingBranch;
	}
	
	
	
	public static void getMasterBranch() {
		RunUpdateRepo updateRepo = new RunUpdateRepo();
		updateRepo.start();
	}
	
	public static String getMasterBranchDirectory() {
		return User.user.getBrainGridRepoDirectory() + File.separator + MASTER_BRANCH_NAME;
	}
	
	public static List<String> fetchGitBranches()
    {
        Collection<Ref> refs;
        List<String> branches = new ArrayList<String>();
        try {
            refs = Git.lsRemoteRepository()
                    .setHeads(true)
                    .setRemote(ProvVisGlobal.BG_REPOSITORY_URI)
                    .call();
            for (Ref ref : refs) {
                branches.add(ref.getName().substring(ref.getName().lastIndexOf("/")+1, ref.getName().length()));
            }
            Collections.sort(branches);
        } catch (InvalidRemoteException e) {
            LOG.severe(" InvalidRemoteException occured in fetchGitBranches\n" + e.getMessage());
            e.printStackTrace();
        } catch (TransportException e) {
            LOG.severe(" TransportException occurred in fetchGitBranches\n" + e.getMessage());
        } catch (GitAPIException e) {
            LOG.severe(" GitAPIException occurr in fetchGitBranches\n" + e.getMessage());
        }
        return branches;
    }
	
	private static final Logger LOG = Logger.getLogger(RepoManager.class.getName());
}
