package edu.uwb.braingrid.workbench.provvisualizer.utility;

import com.jcraft.jsch.*;
import edu.uwb.braingrid.workbench.provvisualizer.model.AuthenticationInfo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ConnectionUtility {
    public static final int SFTP_PORT = 22;

    public static boolean downloadFileViaSftp(String remoteFilePath,
                                              String localFilePath, AuthenticationInfo authenticationInfo){
        return downloadFileViaSftp(remoteFilePath,localFilePath,authenticationInfo.getHostname(),
                authenticationInfo.getUsername(), authenticationInfo.getPassword().toCharArray());
    }

    public static boolean downloadFileViaSftp(String remoteFilePath,
                                String localFilePath, String hostname, String username,
                                char[] password){
        boolean success = true;
        Session session = null;

        try {
            JSch jsch = new JSch();
            // apply user info to connection attempt
            session = jsch.getSession(username, hostname, SFTP_PORT);
            session.setPassword(String.valueOf(password));
            // generic setting up
            java.util.Properties config = new java.util.Properties();
            config.put("StrictHostKeyChecking", "no");
            session.setConfig(config);
            session.connect();
            // download file
            ChannelSftp channelSftp = (ChannelSftp) session.openChannel("sftp");
            channelSftp.connect();

            String localFileDirectory = localFilePath.substring(0,localFilePath.lastIndexOf("/"));
            Path path = Paths.get(localFileDirectory);
            if(!Files.exists(path)){
                Files.createDirectories(path);
            }
            channelSftp.get(remoteFilePath, localFilePath);
            channelSftp.disconnect();
        } catch (JSchException | SftpException e) {
            success = false;
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (session != null && session.isConnected()) {
                session.disconnect();
            }
        }
        return success;
    }
}
