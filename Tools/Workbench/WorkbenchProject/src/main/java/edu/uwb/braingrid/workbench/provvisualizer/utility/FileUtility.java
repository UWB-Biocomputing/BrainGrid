package edu.uwb.braingrid.workbench.provvisualizer.utility;

import edu.uwb.braingrid.workbench.provvisualizer.model.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class FileUtility {
    public static final String PREFIX_LOCAL = "local:";
    public static final String FILE_PATH_PREFIX_REGEX = "^(.*/)";
    public static final String FILE_PROTOCOL_REGEX = "^(.*://)";
    public static final String ARTIFACTS_DIR = "artifacts";

    public static String getNodeFileRemoteRelativePath(Node node){
        String relPath = getNodeFileLocalRelativePath(node);
        relPath = relPath.substring(relPath.indexOf("/") + 1);

    if(relPath.charAt(0) == '~'){
            relPath = relPath.substring(2);
        }

        return relPath;
    }

    public static String getNodeFileLocalRelativePath(Node node){
        String nodeId = node.getId();

        if(nodeId.contains(PREFIX_LOCAL)){
            return nodeId.replaceFirst(PREFIX_LOCAL,"");
        }
        else {
            return nodeId.replaceFirst(FILE_PROTOCOL_REGEX,"");
        }
    }

    public static String getNodeFileLocalAbsolutePath(Node node){
        return System.getProperty("user.dir") + File.separator + ARTIFACTS_DIR + File.separator +
                getNodeFileLocalRelativePath(node);
    }

    public static List<String> fileToLines(String filename) {
        List<String> lines = new LinkedList<String>();
        String line = "";
        BufferedReader in = null;
        try {
            in = new BufferedReader(new FileReader(filename));
            while ((line = in.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (in != null) {
                try {
                    in.close();
                } catch (IOException e) {
                    // ignore ... any errors should already have been
                    // reported via an IOException from the final flush.
                }
            }
        }
        return lines;
    }
}
