package edu.uwb.braingrid.workbench;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import org.apache.commons.io.FileUtils;

/**
 * Handles all file operations for the workbench. The purpose behind this
 * singleton manager is two-fold: 1. Provides conditional support between
 * operating system file paths (this is not possible in static manner without
 * querying the system properties repeatedly); 2. Provides a separation of
 * concerns in terms of workbench configuration for file operations (such as
 * folder hierarchical relationships and names); 3. Provides workbench specific
 * robustness in terms of file operations such as copy.
 *
 * Note: As mentioned in the overview above, this is a singleton class. Many
 * methods are non-static. In order to use the file manager, obtain a reference
 * by calling FileManager.getFileManager.
 *
 * @author Del Davis
 */
public final class FileManager {

    private static FileManager instance = null;

    /**
     * Utility function that returns the last name of a specified path string
     * containing parent folders
     *
     * @param longFilename - path string containing parent folders
     * @return - the last name of the file, including the base name and
     * extension, but no parent folders
     */
    public static String getSimpleFilename(String longFilename) {
        String filename = longFilename;
        if (longFilename.contains("\\")) {
            int lastBackslash = longFilename.lastIndexOf('\\');
            if (lastBackslash != longFilename.length() - 1) {
                filename = longFilename.substring(lastBackslash + 1);
            }
        } else if (longFilename.contains("/")) {
            int lastForwardslash = longFilename.lastIndexOf('/');
            if (lastForwardslash != longFilename.length() - 1) {
                filename = longFilename.substring(lastForwardslash + 1);
            }
        } else {
            filename = longFilename;
        }
        return filename;
    }

    static void copyFolder(String src, String dst) throws IOException {
        FileUtils.copyDirectory(new File(src), new File(dst));
    }

    public String[] getNeuronListFilenames(String projectName) throws IOException {
        String[] filenames = null;
        File[] files;
        String folder = getSimConfigDirectoryPath(projectName, false)
                + neuronListFolderName + folderDelimiter;
        File folderFile = Paths.get(folder).toFile();
        if (folderFile.isDirectory()) {
            files = folderFile.listFiles();
            filenames = new String[files.length];
            for (int i = 0, im = files.length; i < im; i++) {
                filenames[i] = files[i].getCanonicalPath();
            }
        }
        return filenames;
    }
    private final boolean isWindowsSystem;
    private final String folderDelimiter;
    private String projectsFolderName = "projects";
    private String configFilesFolderName = "configfiles";
    private String neuronListFolderName = "NList";

    /**
     * This is here to make sure that classes from other packages cannot
     * instantiate the file manager. This is to ensure that only one file
     * manager is present in the workbench for the life of the workbench control
     * thread.
     */
    private FileManager() {
        String osName = System.getProperty("os.name").toLowerCase();
        isWindowsSystem = osName.startsWith("windows");
        folderDelimiter = isWindowsSystem ? "\\" : "/";
    }

    /**
     * Gets or instantiates this file manager.
     *
     * Note: This is a singleton class with lazy instantiation.
     *
     * @return The file manager for the workbench
     */
    public static FileManager getFileManager() {
        if (instance == null) {
            instance = new FileManager();
        }
        return instance;
    }

    /**
     * Provides the operating system dependent folder delimiter, not to be
     * confused with the poorly named File.pathseperator
     *
     * @return The string that delimits folders from parent folders on the
     * operating system where the workbench was invoked
     */
    public String getFolderDelimiter() {
        return folderDelimiter;
    }

    /**
     * Copies a file from the source path to the target path. If the parent
     * directories required in the tree to target are not present, they will be
     * created.
     *
     * Note: Errors are hidden and may occur do to various exceptions. No
     * security privileges are requested. This is intentional.
     *
     * @param source - The path to the file to be copied
     * @param target - The path to the copy of the source
     * @return True if the file was copied successfully, false if the file
     * represented by the source path does not existing
     * @throws java.io.IOException
     */
    public static boolean copyFile(Path source, Path target) throws IOException {
        boolean success = true;
        File fromFile = source.toFile();
        if (fromFile.exists()) {
            File toFile = target.toFile();
            if (!toFile.exists()) {
                target.getParent().toFile().mkdirs();
                toFile.createNewFile();
            }
        } else {
            success = false;
        }
        Files.copy(source, target,
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.COPY_ATTRIBUTES,
                LinkOption.NOFOLLOW_LINKS);
        return success;
    }

    /**
     * Returns the canonical form of the current working directory
     *
     * @return A string representation of the system dependent unique canonical
     * form of the current working directory
     * @throws java.io.IOException
     */
    public static String getCanonicalWorkingDirectory() throws IOException {
        return Paths.get("").toFile().getCanonicalPath();
    }

    public String getCanonicalProjectsDirectory() throws IOException {
        return getCanonicalWorkingDirectory() + folderDelimiter
                + projectsFolderName;
    }

    /**
     * Indicates whether or not the operating system is some version of
     * Microsoft Windows
     *
     * @return True if the system is a windows-based system, otherwise false. As
     * of the implementing of this workbench, false can be interpreted as a
     * Posix-based system.
     */
    public boolean isWindowsSystem() {
        return isWindowsSystem;
    }

    public String getSimConfigFilePath(String projectName, String filename, boolean mkdirs) throws IOException {
        return getSimConfigDirectoryPath(projectName, mkdirs) + filename;
    }

    public String getNeuronListFilePath(String projectName, String filename, boolean mkdirs) throws IOException {
        String folder = getSimConfigDirectoryPath(projectName, mkdirs)
                + neuronListFolderName + folderDelimiter;
        if (mkdirs) {
            new File(folder).mkdirs();
        }
        return folder + filename;
    }

    public String getSimConfigDirectoryPath(String projectName, boolean mkdirs) throws IOException {
        String folder = getProjectDirectory(projectName, mkdirs)
                + configFilesFolderName + folderDelimiter;
        if (mkdirs) {
            new File(folder).mkdirs();
        }
        return folder;
    }

    public String getProjectDirectory(String projectName, boolean mkdirs) throws IOException {
        String directory = getCanonicalProjectsDirectory() + folderDelimiter
                + projectName + folderDelimiter;
        if (mkdirs) {
            new File(directory).mkdirs();
        }
        return directory;
    }

    public String getUserDir() {
        return System.getProperty("user.home") + folderDelimiter;
    }
}
