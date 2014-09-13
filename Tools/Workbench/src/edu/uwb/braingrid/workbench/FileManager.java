package edu.uwb.braingrid.workbench;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

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
        String filename = "error";
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
    private final boolean isWindowsSystem;
    private final String folderDelimiter;

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
     * confuses with the poorly named File.pathseperator
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
     * @return True if the file was copied successfully, false if any exceptions
     * occur or if the file located at the source path does not exists
     */
    public static boolean copyFile(Path source, Path target) {
        boolean success = true;
        try {
            File fromFile = source.toFile();
            if (fromFile.exists()) {
                fromFile.mkdirs();
                File toFile = target.toFile();
                if (!toFile.exists()) {
                    toFile.mkdirs();
                    toFile.createNewFile();
                    Files.copy(source, target,
                            StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.COPY_ATTRIBUTES,
                            LinkOption.NOFOLLOW_LINKS);
                }
            } else {
                success = false;
            }
        } catch (Exception e) {
            success = false;
        }
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
}
