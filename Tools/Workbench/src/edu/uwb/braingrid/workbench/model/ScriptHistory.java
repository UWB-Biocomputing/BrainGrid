package edu.uwb.braingrid.workbench.model;
// NOT CLEANED

/**
 * FIX THIS!!! (Needs JavaDocs) ALL CLASSES NEED A HEADER TOO
 *
 * @author Aaron
 */
public class ScriptHistory {

    private String startedAt;
    private String completedAt;
    private boolean outputAnalyzed;
    private boolean ran;
    private String filename;
    private int version;

    public ScriptHistory() {
        startedAt = null;
        completedAt = null;
        outputAnalyzed = false;
        ran = false;
        filename = null;
        version = 0;
    }

    public String getStartedAt() {
        return startedAt;
    }

    public String getCompletedAt() {
        return completedAt;
    }

    public boolean getOutputAnalyzed() {
        return outputAnalyzed;
    }

    public boolean getRan() {
        return ran;
    }

    public String getFilename() {
        return filename;
    }

    public int getVersion() {
        return version;
    }

    public void setStartedAt(String startedAt) {
        this.startedAt = startedAt;
    }

    public void setCompletedAt(String completedAt) {
        this.completedAt = completedAt;
    }

    public void setOutputAnalyzed(boolean outputAnalyzed) {
        this.outputAnalyzed = outputAnalyzed;
    }

    public void setRan(boolean ran) {
        this.ran = ran;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public void setVersion(int version) {
        this.version = version;
    }

    public void incrementVersion() {
        version++;
    }
    
    /**
     * Provides the version number of the next script that will be added to the
     * project. This is a convenience function, the version number is determined
     * based on the current script version
     *
     * @return The version number of the next script that will be added to the
     * project when another script is generated.
     */
    public String getNextScriptVersion() {

        String scriptVerNum;
        try {
            scriptVerNum = String.valueOf(version + 1);
        } catch (NumberFormatException e) {
            scriptVerNum = String.valueOf(version = 0);
        }
        return scriptVerNum;
    }
}
