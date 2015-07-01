package edu.uwb.braingrid.workbench.model;

/**
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
}
