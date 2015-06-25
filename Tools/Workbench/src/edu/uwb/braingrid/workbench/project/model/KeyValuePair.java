package edu.uwb.braingrid.workbench.project.model;

/**
 *
 * @author Aaron
 */
public class KeyValuePair {
    
    private String key;
    private String value;
    
    public KeyValuePair(String key, String value) {
        this.key = key;
        this.value = value;
    }
    
    public String getKey() {
        return key;
    }
    
    public String getValue() {
        return value;
    }
    
    public void setValue(String newVal) {
        value = newVal;
    }
}
