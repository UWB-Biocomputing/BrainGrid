package edu.uwb.braingrid.workbench.project.model;

import java.util.ArrayList;
import java.util.List;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * @author Aaron
 */
public class ProjectData {
    
    private String name;
    private List<Datum> data;
    
    public ProjectData(String name) {
        this.name = name;
        data = new ArrayList<>();
    }
    
    public String getName() {
        return name;
    }
    
    public void addDatum(String name, String content, List<KeyValuePair> attributes) {
        Datum datum = new Datum(name);
        datum.setContent(content).setAttributes(attributes);
        data.add(datum);
    }
    
    public Datum getDatum(String name) {
        Datum datum = null;
        for (Datum d : data) {
            if (d.getName().equals(name)) {
                datum = d;
                break;
            }
        }
        return datum;
    }
    
    public Element getElement(Document doc) {
        Element e = doc.createElement(name);
        for (Datum d : data) {
            e.appendChild(d.getElement(doc));
        }
        return e;
    }
}
