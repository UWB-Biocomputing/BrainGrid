package edu.uwb.braingrid.workbench.project.model;
// NOT CLEANED! (Needs Class Header / JavaDocs / Line comments in append f(x))

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * @author Aaron
 */
public class ProjectData {

    private String name;
    private HashMap<String, Datum> data;
    private HashMap<String, String> attributes;

    public ProjectData(String name) {
        this.name = name;
        data = new HashMap<>();
        attributes = new HashMap<>();
    }

    public String getName() {
        return name;
    }

    public Datum addDatum(String name, String content, List<KeyValuePair> attributes) {
        Datum datum = data.get(name);
        if (datum == null) {
            datum = new Datum(name);
            data.put(name, datum);
        }
        if (content != null) {
            datum.setContent(content);
            if (attributes != null) {
                datum.setAttributes(attributes);
            }
        }
        return datum;
    }

    public Datum getDatum(String name) {
        Datum datum;
        if ((datum = data.get(name)) == null) {
            datum = addDatum(name, null, null);
        }
        return datum;
    }

    public String getAttribute(String key) {
        return attributes.get(key);
    }

    public void addAttribute(String key, String value) {
        attributes.put(key, value);
    }

    public void addAtrributes(List<KeyValuePair> attributes) {
        for (int i = 0, im = attributes.size(); i < im; i++) {
            this.attributes.put(attributes.get(i).getKey(),
                    attributes.get(i).getValue());
        }
    }

    public Element appendElement(Document doc, Element parent) {
        Element e = doc.createElement(name);
        Set<String> attrKeys = attributes.keySet();
        for (String key : attrKeys) {
            e.setAttribute(key, attributes.get(key));
        }
        Set<String> datumKeys = data.keySet();
        for (String key : datumKeys) {
            e.appendChild(data.get(key).getElement(doc));
        }
        parent.appendChild(e);
        return e;
    }
}
