package edu.uwb.braingrid.workbench.project.model;
// NOT CLEANED! (Needs Class Header / JavaDocs / Line comments in append f(x))

import java.util.ArrayList;
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
    private List<Datum> data;
    private HashMap<String, String> attributes;

    public ProjectData(String name) {
        this.name = name;
        data = new ArrayList<>();
        attributes = new HashMap<>();
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
        Set<String> keys = attributes.keySet();
        for (String key : keys) {
            e.setAttribute(key, attributes.get(key));
        }
        for (Datum d : data) {
            e.appendChild(d.getElement(doc));
        }
        parent.appendChild(e);
        return e;
    }
}
