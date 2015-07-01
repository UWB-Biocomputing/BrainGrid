package edu.uwb.braingrid.workbench.project.model;
// NOT CLEANED!!! (Needs JavaDocs, and class header)

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * @author Aaron
 */
public class Datum {

    private String name;
    private HashMap<String, String> attributes;
    private String content;

    public Datum(String name) {
        this.name = name;
        content = "";
    }

    public String getName() {
        return name;
    }

    public String getContent() {
        return content;
    }

    public Datum setContent(String newContent) {
        content = newContent;
        return this;
    }

    public String getAttribute(String key) {
        return attributes.get(key);
    }

    public void setAttribute(String key, String value) {
        attributes.put(key, value);
    }

    public void setAttributes(List<KeyValuePair> attributes) {
        for (int i = 0, im = attributes.size(); i < im; i++) {
            this.attributes.put(attributes.get(i).getKey(),
                    attributes.get(i).getValue());
        }
    }

    public Node getElement(Document doc) {
        Element e = doc.createElement(name);
        Set<String> keys = attributes.keySet();
        for (String key : keys) {
            e.setAttribute(key, attributes.get(key));
        }
        e.setTextContent(content);
        return e;
    }
}
