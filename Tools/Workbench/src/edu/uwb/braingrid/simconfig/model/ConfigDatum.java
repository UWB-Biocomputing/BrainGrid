package edu.uwb.braingrid.simconfig.model;

import java.util.HashMap;
import java.util.Set;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

/**
 * Contains the information relevant to one equivalent XML node, not including
 * its children. A ConfigDatum must have a type, which designates what data it
 * is likely to contain. Any datum type may contain any data, but some functions
 * are designed for use with only certain types.
 *
 * @author Aaron
 */
public class ConfigDatum {

    //Datum types
    public static final int NULL_TYPE = 0;
    public static final int TAB_TYPE = 1;
    public static final int SUBHEAD_TYPE = 2;
    public static final int PARAM_TYPE = 3;

    //Specific attribute tags
    private static final String LABEL_TAG = "label";
    private static final String FILE_CHOOSER_TAG = "fileChooser";

    private int datumType;
    private String tagName;
    private String content;
    private HashMap<String, String> attributes;

    public ConfigDatum(Node element, int datumType) {
        //Check whether datum type is valid
        if (datumType < NULL_TYPE || datumType > PARAM_TYPE) {
            this.datumType = NULL_TYPE;
        } else {
            this.datumType = datumType;
        }

        tagName = element.getNodeName();
        content = element.getTextContent();
        attributes = new HashMap<>();

        NamedNodeMap nodes = element.getAttributes();
        if (nodes != null) {
            Node node;
            for (int i = 0, im = nodes.getLength(); i < im; i++) {
                node = nodes.item(i);
                attributes.put(node.getNodeName(), node.getNodeValue());
            }
        }
    }

    /**
     * Gets this datum's type
     *
     * @return datumType
     */
    public int getDatumType() {
        return datumType;
    }

    /**
     * Gets this datum's tag name
     *
     * @return tagName
     */
    public String getName() {
        return tagName;
    }

    /**
     * Gets this datum's text content
     *
     * @return content
     */
    public String getContent() {
        return content;
    }

    /**
     * Sets the text content to the new value
     *
     * @param newContent
     */
    public void setContent(String newContent) {
        content = newContent;
    }

    /**
     * Gets the content of the attribute "LABEL_TAG" if the datum is not of
     * NULL_TYPE. Returns null otherwise, or if the attribute doesn't exist.
     *
     * @return content of attribute LABEL_TAG
     */
    public String getLabel() {
        String label = null;
        if (datumType != NULL_TYPE) {
            label = attributes.get(LABEL_TAG);
        }
        return label;
    }

    /**
     * Gets the content of the attribute "FILE_CHOOSER_TAG" if the datum is of
     * PARAM_TYPE, and converts it to a boolean which is returned. False is
     * return otherwise, or if the attribute doesn't exist.
     *
     * @return boolean content of attribute FILE_CHOOSER_TAG
     */
    public boolean isFileChooser() {
        boolean isChooser = false;
        if (datumType == PARAM_TYPE) {
            String chooser = attributes.get(FILE_CHOOSER_TAG);
            if (chooser != null) {
                isChooser = Boolean.parseBoolean(chooser);
            }
        }
        return isChooser;
    }

    /**
     * Returns this datum structured as an Element. The Element will look the
     * same as it was when the datum was created, but possibly with the content
     * changed.
     *
     * @param doc Document
     * @return this datum as an element
     */
    public Node getElement(Document doc) {
        Element e = doc.createElement(tagName);
        Set<String> keys = attributes.keySet();

        for (String key : keys) {
            e.setAttribute(key, attributes.get(key));
        }
        e.setTextContent(content);
        return e;
    }
}
