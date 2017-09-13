package edu.uwb.braingrid.workbench.provvisualizer.model;

public class Edge {
    private boolean isDirected = false;
    private String fromNodeId;
    private String toNodeId;
    private String relationship ;

    public Edge(String fromNodeId, String toNodeId, String relationship) {
        this.fromNodeId = fromNodeId;
        this.toNodeId = toNodeId;
        this.relationship = relationship;
    }

    public String getEdgeId(){
        return fromNodeId + relationship + toNodeId;
    }

    public String getFromNodeId() {
        return fromNodeId;
    }

    public void setFromNodeId(String fromNodeId) {
        this.fromNodeId = fromNodeId;
    }

    public String getToNodeId() {
        return toNodeId;
    }

    public void setToNodeId(String toNodeId) {
        this.toNodeId = toNodeId;
    }

    public String getRelationship() {
        return relationship;
    }

    public void setRelationship(String relationship) {
        this.relationship = relationship;
    }

    public boolean isDirected() {
        return isDirected;
    }

    public void setDirected(boolean directed) {
        isDirected = directed;
    }
}
