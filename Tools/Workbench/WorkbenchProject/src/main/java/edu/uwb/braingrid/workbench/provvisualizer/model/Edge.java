package edu.uwb.braingrid.workbench.provvisualizer.model;

public class Edge {
    private boolean directed = false;
    private String fromNodeId;
    private String toNodeId;
    private String relationship ;

    public Edge() {
    }

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

    public Edge setFromNodeId(String fromNodeId) {
        this.fromNodeId = fromNodeId;
        return this;
    }

    public String getToNodeId() {
        return toNodeId;
    }

    public Edge setToNodeId(String toNodeId) {
        this.toNodeId = toNodeId;
        return this;
    }

    public String getRelationship() {
        return relationship;
    }

    public Edge setRelationship(String relationship) {
        this.relationship = relationship;
        return this;
    }

    public boolean isDirected() {
        return directed;
    }

    public Edge setDirected(boolean directed) {
        this.directed = directed;
        return this;
    }

    @Override
    public Edge clone(){
        return new Edge(fromNodeId,toNodeId,relationship);
    }
}
