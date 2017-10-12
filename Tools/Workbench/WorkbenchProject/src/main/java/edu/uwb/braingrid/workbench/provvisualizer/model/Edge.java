package edu.uwb.braingrid.workbench.provvisualizer.model;

import java.util.HashMap;

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

    public String getShortRelationship() {
        int lastInd = relationship.lastIndexOf('#');

        if(lastInd != -1){
            return relationship.substring(lastInd + 1);
        }
        else{
            return relationship;
        }
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

    public boolean equals(Edge edge){
        return this.getEdgeId().equals(edge.getEdgeId());
    }

    public int hashCode(){ return this.getEdgeId().hashCode();}

    /**
     * Determine if a point(x,y) is on this edge's rectangular buffer area.
     * 1. Calculate the slope of edge and the slope of the lines perpendicular to the edge.
     * 2. Using the coordinate of the connected nodes, the slope and the width of the buffer area to find the coordinate
     *    of the four corners of the buffer area.
     * 3. Use the slope-intercept form (y = mx +b) to find the range of y-intercepts(b).
     * 4. If the lines passing through point(x,y) with the same slopes have y-intercepts within the above ranges, the point
     *    is inside the buffer area, i.e., return true.
     * @param x
     * @param y
     * @param zoomRatio
     * @return true if the point is in the buffer area.
     */
    public boolean isPointOnEdge(HashMap<String,Node> nodes, double x, double y, double zoomRatio){
        double bufferLength = 5;
        Node fromNode = nodes.get(fromNodeId);
        Node toNode = nodes.get(toNodeId);
        double[] fromNodePoint = new double[]{ fromNode.getX() + fromNode.getSize()/zoomRatio /2, fromNode.getY() + fromNode.getSize()/zoomRatio/2 };
        double[] toNodePoint = new double[]{ toNode.getX() + toNode.getSize()/zoomRatio / 2, toNode.getY() + toNode.getSize()/zoomRatio/2};

        double edgeSlope = GraphUtility.calculateSlope(fromNodePoint,toNodePoint);
        double edgeSlopeAngle = Math.atan(edgeSlope);
        double edgeRightAngleSlope = -1 / edgeSlope;


        //only need diagonal points
        //double[] fromNodePoint1 = GraphUtility.findPointWithAngleDistance(fromNodePoint, edgeSlopeAngle + Math.PI/2, bufferLength);
        double[] fromNodePoint2 = GraphUtility.findPointWithAngleDistance(fromNodePoint, edgeSlopeAngle - Math.PI/2, bufferLength);
        double[] toNodePoint1 = GraphUtility.findPointWithAngleDistance(toNodePoint, edgeSlopeAngle + Math.PI/2, bufferLength);
        //double[] toNodePoint2 = GraphUtility.findPointWithAngleDistance(toNodePoint, edgeSlopeAngle - Math.PI/2, bufferLength);

        double[] yInterceptEdgeSlope = new double[]{fromNodePoint2[1] - edgeSlope * fromNodePoint2[0], toNodePoint1[1] - edgeSlope * toNodePoint1[0]};
        double[] yInterceptEdgeRightAngleSlope = new double[]{fromNodePoint2[1] - edgeRightAngleSlope * fromNodePoint2[0], toNodePoint1[1] - edgeRightAngleSlope * toNodePoint1[0], };
        if(yInterceptEdgeSlope[0] > yInterceptEdgeSlope[1]){
            double temp = yInterceptEdgeSlope[0];
            yInterceptEdgeSlope[0] = yInterceptEdgeSlope[1];
            yInterceptEdgeSlope[1] = temp;
        }

        if(yInterceptEdgeRightAngleSlope[0] > yInterceptEdgeRightAngleSlope[1]){
            double temp = yInterceptEdgeRightAngleSlope[0];
            yInterceptEdgeRightAngleSlope[0] = yInterceptEdgeRightAngleSlope[1];
            yInterceptEdgeRightAngleSlope[1] = temp;
        }

        double pointYInterceptEdgeSlope = y - edgeSlope * x;
        double pointYInterceptEdgeRightAngleSlope = y - edgeRightAngleSlope * x;

        if(pointYInterceptEdgeSlope >= yInterceptEdgeSlope[0] && pointYInterceptEdgeSlope <= yInterceptEdgeSlope[1] &&
                pointYInterceptEdgeRightAngleSlope >= yInterceptEdgeRightAngleSlope[0] && pointYInterceptEdgeRightAngleSlope <= yInterceptEdgeRightAngleSlope[1]){
            return true;
        }

        return false;
    }
}
