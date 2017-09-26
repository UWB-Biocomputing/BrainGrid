package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.animation.AnimationTimer;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Graph {
    public static final double LABEL_FONT_SIZE = 20;

    private HashMap<String, Node> nodes = new HashMap<>();
    private HashMap<String, Edge> edges = new HashMap<>();
    private boolean showAllNodeIds = false;
    private boolean showAllRelationships = false;
    private HashSet<Node> dispNodeIds = new HashSet<>();
    private HashSet<Edge> dispRelationships = new HashSet<>();

    private Node mouseOnNode ;

    private double c1 = 2;      //default value = 2;
    private double c2 = 1;      //default value = 1;
    private double c3 = 5000;   //default value = 1;
    private double c4 = 0.1;    //default value = 0.1;

    public Graph(){
    }

    public double getC3(){
        return c3;
    }

    public void setC3(double c3){
        this.c3 = c3;
    }

    public void addNode(Node node){
        this.nodes.put(node.getId(),node);
    }

    public void addNodes(Node... nodes){
        for(Node node : nodes) {
            this.nodes.put(node.getId(),node);
        }
    }

    public boolean isNodeAdded(String nodeId){
        if(this.nodes.containsKey(nodeId)) {
            return true;
        }
        else{
            return false;
        }
    }

    public Node getNode(String nodeId){
        if(this.nodes.containsKey(nodeId)) {
            return this.nodes.get(nodeId);
        }
        else{
            return null;
        }
    }

    public void addEdge(Edge edge){
        this.edges.put(edge.getEdgeId(),edge);
    }

    public void addEdges(Edge... edges){
        for(Edge edge : edges) {
            this.edges.put(edge.getEdgeId(),edge);
        }
    }

    public Edge getEdge(String edgeId){
        return this.edges.get(edgeId);
    }

    public boolean isEdgeAdded(String edgeId){
        if(this.edges.containsKey(edgeId)) {
            return true;
        }
        else{
            return false;
        }
    }

    /**
     * Using Force-directed graph layout algorithm to optimize the node positions
     * @param draggedNode
     */
    public void moveNodes(Node draggedNode) {
        for (Node node1: nodes.values()) {
            // loop over all node pairs and calculate the net force
            double[] netForce = new double[]{0, 0};
            for (Node node2: nodes.values()) {
                if (!node1.equals(node2)) {
                    if (node1.isConnected(node2)) {
                        double[] repellingForce = repellingForce(node1,node2);
                        double[] attractiveForce = attractiveForce(node1,node2);
                        // if connected
                        netForce[0] = netForce[0]+repellingForce[0]+attractiveForce[0];
                        netForce[1] = netForce[1]+repellingForce[1]+attractiveForce[1];
                    } else {
                        // if not connected
                        double[] repellingForce = repellingForce(node1,node2);
                        netForce[0] = netForce[0]+repellingForce[0];
                        netForce[1] = netForce[1]+repellingForce[1];
                    }
                }
            }
            //apply the force to the node
            node1.setX(node1.getX() + c4 * netForce[0]);
            node1.setY(node1.getY() + c4 * netForce[1]);
        }
    }

    /**
     * Computes the vector of the attractive force between two node.
     * @param from ID of the first node
     * @param to ID of the second node
     * @return force vector
     */
    public double[] attractiveForce(Node from, Node to){
        double [] vec;
        double distance=getDistance(from, to);
        vec=computeNormalizedVector(from, to);//*distance;
        double factor = c1*Math.log(distance/c2);
        vec[0]=vec[0]*factor;
        vec[1]=vec[1]*factor;
        return vec;
    }

    /**
     * Computes the vector of the repelling force between two node.
     * @param from ID of the first node
     * @param to ID of the second node
     * @return force vector
     */
    public double[] repellingForce(Node from, Node to){
        double [] vec;
        double distance=getDistance(from, to);
        vec=computeNormalizedVector(from, to);//*distance;
        double factor = -c3/Math.pow(distance, 2);
        vec[0]=vec[0]*factor;
        vec[1]=vec[1]*factor;
        return vec;
    }

    /**
     * Computes the connecting vector between node1 and node2.
     * @param node1 ID of the first node
     * @param node2 ID of the second node
     * @return the connecting vector between node1 and node2
     */
    public double[] computeNormalizedVector(Node node1, Node node2){
        double vectorX = node2.getX()-node1.getX();
        double vectorY = node2.getY()-node1.getY();
        double length=Math.sqrt(Math.pow(Math.abs(vectorX),2)+Math.pow(Math.abs(vectorY),2));
        return new double[]{vectorX/length, vectorY/length};
    }

    /**
     * Computes the euclidean distance between two given nodes.
     * @param node1 first node
     * @param node2 second node
     * @return euclidean distance between the nodes
     */
    public double getDistance(Node node1, Node node2){
        return Math.sqrt(Math.pow(Math.abs(node1.getX()-node2.getX()),2)
                +Math.pow(Math.abs(node1.getY()-node2.getY()),2));
    }

    public void drawOnCanvas(Canvas canvas, double[] displayWindowLocation, double[] displayWindowSize, double zoomRatio){
        GraphicsContext gc = canvas.getGraphicsContext2D();
        //draw background
        gc.setFill(Color.BEIGE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.setFont(Font.font(LABEL_FONT_SIZE));

        for(Edge edge : edges.values()){
            Node fromNode = nodes.get(edge.getFromNodeId());
            Node toNode = nodes.get(edge.getToNodeId());
            if(fromNode != null && toNode != null) {
                if (fromNode.isInDisplayWindow(displayWindowLocation, displayWindowSize) ||
                        toNode.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                    double[] transformedFromNodeXY = transformToRelativeXY(fromNode.getX(), fromNode.getY(), displayWindowLocation, zoomRatio);
                    double[] transformedToNodeXY = transformToRelativeXY(toNode.getX(), toNode.getY(), displayWindowLocation, zoomRatio);
                    gc.strokeLine(transformedFromNodeXY[0] + fromNode.getSize() / 2, transformedFromNodeXY[1] + fromNode.getSize() / 2,
                            transformedToNodeXY[0] + toNode.getSize() / 2, transformedToNodeXY[1] + toNode.getSize() / 2);
                }
            }
        }

        for(Node node : nodes.values()){
            if(node.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                double[] transformedNodeXY = transformToRelativeXY(node.getX(),node.getY(), displayWindowLocation, zoomRatio);
                double nodeSize = node.getSize();

                gc.setFill(node.getColor());

                if(node.getShape() == Node.NodeShape.CIRCLE) {
                    if(node == mouseOnNode){
                        gc.fillOval(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1);
                        gc.strokeOval(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1);
                    }
                    else{
                        gc.fillOval(transformedNodeXY[0], transformedNodeXY[1], nodeSize, nodeSize);
                    }
                }
                else if(node.getShape() == Node.NodeShape.SQUARE){
                    if(node == mouseOnNode){
                        gc.fillRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1);
                        gc.strokeRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1);
                    }
                    else {
                        gc.fillRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize, nodeSize);
                    }
                }
                else if(node.getShape() == Node.NodeShape.ROUNDED_SQUARE){
                    if(node == mouseOnNode){
                        gc.fillRoundRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1, nodeSize/10, nodeSize/10);
                        gc.strokeRoundRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize * 1.1, nodeSize * 1.1, nodeSize/10, nodeSize/10);
                    }
                    else{
                        gc.fillRoundRect(transformedNodeXY[0], transformedNodeXY[1], nodeSize, nodeSize, nodeSize/10, nodeSize/10);
                    }
                }
            }
        }

        if(showAllNodeIds){
            showAllNodeIds(gc, displayWindowLocation, displayWindowSize, zoomRatio);
        }
        else {
            for (Node node : dispNodeIds) {
                if (node.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                    double[] transformedNodeXY = transformToRelativeXY(node.getX(), node.getY(), displayWindowLocation, zoomRatio);

                    gc.setFill(Color.BLACK);
                    gc.fillText(node.getId(), transformedNodeXY[0], transformedNodeXY[1] + node.getSize() + LABEL_FONT_SIZE);
                }
            }
        }

        if(showAllRelationships){
            showAllRelationships(gc, displayWindowLocation, displayWindowSize, zoomRatio);
        }
        else {
            for (Edge edge : dispRelationships) {
                Node fromNode = nodes.get(edge.getFromNodeId());
                Node toNode = nodes.get(edge.getToNodeId());
                if(fromNode != null && toNode != null) {
                    if (fromNode.isInDisplayWindow(displayWindowLocation, displayWindowSize) ||
                            toNode.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                        double[] transformedFromNodeXY = transformToRelativeXY(fromNode.getX(), fromNode.getY(), displayWindowLocation, zoomRatio);
                        double[] transformedToNodeXY = transformToRelativeXY(toNode.getX(), toNode.getY(), displayWindowLocation, zoomRatio);
                        double[] transformedMidXY = new double[]{(transformedFromNodeXY[0] + transformedToNodeXY[0]) / 2.0,
                                (transformedFromNodeXY[1] + transformedToNodeXY[1]) / 2.0};

                        gc.setFill(Color.BROWN);
                        gc.fillText(edge.getRelationship(), transformedMidXY[0], transformedMidXY[1] + LABEL_FONT_SIZE);
                    }
                }
            }
        }

        if(mouseOnNode != null){
            showNodeId(mouseOnNode, canvas, displayWindowLocation,zoomRatio);
        }
    }

    private void showNodeId(Node node, Canvas canvas, double[] displayWindowLocation, double zoomRatio){
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double[] transformedNodeXY = transformToRelativeXY(node.getX(),node.getY(), displayWindowLocation, zoomRatio);
        gc.setFill(Color.BLACK);
        gc.fillText(node.getId(), transformedNodeXY[0], transformedNodeXY[1] + node.getSize() + LABEL_FONT_SIZE);
    }

    private double[] transformToRelativeXY(double x, double y, double[] displayWindowLocation, double zoomRatio){
        double[] xy = new double[2];
        xy[0] = (x - displayWindowLocation[0]) * zoomRatio;
        xy[1] = (y - displayWindowLocation[1]) * zoomRatio;

        return xy;
    }

    public Node getSelectedNode(double x, double y, double zoomRatio) {
        for(Node node : nodes.values()){
            if(node.isPointOnNode(x, y, zoomRatio)){
                return node;
            }
        }
        return null;
    }

    public void setNeighbors(){
        for (Edge edge: edges.values()){
            Node fromNode = nodes.get(edge.getFromNodeId());
            Node toNode = nodes.get(edge.getToNodeId());
            if(fromNode != null && toNode !=null) {
                fromNode.getNeighborNodes().add(toNode);
                toNode.getNeighborNodes().add(fromNode);
            }
        }
    }

    public void showAllNodeIds(GraphicsContext gc, double[] displayWindowLocation, double[] displayWindowSize, double zoomRatio) {
        for(Node node : nodes.values()){
            if(node.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                double[] transformedNodeXY = transformToRelativeXY(node.getX(),node.getY(), displayWindowLocation, zoomRatio);

                gc.setFill(Color.BLACK);
                gc.fillText(node.getId(), transformedNodeXY[0], transformedNodeXY[1] + node.getSize() + LABEL_FONT_SIZE);
            }
        }
    }

    public void showAllRelationships(GraphicsContext gc, double[] displayWindowLocation, double[] displayWindowSize, double zoomRatio) {
        for(Edge edge : edges.values()){
            Node fromNode = nodes.get(edge.getFromNodeId());
            Node toNode = nodes.get(edge.getToNodeId());
            if(fromNode != null && toNode != null) {
                if (fromNode.isInDisplayWindow(displayWindowLocation, displayWindowSize) || toNode.isInDisplayWindow(displayWindowLocation, displayWindowSize)) {
                    double[] transformedFromNodeXY = transformToRelativeXY(fromNode.getX(), fromNode.getY(), displayWindowLocation, zoomRatio);
                    double[] transformedToNodeXY = transformToRelativeXY(toNode.getX(), toNode.getY(), displayWindowLocation, zoomRatio);
                    double[] transformedMidXY = new double[]{(transformedFromNodeXY[0] + transformedToNodeXY[0]) / 2.0,
                            (transformedFromNodeXY[1] + transformedToNodeXY[1]) / 2.0};

                    gc.setFill(Color.BROWN);
                    gc.fillText(edge.getRelationship(), transformedMidXY[0], transformedMidXY[1] + LABEL_FONT_SIZE);
                }
            }
        }
    }

    public void addOrRemoveDispNodeId(Node node){
        if(dispNodeIds.contains(node)){
            dispNodeIds.remove(node);
        }
        else{
            dispNodeIds.add(node);
        }
    }

    public boolean isShowAllNodeIds() {
        return showAllNodeIds;
    }

    public void setShowAllNodeIds(boolean showAllNodeIds) {
        this.showAllNodeIds = showAllNodeIds;
    }

    public boolean isShowAllRelationships() {
        return showAllRelationships;
    }

    public void setShowAllRelationships(boolean showAllRelationships) {
        this.showAllRelationships = showAllRelationships;
    }

    public Node getMouseOnNode() {
        return this.mouseOnNode;
    }

    public void setMouseOnNode(Node mouseOnNode) {
        this.mouseOnNode = mouseOnNode;
    }
}
