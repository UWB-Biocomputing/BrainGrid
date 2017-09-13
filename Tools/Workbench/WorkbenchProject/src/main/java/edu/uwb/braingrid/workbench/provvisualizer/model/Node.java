package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

import java.util.ArrayList;

public class Node {
    private String id ;
    private double x ;
    private double y ;
    private double size;
    private Color color;
    private String label;
    private double[] netForce ;
    private ArrayList<Node> neighborNodes ;

    public Node(String id, double x, double y, double size, Color color) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = color;
        this.label = "";
        this.netForce = new double[]{0,0};
        this.neighborNodes = new ArrayList<>();
    }

    public Node(String id, double x, double y, double size, Color color, String label) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = color;
        this.label = label;
        this.netForce = new double[]{0,0};
        this.neighborNodes = new ArrayList<>();
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getSize() {
        return size;
    }

    public void setSize(double size) {
        this.size = size;
    }

    public Color getColor() {
        return color;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public boolean isPointOnNode(double x, double y, double zoomRatio){
        if(x >= this.x && x <= this.x + this.size / zoomRatio && y >= this.y && y <= this.y + this.size / zoomRatio){
            return true;
        }

        return false;
    }

    @Override
    public Node clone(){
        return new Node(id,x,y,size,color,label);
    }

    public boolean equals(Node node){
        return this.id == node.id;
    }

    public double[] getNetForce() {
        return netForce;
    }

    public void setNetForce(double[] netForce) {
        this.netForce = netForce;
    }

    public boolean isConnected(Node node2) {
        return this.neighborNodes.contains(node2);
    }

    public ArrayList<Node> getNeighborNodes() {
        return neighborNodes;
    }

    public void setNeighborNodes(ArrayList<Node> neighborNodes) {
        this.neighborNodes = neighborNodes;
    }
}
