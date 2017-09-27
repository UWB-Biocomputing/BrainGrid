package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.canvas.Canvas;
import javafx.scene.paint.Color;
import javafx.scene.shape.Polygon;

import java.awt.*;
import java.util.ArrayList;

public class Node {
    public enum NodeShape{
        CIRCLE,
        SQUARE,
        ROUNDED_SQUARE
    }

    private String id ;
    private double x ;
    private double y ;
    private NodeShape shape;
    private double size;
    private Color color;
    private String label;
    private ArrayList<Node> neighborNodes ;

    public Node() {
        this.label = "";
        this.neighborNodes = new ArrayList<>();
    }

    public Node(double size, Color color, NodeShape nodeShape) {
        this.size = size;
        this.color = color;
        this.shape = nodeShape;
        this.label = "";
        this.neighborNodes = new ArrayList<>();
    }

    public Node(String id, double x, double y, double size, Color color, NodeShape nodeShape) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = color;
        this.shape = nodeShape;
        this.label = "";
        this.neighborNodes = new ArrayList<>();
    }

    public Node(String id, double x, double y, double size, Color color, String label, NodeShape nodeShape) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = color;
        this.shape = nodeShape;
        this.label = label;
        this.neighborNodes = new ArrayList<>();
    }

    public double getX() {
        return x;
    }

    public Node setX(double x) {
        this.x = x;
        return this;
    }

    public double getY() {
        return y;
    }

    public Node setY(double y) {
        this.y = y;
        return this;
    }

    public double getSize() {
        return size;
    }

    public Node setSize(double size) {
        this.size = size;
        return this;
    }

    public Color getColor() {
        return color;
    }

    public Node setColor(Color color) {
        this.color = color;
        return this;
    }

    public String getLabel() {
        return label;
    }

    public Node setLabel(String label) {
        this.label = label;
        return this;
    }

    public String getId() {
        return id;
    }

    public Node setId(String id) {
        this.id = id;
        return this;
    }

    public NodeShape getShape() {
        return shape;
    }

    public void setShape(NodeShape shape) {
        this.shape = shape;
    }

    public boolean isPointOnNode(double x, double y, double zoomRatio){
        if(x >= this.x && x <= this.x + this.size / zoomRatio && y >= this.y && y <= this.y + this.size / zoomRatio){
            return true;
        }

        return false;
    }

    @Override
    public Node clone(){
        return new Node(id,x,y,size,color,label,shape);
    }

    public boolean equals(Node node){
        return this.id.equals(node.id);
    }

    public int hashCode(){ return this.id.hashCode();}

    public boolean isConnected(Node node2) {
        return this.neighborNodes.contains(node2);
    }

    public ArrayList<Node> getNeighborNodes() {
        return neighborNodes;
    }

    public void setNeighborNodes(ArrayList<Node> neighborNodes) {
        this.neighborNodes = neighborNodes;
    }

    public boolean isInDisplayWindow(double[] displayWindowLocation, double[] displayWindowSize){
        if (this.x < displayWindowLocation[0] && this.x > displayWindowLocation[0] + displayWindowSize[0] ||
                this.y < displayWindowLocation[1] && this.y > displayWindowLocation[1] + displayWindowSize[1]){
            return false;
        }
        else {
            return true;
        }
    }
}
