package edu.uwb.braingrid.workbench.provvisualizer.model;

import edu.uwb.braingrid.workbench.provvisualizer.utility.FileUtility;
import javafx.scene.paint.Color;

import java.util.ArrayList;

public class Node {
    private static final double SELECTION_TOLERANCE = 20;

    private String id ;
    private String displayId ;
    private double x ;
    private double y ;
    private double width;
    private double height;
    private Color color;
    private String label;
    private ArrayList<Node> neighborNodes ;
    private boolean absoluteXY ;

    public Node() {
        this.label = "";
        this.neighborNodes = new ArrayList<>();
        this.absoluteXY = false;
    }

    public Node(double width, double height, Color color) {
        this.width = width;
        this.height = height;
        this.color = color;
        this.label = "";
        this.neighborNodes = new ArrayList<>();
        this.absoluteXY = false;
    }

    public Node(String id, double x, double y, double width, double height, Color color) {
        this.id = id;
        if(id!=null){
            this.displayId = id.replaceFirst(FileUtility.FILE_PATH_PREFIX_REGEX,"");
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
        this.label = "";
        this.neighborNodes = new ArrayList<>();
        this.absoluteXY = false;
    }

    public Node(String id, double x, double y, double width, double height, Color color, String label) {
        this.id = id;
        if(id!=null){
            this.displayId = id.replaceFirst(FileUtility.FILE_PATH_PREFIX_REGEX,"");
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
        this.label = label;
        this.neighborNodes = new ArrayList<>();
        this.absoluteXY = false;
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
        if(id!=null){
            this.displayId = id.replaceFirst(FileUtility.FILE_PATH_PREFIX_REGEX,"");
        }
        return this;
    }

    public boolean isPointOnNode(double x, double y, double zoomRatio, boolean withTolerance){
        double cornerX = this.x - this.width/zoomRatio/2;
        double cornerY = this.y - this.height/zoomRatio/2;
        if(withTolerance &&
                x >= cornerX - SELECTION_TOLERANCE / zoomRatio &&
                x <= cornerX + (this.width + SELECTION_TOLERANCE) / zoomRatio &&
                y >= cornerY - SELECTION_TOLERANCE / zoomRatio &&
                y <= cornerY + (this.height + SELECTION_TOLERANCE) / zoomRatio){
            return true;
        }

        if(!withTolerance && x >= cornerX && x <= cornerX + this.width / zoomRatio && y >= cornerY &&
                y <= cornerY + this.height / zoomRatio){
            return true;
        }

        return false;
    }

    @Override
    public Node clone(){
        return new Node(id,x,y,width,height,color,label);
    }

    @Override
    public boolean equals(Object node){
        return node instanceof Node && this.id.equals(((Node) node).id);
    }

    @Override
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

    public boolean isAbsoluteXY() {
        return absoluteXY;
    }

    public Node setAbsoluteXY(boolean absoluteXY) {
        this.absoluteXY = absoluteXY;
        return this;
    }

    public String getDisplayId() {
        return displayId;
    }

    public void setDisplayId(String displayId) {
        this.displayId = displayId;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public boolean isInDisplayWindow(double[] displayWindowLocation, double[] displayWindowSize){
        double cornerX = this.x - this.width/2;
        double cornerY = this.y - this.height/2;
        if (cornerX < displayWindowLocation[0] || cornerX > displayWindowLocation[0] + displayWindowSize[0] ||
                cornerY < displayWindowLocation[1] || cornerY > displayWindowLocation[1] + displayWindowSize[1]){
            return false;
        }
        else {
            return true;
        }
    }
}
