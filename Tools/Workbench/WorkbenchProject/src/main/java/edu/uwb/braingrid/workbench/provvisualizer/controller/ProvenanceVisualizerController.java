package edu.uwb.braingrid.workbench.provvisualizer.controller;

import edu.uwb.braingrid.workbench.provvisualizer.model.Edge;
import edu.uwb.braingrid.workbench.provvisualizer.model.Node;
import edu.uwb.braingrid.workbench.provvisualizer.view.VisCanvas;
import javafx.animation.AnimationTimer;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;
import org.apache.jena.riot.RDFDataMgr;
import org.controlsfx.control.ToggleSwitch;

import java.util.ArrayList;
import java.util.HashMap;

public class ProvenanceVisualizerController {
    private static String RDF_SYNTAX_PREFIX = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    private static String RDF_SCHEME_PREFIX = "http://www.w3.org/2000/01/rdf-schema#";
    private static String PROV_PREFIX = "http://www.w3.org/ns/prov#";
    private static String RDF_TYPE = RDF_SYNTAX_PREFIX + "type";
    private static String RDF_LABEL = RDF_SCHEME_PREFIX + "label";
    private static String PROV_ACTIVITY = PROV_PREFIX + "Activity";
    private static String PROV_SW_AGENT = PROV_PREFIX + "SoftwareAgent";
    private static String PROV_ENTITY = PROV_PREFIX + "Entity";
    private static final double NODE_SIZE = 20;

    private GraphicsContext gc ;
    private Model provModel ;
    private HashMap<String, Node> nodes = new HashMap<>();
    private HashMap<String, Edge> edges = new HashMap<>();
    private AnimationTimer timer;
    private double zoomRatio = 1;
    private Node draggedNode ;
    private double zoomSpeed=.2;

    private double[] pressedXY ;

    private double[] displayWindowLocation = new double[]{ 0, 0 };
    private double[] displayWindowSize = new double[]{ 0, 0 };

    private double[] displayWindowLocationTmp ;

    //private double c1 = 2;
    //private double c2 = 1;
    //private double c3 = 1;
    //private double c4 = 0.1;

    private double c1 = 2;
    private double c2 = 1;
    private double c3 = 3000;
    private double c4 = 0.1;

    @FXML
    private VisCanvas visCanvas;
    @FXML
    private AnchorPane canvasPane;
    @FXML
    private ToggleSwitch showNodeIds;
    @FXML
    private ToggleSwitch showRelationships;

    /**
     * Initializes the controller.
     */
    @FXML
    public void initialize(){
        gc = visCanvas.getGraphicsContext2D();

        initNodeEdge(System.getProperty("user.dir") + "/projects/haha/provenance/haha.ttl");
        // Bind canvas size to stack pane size.

        visCanvas.widthProperty().bind(
                canvasPane.widthProperty());
        visCanvas.heightProperty().bind(
                canvasPane.heightProperty());

        initMouseEvents();

        timer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                moveNodes(draggedNode);
                drawOnCanvas();
            }
        };

        timer.start();
    }

    /**
     * Using Force-directed graph layout algorithm to optimize the node positions
     * @param draggedNode
     */
    private void moveNodes(Node draggedNode) {
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
        double factor = attractiveFunction(distance);
        vec[0]=vec[0]*factor;
        vec[1]=vec[1]*factor;
        return vec;
    }

    /**
     * Computes the value of the scalar attractive force function based on the given distance of two nodes.
     * @param distance the distance between the two nodes
     * @return attractive force
     */
    private double attractiveFunction(double distance) {
        //if (distance<stabilizer1){
        //    distance=stabilizer1;
        //}
        //return c1*Math.log(distance/c2)*(1/(stabilizer2*numON));
        return c1*Math.log(distance/c2);
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
        double factor = -repellingFunction(distance);
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
     * Computes the value of the scalar repelling force function based on the given distance of two nodes.
     * @param distance the distance between the two nodes
     * @return attractive force
     */
    private double repellingFunction(double distance) {
        //if (distance<stabilizer1){
        //    distance=stabilizer1;
        //}
        return c3/Math.pow(distance, 2);
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

    private void initMouseEvents(){

        visCanvas.setOnMouseDragged(new EventHandler<MouseEvent>(){
            @Override
            public void handle(MouseEvent event) {
                if (draggedNode != null) { // drag node
                    draggedNode.setX(event.getX() / zoomRatio + displayWindowLocation[0]);
                    draggedNode.setY(event.getY() / zoomRatio + displayWindowLocation[1]);
                }else{
                    displayWindowLocation[0] = displayWindowLocationTmp[0] + pressedXY[0] - event.getX() / zoomRatio;
                    displayWindowLocation[1] = displayWindowLocationTmp[1] + pressedXY[1] - event.getY() / zoomRatio;
                }

            }
        });

        visCanvas.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                draggedNode = getSelectedNode(event.getX() / zoomRatio + displayWindowLocation[0], event.getY() / zoomRatio + displayWindowLocation[1]);
                if(draggedNode == null){
                    pressedXY = new double[]{event.getX() / zoomRatio, event.getY() / zoomRatio};
                    displayWindowLocationTmp = displayWindowLocation.clone();
                }
            }
        });

        visCanvas.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                draggedNode = null;
            }
        });

        visCanvas.setOnScroll(new EventHandler<ScrollEvent>() {
            @Override
            public void handle(ScrollEvent event) {
                //update zoomRatio
                double deltaY = event.getDeltaY();
                double oldZoomRatio = zoomRatio;

                if(deltaY > 0){
                    zoomRatio = zoomRatio * (1 + zoomSpeed);
                }
                else if(deltaY < 0){
                    zoomRatio = zoomRatio / (1 + zoomSpeed);
                }

                if(deltaY != 0){
                    displayWindowSize[0] = visCanvas.getWidth() / zoomRatio;
                    displayWindowSize[1] = visCanvas.getHeight() / zoomRatio;
                    displayWindowLocation[0] = ((zoomRatio - oldZoomRatio) / (zoomRatio * oldZoomRatio)) * event.getX() + displayWindowLocation[0];
                    displayWindowLocation[1] = ((zoomRatio - oldZoomRatio) / (zoomRatio * oldZoomRatio)) * event.getY() + displayWindowLocation[1];
                }
            }
        });
    }

    private Node getSelectedNode(double x, double y) {
        for(Node node : nodes.values()){
            if(node.isPointOnNode(x, y, zoomRatio)){
                return node;
            }
        }
        return null;
    }

    private void initNodeEdge(String provFileURI){
        provModel = RDFDataMgr.loadModel(provFileURI);
        StmtIterator iter = provModel.listStatements();
        Statement stmt;
        while (iter.hasNext()) {
            stmt = iter.nextStatement();
            String predicateStr = stmt.getPredicate().toString();
            if(predicateStr.equals(RDF_TYPE)){
                String subjectStr = stmt.getSubject().toString();
                String objectStr = stmt.getObject().toString();
                if(objectStr.equals(PROV_ACTIVITY) && !nodes.containsKey(subjectStr)){
                    //create Activity Node
                    nodes.put(subjectStr,new Node(subjectStr,Math.random()*visCanvas.getWidth(),Math.random()*visCanvas.getHeight(),NODE_SIZE ,Color.BLUE));
                }
                else if(objectStr.equals(PROV_SW_AGENT) && !nodes.containsKey(subjectStr)){
                    //create Agent Node
                    nodes.put(subjectStr,new Node(subjectStr,Math.random()*visCanvas.getWidth(),Math.random()*visCanvas.getHeight(),NODE_SIZE ,Color.BLUE));
                }
                else if(objectStr.equals(PROV_ENTITY) && !nodes.containsKey(subjectStr)){
                    //create Entity Node
                    nodes.put(subjectStr, new Node(subjectStr,Math.random()*visCanvas.getWidth(),Math.random()*visCanvas.getHeight(),NODE_SIZE ,Color.BLUE));
                }
            }
            else if(predicateStr.equals(RDF_LABEL)){
                String subjectStr = stmt.getSubject().toString();
                if(nodes.containsKey(subjectStr)) {
                    nodes.get(subjectStr).setLabel(stmt.getObject().toString());
                }
                else{
                    nodes.put(subjectStr, new Node(subjectStr,Math.random()*visCanvas.getWidth(),Math.random()*visCanvas.getHeight(),NODE_SIZE ,Color.BLUE, stmt.getObject().toString()));
                }
            }
            else if(stmt.getObject().isURIResource()){
                Edge edge = new Edge(stmt.getSubject().toString(), stmt.getObject().toString(), stmt.getPredicate().toString());
                edges.put(edge.getEdgeId(), edge);
            }
            //System.out.println(stmt.getSubject().toString() + " " + stmt.getPredicate().toString() + " " + stmt.getObject().toString());
        }

        //set neighbors
        for (Edge edge: edges.values()){
            Node fromNode = nodes.get(edge.getFromNodeId());
            Node toNode = nodes.get(edge.getToNodeId());
            fromNode.getNeighborNodes().add(toNode);
            toNode.getNeighborNodes().add(fromNode);
        }
    }

    private void drawOnCanvas(){
        //draw background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, visCanvas.getWidth(), visCanvas.getHeight());
        gc.setStroke(Color.BLACK);

        for(Edge edge : edges.values()){
            Node fromNode = nodes.get(edge.getFromNodeId());
            Node toNode = nodes.get(edge.getToNodeId());
            if(isInDisplayWindow(fromNode) || isInDisplayWindow(toNode)){
                double[] transformedFromNodeXY = transformToRelativeXY(fromNode.getX(),fromNode.getY());
                double[] transformedToNodeXY = transformToRelativeXY(toNode.getX(),toNode.getY());
                gc.strokeLine(transformedFromNodeXY[0] + fromNode.getSize()/2,transformedFromNodeXY[1] + fromNode.getSize()/2,
                        transformedToNodeXY[0] + toNode.getSize()/2,transformedToNodeXY[1] + toNode.getSize()/2);
            }
        }

        for(Node node : nodes.values()){
            if(isInDisplayWindow(node)) {
                double[] transformedNodeXY = transformToRelativeXY(node.getX(),node.getY());
                gc.setFill(node.getColor());
                gc.strokeOval(transformedNodeXY[0], transformedNodeXY[1], node.getSize(), node.getSize());
                gc.fillOval(transformedNodeXY[0], transformedNodeXY[1], node.getSize(), node.getSize());
            }
        }

        if(showNodeIds.isSelected() || showRelationships.isSelected()) {
            for(Node node : nodes.values()){
                if(isInDisplayWindow(node)) {
                    double[] transformedNodeXY = transformToRelativeXY(node.getX(),node.getY());
                    gc.strokeText(node.getId(), transformedNodeXY[0], transformedNodeXY[1] + node.getSize());
                }
            }
        }
    }

    private boolean isInDisplayWindow(Node node){
        if (node.getX() < displayWindowLocation[0] && node.getX() > displayWindowLocation[0] + displayWindowSize[0] ||
                node.getY() < displayWindowLocation[1] && node.getY() > displayWindowLocation[1] + displayWindowSize[1]){
            return false;
        }
        else {
            return true;
        }
    }

    private double[] transformToRelativeXY(double x, double y){
        double[] xy = new double[2];
        xy[0] = (x - displayWindowLocation[0]) * zoomRatio;
        xy[1] = (y - displayWindowLocation[1]) * zoomRatio;
        
        return xy;
    }
}
