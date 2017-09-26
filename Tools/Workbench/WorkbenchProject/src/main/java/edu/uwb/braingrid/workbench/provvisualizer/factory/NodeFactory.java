package edu.uwb.braingrid.workbench.provvisualizer.factory;

import edu.uwb.braingrid.workbench.provvisualizer.model.Node;
import javafx.scene.paint.Color;

public class NodeFactory {
    private static NodeFactory nodeFactory;
    private static final double ACTIVITY_NODE_SIZE = 20;
    private static final Color ACTIVITY_NODE_COLOR = Color.GREEN;
    private static final Node.NodeShape ACTIVITY_NODE_SHAPE = Node.NodeShape.ROUNDED_SQUARE;
    private static final double AGENT_NODE_SIZE = 15;
    private static final Color AGENT_NODE_COLOR = Color.RED;
    private static final Node.NodeShape AGENT_NODE_SHAPE = Node.NodeShape.SQUARE;
    private static final double ENTITY_NODE_SIZE = 20;
    private static final Color ENTITY_NODE_COLOR = Color.BLUE;
    private static final Node.NodeShape ENTITY_NODE_SHAPE = Node.NodeShape.CIRCLE;

    private Node defaultNode = null;
    private Node entityNode = null;
    private Node agentNode = null;
    private Node activityNode = null;

    private NodeFactory(){
    }

    public static NodeFactory getInstance(){
        if(nodeFactory == null){
            return new NodeFactory();
        }
        else{
            return nodeFactory;
        }
    }

    public Node createDefaultNode(){
        if(defaultNode == null){
            defaultNode = new Node();
        }

        return defaultNode.clone();
    }

    public Node createEntityNode(){
        if(entityNode == null){
            entityNode = new Node(ENTITY_NODE_SIZE, ENTITY_NODE_COLOR, ENTITY_NODE_SHAPE);
        }

        return entityNode.clone();
    }

    public Node createAgentNode(){
        if(agentNode == null){
            agentNode = new Node(AGENT_NODE_SIZE, AGENT_NODE_COLOR, AGENT_NODE_SHAPE);
        }

        return agentNode.clone();
    }

    public Node createActivityNode(){
        if(activityNode == null){
            activityNode = new Node(ACTIVITY_NODE_SIZE, ACTIVITY_NODE_COLOR, ACTIVITY_NODE_SHAPE);
        }

        return activityNode.clone();
    }

    public Node convertToEntityNode(Node node){
        node.setSize(ENTITY_NODE_SIZE).setColor(ENTITY_NODE_COLOR).setShape(ENTITY_NODE_SHAPE);

        return node;
    }

    public Node convertToAgentNode(Node node){
        node.setSize(AGENT_NODE_SIZE).setColor(AGENT_NODE_COLOR).setShape(AGENT_NODE_SHAPE);

        return node;
    }

    public Node convertToActivityNode(Node node){
        node.setSize(ACTIVITY_NODE_SIZE).setColor(ACTIVITY_NODE_COLOR).setShape(ACTIVITY_NODE_SHAPE);

        return node;
    }
}
