package edu.uwb.braingrid.workbench.provvisualizer.factory;

import edu.uwb.braingrid.workbench.provvisualizer.model.*;
import javafx.scene.paint.Color;

import javax.swing.text.html.parser.Entity;

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
    private static final double COMMIT_NODE_SIZE = 25;
    private static final Color COMMIT_NODE_COLOR = Color.BLUEVIOLET;
    private static final Node.NodeShape COMMIT_NODE_SHAPE = Node.NodeShape.DOUBLE_CIRCLE;

    private Node defaultNode = null;
    private EntityNode entityNode = null;
    private AgentNode agentNode = null;
    private ActivityNode activityNode = null;
    private CommitNode commitNode = null;

    private NodeFactory(){
    }

    public static NodeFactory getInstance(){
        if(nodeFactory == null){
            nodeFactory = new NodeFactory();
        }

        return nodeFactory;
    }

    public Node createDefaultNode(){
        if(defaultNode == null){
            defaultNode = new Node();
        }

        return defaultNode.clone();
    }

    public ActivityNode createActivityNode(){
        if(activityNode == null){
            activityNode = new ActivityNode(ACTIVITY_NODE_SIZE, ACTIVITY_NODE_COLOR, ACTIVITY_NODE_SHAPE);
        }

        return activityNode.clone();
    }

    public AgentNode createAgentNode(){
        if(agentNode == null){
            agentNode = new AgentNode(AGENT_NODE_SIZE, AGENT_NODE_COLOR, AGENT_NODE_SHAPE);
        }

        return agentNode.clone();
    }

    public CommitNode createCommitNode(){
        if(commitNode == null){
            commitNode = new CommitNode(COMMIT_NODE_SIZE, COMMIT_NODE_COLOR, COMMIT_NODE_SHAPE);
        }

        return commitNode.clone();
    }

    public EntityNode createEntityNode(){
        if(entityNode == null){
            entityNode = new EntityNode(ENTITY_NODE_SIZE, ENTITY_NODE_COLOR, ENTITY_NODE_SHAPE);
        }

        return entityNode.clone();
    }

    public ActivityNode convertToActivityNode(Node node){
        ActivityNode activityNode = createActivityNode();
        activityNode.setId(node.getId()).setX(node.getX()).setY(node.getY()).setLabel(node.getLabel());

        return activityNode;
    }

    public AgentNode convertToAgentNode(Node node){
        AgentNode agentNode = createAgentNode();
        agentNode.setId(node.getId()).setX(node.getX()).setY(node.getY()).setLabel(node.getLabel());

        return agentNode;
    }

    public CommitNode convertToCommitNode(Node node){
        CommitNode commitNode = createCommitNode();
        commitNode.setId(node.getId()).setX(node.getX()).setY(node.getY()).setLabel(node.getLabel());

        return commitNode;
    }

    public EntityNode convertToEntityNode(Node node){
        EntityNode entityNode = createEntityNode();
        entityNode.setId(node.getId()).setX(node.getX()).setY(node.getY()).setLabel(node.getLabel());

        return entityNode;
    }
}
