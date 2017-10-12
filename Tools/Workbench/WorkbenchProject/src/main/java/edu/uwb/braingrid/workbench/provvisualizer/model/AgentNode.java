package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class AgentNode extends Node{
    public AgentNode(double size, Color color, NodeShape nodeShape){
        super(size,color,nodeShape);
    }

    public AgentNode(String id, double x, double y, double size, Color color, String label, NodeShape nodeShape){
        super(id, x, y, size, color, label, nodeShape);
    }


    public AgentNode clone(){
        return new AgentNode(super.getId(),super.getX(), super.getY(),super.getSize(),super.getColor(),super.getLabel(),
                super.getShape());
    }
}
