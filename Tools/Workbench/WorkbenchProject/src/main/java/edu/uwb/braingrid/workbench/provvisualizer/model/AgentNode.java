package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class AgentNode extends Node{
    public AgentNode(double width, double height, Color color){
        super(width,height,color);
    }

    public AgentNode(String id, double x, double y, double width, double height, Color color, String label){
        super(id, x, y, width, height, color, label);
    }


    public AgentNode clone(){
        return new AgentNode(super.getId(),super.getX(), super.getY(),super.getWidth(),super.getHeight(),super.getColor(),
                super.getLabel());
    }
}
