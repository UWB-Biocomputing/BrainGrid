package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class EntityNode extends Node {
    public EntityNode(double width, double height, Color color){
        super(width,height,color);
    }

    public EntityNode(String id, double x, double y, double width, double height, Color color, String label){
        super(id, x, y, width,height, color, label);
    }


    public EntityNode clone(){
        return new EntityNode(super.getId(),super.getX(), super.getY(),super.getWidth(),super.getHeight(),super.getColor(),
                super.getLabel());
    }
}
