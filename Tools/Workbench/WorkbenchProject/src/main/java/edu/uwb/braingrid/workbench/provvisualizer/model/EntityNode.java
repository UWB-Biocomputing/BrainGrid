package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class EntityNode extends Node {
    public EntityNode(double size, Color color, NodeShape nodeShape){
        super(size,color,nodeShape);
    }

    public EntityNode(String id, double x, double y, double size, Color color, String label, NodeShape nodeShape){
        super(id, x, y, size, color, label, nodeShape);
    }


    public EntityNode clone(){
        return new EntityNode(super.getId(),super.getX(), super.getY(),super.getSize(),super.getColor(),super.getLabel(),
                super.getShape());
    }
}
