package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class CommitNode extends Node{
    public CommitNode(double width, double height, Color color){
        super(width,height,color);
    }

    public CommitNode(String id, double x, double y, double width, double height, Color color, String label){
        super(id, x, y, width,height, color, label);
    }


    public CommitNode clone(){
        return new CommitNode(super.getId(),super.getX(), super.getY(),super.getWidth(),super.getHeight(),super.getColor(),
                super.getLabel());
    }
}
