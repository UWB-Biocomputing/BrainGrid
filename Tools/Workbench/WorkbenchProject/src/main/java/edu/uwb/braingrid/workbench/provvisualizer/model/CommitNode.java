package edu.uwb.braingrid.workbench.provvisualizer.model;

import javafx.scene.paint.Color;

public class CommitNode extends Node{
    public CommitNode(double size, Color color, Node.NodeShape nodeShape){
        super(size,color,nodeShape);
    }

    public CommitNode(String id, double x, double y, double size, Color color, String label, Node.NodeShape nodeShape){
        super(id, x, y, size, color, label, nodeShape);
    }


    public CommitNode clone(){
        return new CommitNode(super.getId(),super.getX(), super.getY(),super.getSize(),super.getColor(),super.getLabel(),
                super.getShape());
    }
}
