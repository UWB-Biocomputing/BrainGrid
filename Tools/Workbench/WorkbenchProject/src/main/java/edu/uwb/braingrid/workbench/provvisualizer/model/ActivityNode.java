package edu.uwb.braingrid.workbench.provvisualizer.model;

import edu.uwb.braingrid.workbench.utils.DateTime;
import javafx.scene.paint.Color;

import java.util.Calendar;

public class ActivityNode extends Node{
    Calendar startTime;
    Calendar endTime;

    public ActivityNode(double size, Color color, NodeShape nodeShape){
        super(size,color,nodeShape);
    }

    public ActivityNode(String id, double x, double y, double size, Color color, String label, NodeShape nodeShape,
                        Calendar startTime, Calendar endTime){
        super(id, x, y, size, color, label, nodeShape);
        this.startTime = startTime;
        this.endTime = endTime;
    }

    public ActivityNode clone(){
        return new ActivityNode(super.getId(),super.getX(), super.getY(),super.getSize(),super.getColor(),super.getLabel(),
                super.getShape(),this.startTime,this.endTime);
    }
}
