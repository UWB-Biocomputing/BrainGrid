package edu.uwb.braingrid.workbench.provvisualizer.model;

import edu.uwb.braingrid.workbench.utils.DateTime;
import javafx.scene.paint.Color;

import java.time.LocalDateTime;
import java.util.Calendar;

public class ActivityNode extends Node{
    private String startTime;
    private String endTime;

    public ActivityNode(double width, double height, Color color){
        super(width,height,color);
    }

    public ActivityNode(String id, double x, double y, double width, double height, Color color, String label,
                        String startTime, String endTime){
        super(id, x, y, width, height, color, label);
        this.startTime = startTime;
        this.endTime = endTime;
    }

    public ActivityNode clone(){
        return new ActivityNode(super.getId(),super.getX(), super.getY(),super.getWidth(),super.getHeight(),super.getColor(),
                super.getLabel(),this.startTime,this.endTime);
    }

    public String getStartTime() {
        return startTime;
    }

    public ActivityNode setStartTime(String startTime) {
        this.startTime = startTime;
        return this;
    }

    public String getEndTime() {
        return endTime;
    }

    public ActivityNode setEndTime(String endTime) {
        this.endTime = endTime;
        return this;
    }
}
