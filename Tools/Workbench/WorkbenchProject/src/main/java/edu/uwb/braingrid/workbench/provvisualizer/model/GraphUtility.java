package edu.uwb.braingrid.workbench.provvisualizer.model;

public class GraphUtility {
    public static double calculateSlope(double[] fromPoint, double[] toPoint){
        double slope = (toPoint[1] - fromPoint[1])/(toPoint[0] - fromPoint[0]);
        return slope;
    }

    public static double calculateSlopeAngle(double[] fromPoint, double[] toPoint){
        if(toPoint[0] < fromPoint[0]){
            return Math.atan(calculateSlope(fromPoint, toPoint)) + Math.PI;
        }
        else {
            return Math.atan(calculateSlope(fromPoint, toPoint));
        }
    }

    public static double[] calculateMidPoint(double[] point1, double[] point2){
        double[] midPoint = new double[]{(point1[0] + point2[0])/2, (point1[1] + point2[1])/2};
        return midPoint;
    }

    public static double[] findPointWithAngleDistance(double[] fromPoint, double angle, double distance){
        double[] targetPoint = {fromPoint[0] + distance * Math.cos(angle), fromPoint[1] + distance * Math.sin(angle)};
        return targetPoint;
    }
}