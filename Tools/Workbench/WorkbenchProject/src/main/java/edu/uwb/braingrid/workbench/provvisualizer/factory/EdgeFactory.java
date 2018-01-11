package edu.uwb.braingrid.workbench.provvisualizer.factory;

import edu.uwb.braingrid.workbench.provvisualizer.model.Edge;

public class EdgeFactory {

    private static EdgeFactory edgeFactory;

    private Edge defaultEdge = null;
    private Edge dashEdge = null;

    private EdgeFactory(){
    }

    public static EdgeFactory getInstance(){
        if(edgeFactory == null){
            edgeFactory = new EdgeFactory();
        }

        return edgeFactory;
    }

    public Edge createDefaultEdge(){
        if(defaultEdge == null){
            defaultEdge = new Edge();
        }

        return defaultEdge.clone();
    }

    public Edge createDashEdge(){
        if(dashEdge == null){
            dashEdge = new Edge(true);
        }

        return dashEdge.clone();
    }
}
