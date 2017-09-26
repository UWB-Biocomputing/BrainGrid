package edu.uwb.braingrid.workbench.provvisualizer.factory;

import edu.uwb.braingrid.workbench.provvisualizer.model.Edge;

public class EdgeFactory {

    private static EdgeFactory edgeFactory;

    private Edge defaultEdge = null;

    private EdgeFactory(){
    }

    public static EdgeFactory getInstance(){
        if(edgeFactory == null){
            return new EdgeFactory();
        }
        else{
            return edgeFactory;
        }
    }

    public Edge createDefaultEdge(){
        if(defaultEdge == null){
            defaultEdge = new Edge();
        }

        return defaultEdge.clone();
    }
}
