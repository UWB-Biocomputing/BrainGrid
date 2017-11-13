package edu.uwb.braingrid.workbench.provvisualizer;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class ProvenanceVisualizer extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        Parent root;
        try {
            root = FXMLLoader.load(getClass().getResource("/provvisualizer/view/ProvenanceVisualizerView.fxml"));
            Scene scene = new Scene(root,1200,600);
            primaryStage.setScene(scene);
            primaryStage.setTitle("Provenance Visualizer");
            primaryStage.show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
