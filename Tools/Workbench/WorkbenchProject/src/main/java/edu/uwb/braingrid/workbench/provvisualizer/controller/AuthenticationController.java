package edu.uwb.braingrid.workbench.provvisualizer.controller;

import com.jcraft.jsch.*;
import edu.uwb.braingrid.workbench.provvisualizer.ProvVisGlobal;
import edu.uwb.braingrid.workbench.provvisualizer.factory.EdgeFactory;
import edu.uwb.braingrid.workbench.provvisualizer.factory.NodeFactory;
import edu.uwb.braingrid.workbench.provvisualizer.model.*;
import edu.uwb.braingrid.workbench.provvisualizer.view.VisCanvas;
import javafx.animation.AnimationTimer;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;
import org.apache.jena.riot.RDFDataMgr;
import org.controlsfx.control.ToggleSwitch;

import java.io.File;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.function.Consumer;

public class AuthenticationController {
    private Consumer<AuthenticationInfo> okButtonCallback ;

    @FXML
    private Label hostnameLbl;

    @FXML
    private TextField usernameTxtFd;

    @FXML
    private PasswordField passwordPwdFd;

    @FXML
    private Button okBtn;

    @FXML
    private Button cancelBtn;

    /**
     * Initializes the controller.
     */
    @FXML
    public void initialize(){
        okBtn.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if(okButtonCallback != null){
                    okButtonCallback.accept(new AuthenticationInfo(hostnameLbl.getText(),usernameTxtFd.getText(),passwordPwdFd.getText()));
                }
                ((Stage)okBtn.getScene().getWindow()).close();
            }
        });
        okBtn.setDefaultButton(true);

        cancelBtn.setCancelButton(true);
        cancelBtn.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                ((Stage)cancelBtn.getScene().getWindow()).close();
            }
        });
    }

    public void setOkBtnCallback(Consumer<AuthenticationInfo> callback){
        okButtonCallback = callback;
    }

    public void setHostname(String hostname) {
        hostnameLbl.setText(hostname);
    }

    public String getUsername() {
        return usernameTxtFd.getText();
    }

    public void setUsername(String username) {
        usernameTxtFd.setText(username);
    }

    public String getPassword() {
        return passwordPwdFd.getText();
    }
}
