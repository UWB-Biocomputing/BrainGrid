package edu.uwb.braingrid.workbench.provvisualizer.controller;

import edu.uwb.braingrid.workbench.provvisualizer.model.*;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.Stage;
import java.util.function.Consumer;

public class AuthenticationController {
	private Consumer<AuthenticationInfo> okButtonCallback;

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
	public void initialize() {
		okBtn.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				if (okButtonCallback != null) {
					okButtonCallback.accept(new AuthenticationInfo(hostnameLbl.getText(), usernameTxtFd.getText(),
							passwordPwdFd.getText()));
				}
				((Stage) okBtn.getScene().getWindow()).close();
			}
		});
		okBtn.setDefaultButton(true);

		cancelBtn.setCancelButton(true);
		cancelBtn.setOnAction(new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
				((Stage) cancelBtn.getScene().getWindow()).close();
			}
		});
	}

	public void setOkBtnCallback(Consumer<AuthenticationInfo> callback) {
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
