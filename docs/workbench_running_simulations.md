## 2. Running Simulations
### 2.1. The workflow
* If it is the first time running BrainGrid Workbench, copy “BaseTemplates”, “ParamsClassTemplateConfig” and “BaseTemplateConfig.xml” – under directory "Tools/Workbench/TemplateSelectionConfig" to the work directory berfore running BrainGrid Workbench.

![alt text](images/FilesToCopy.png "Copy files")
 
* If the Workbench is started successfully, a new window as shown in the below screen dump will be displayed.

![alt text](images/WorkbenchHome.png "Home")


* Create a new simulation project. Choose File - > New.

![alt text](images/WorkbenchNewProject.png "New Project")

* Enter the project name and press the “OK” button.

![alt text](images/WorkbenchNewProjectOK.png "Press OK")
 
* After that, a message “New project specified” is displayed. Click the “Configure” button to start configuring the simulation.

![alt text](images/WorkbenchConfigButton.png "Configure")
 
* A new window to setup the parameters classes is displayed. After choosing the parameters classes, press the “OK” button.

![alt text](images/WorkbenchParamClassesSelection.png "Select Classes")
 
* Depending on the parameters classes selected, the corresponding parameters are displayed with their default values for user to set. Use the “Import” button in the “LayoutParams” tab to import layout files, if necessary. After setting all the parameter values, press the “Build” button to generate the config file for the simulation.

![alt text](images/WorkbenchSetParams1.png "Set Parameters")

![alt text](images/WorkbenchSetParams2.png "Set Parameters")
 
* If the config file is generated successfully, press the “OK” button to close the window.

![alt text](images/WorkbenchSetParams3.png "Press OK")
 
* Press the “Specify” button to open the “Script Specification” window.

![alt text](images/WorkbenchSpecifyButton.png "Specify")
 
* Setup the parameters for script generation. The “SHA1 Checkout Key” is the commit number. If it is blank, it will checkout the latest version. 

![alt text](images/WorkbenchScriptSpecification.png "Script Specification")
 
* Press the “Generate” button to generate the script file.

![alt text](images/WorkbenchGenerateButton.png "Generate script")
 
* Press the “Run script” to run the script. If the simulator location is a remote location, it transfers the configuration, neuron list and the script files to the remote machine before running the script on the remote machine.

![alt text](images/WorkbenchRunScriptButton.png "Run script")
 
* Before sending files to remote machine, enter the username and password to establish connection to the remote machine.

![alt text](images/WorkbenchCredential.png "Credential")

* If success, the message box will show the layout files have been uploaded to the remote location and the script is being executed.

![alt text](images/WorkbenchScriptStarted.png "Script Started")
 
* Press the “Analyze” button to download simulation results after the completion of the simulation.

![alt text](images/WorkbenchAnalyzeButton.png "Analyze")
 
* The execution time of the simulation depends on various factors, such as the configuration file, the available resources on the remote machine. So, if the simulation is not completed or unexpected errors occur during the execution, a message will be displayed to indicate that the download of execution result failed. Then, users need to investigate on the remote machine to find out the root cause.
 
![alt text](images/WorkbenchResultDownloadFail.png "Fail")

### 1.2. About the generated script files
The generated script files do the following steps.
1. Make an empty directory and clone the BrainGrid repository.
2. Checkout the revision specified in SHA1 field of the script configuration dialog.
3. Execute "make -clean" to remove any existing binary files from the repository directory.
4. Execute "make" to build the simulator.
5. Run the simulation by calling the simulator with the corresponding parameters.