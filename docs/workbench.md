# 1. BrainGrid Workbench

## 1.1. What is BrainGrid Workbench?

BrainGrid Workbench is designed for a desktop application with a Graphical User Interface (GUI)

## 1.2. Getting Started
### 1.2.1. Clone the BrainGrid repository from GitHub.
* Make sure “Git” is installed in your operating system. Check https://git-scm.com/ for more information about Git.

* Clone the repository to a local folder by typing the following command.
```shell
git clone https://github.com/UWB-Biocomputing/BrainGrid.git [folder_name]
```
* If you would like to checkout a specific branch or commit, use the following command.
```shell
git checkout [branch_name/commit_number]
```

### 1.2.2. Compile and build the Workbench.
#### 1.2.2.1. Compile in command line.

* Make sure Maven is installed by checking Maven version using the following command.
```shell
mvn -v
```

* If Maven is installed, the output will be like the screen dump below.

![alt text](images/CheckMavenVersion.png "Maven is installed.")

* Type the following Maven command under the directory "Tools/Workbench/WorkbenchProject" to compile and create executable JAR file.
```shell
mvn clean install
```
* After the project is built successfully, a JAR file "BrainGridWorkbench-1.0-SNAPSHOT.jar " will be created under the directory "Tools/Workbench/WorkbenchProject/target". Run this JAR file to start using the Workbench.

#### 1.2.2.1. Compile in NetBeans.

* Create a New Project in NetBeans. Then, select "Maven" under project categories and choose the option "Project with existing POM". And, press “Next”.

![alt text](images/NetbeansNewMavenProject.png "New Netbeans Project with existing POM.")
 
* After pressing the "Finish" button, choose "Tools/Workbench/WorkbenchProject", then press the “Open Project” button.

![alt text](images/NetbeansNewProjectFinish.png "Press the finsih button.")

![alt text](images/NetbeansOpenProject.png "Open the Maven project.")

* After the project is opened, you can compile and build it by pressing the "Clean and Build Project" Menu option in NetBeans. Or, you can run the program directly.
 
* After the project is built successfully, it will generate a JAR file " BrainGridWorkbench-1.0-SNAPSHOT.jar " under the directory "Tools/Workbench/WorkbenchProject/target". Run this JAR file to start using the Workbench.

### 1.2.3. Using BrainGrid Workbench to run neural simulation
* If it is the first time running BrainGrid Workbench, copy “BaseTemplates”, “ParamsClassTemplateConfig” and “BaseTemplateConfig.xml” – under directory "Tools/Workbench/TemplateSelectionConfig" to the working directory berfore running BrainGrid Workbench.

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

### 1.2.4. About the generated script files
The generated script files do the following steps.
1. Make an empty directory and clone the BrainGrid repository.
2. Checkout the revision specified in SHA1 field of the script configuration dialog.
3. Execute "make -clean" to remove any existing binary files from the repository directory.
4. Execute "make" to build the simulator.
5. Run the simulation by calling the simulator with the corresponding parameters.