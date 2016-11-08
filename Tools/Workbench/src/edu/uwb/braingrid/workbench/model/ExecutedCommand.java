package edu.uwb.braingrid.workbench.model;
// CLEANED

import java.util.Date;

/**
 * Created by Nathan on 9/1/2014. Modified by Del on 9/3/2014
 */
public class ExecutedCommand {

    private final String fullCommand;
    private final String simpleCommand;

    private Date timeStarted;
    private Date timeCompleted;
    private int exitStatus;

    /**
     * Responsible for allocating this executed command and setting all members
     * to their initial values.
     *
     * Note: The exit status is not initialized during construction in order to
     * differentiate between the code for a normal exit or an error-state and
     * the fact that the program did not exit
     *
     * @param newCommand - The exact command that was invoked, including all
     * parameters
     */
    public ExecutedCommand(String newCommand) {
        fullCommand = newCommand;
        int firstSpaceIndex = fullCommand.indexOf(" ");
        if (firstSpaceIndex < 0) {
            firstSpaceIndex = fullCommand.length();
        }
        simpleCommand = fullCommand.substring(0, firstSpaceIndex);
        timeStarted = null;
        timeCompleted = null;
    }

    /**
     * Indicates the exact command that was invoked, including all parameters.
     * If several commands were executed in series, this will include those as
     * well.
     *
     * @return The exact command that was invoked
     */
    public String getFullCommand() {
        return fullCommand;
    }

    /**
     * Indicates the name of the executable file which was called to invoke the
     * command.
     *
     * @return - The name of the executable file which was called to execute the
     * command. This does not include any arguments that were provided to the
     * executable at invocation
     */
    public String getSimpleCommand() {
        return simpleCommand;
    }

    /**
     * Sets the date at which the command was executed
     *
     * @param newStartedTime
     */
    public void setTimeStarted(Date newStartedTime) {
        timeStarted = newStartedTime;
    }

    /**
     * Provides the date when the command was executed
     *
     * @return The date when the command was executed
     */
    public Date getTimeStarted() {
        return timeStarted;
    }

    /**
     * Sets the date at which the command finished execution
     *
     * @param newCompletedTime - The date at which the command completed
     * execution
     */
    public void setTimeCompleted(Date newCompletedTime) {
        timeCompleted = newCompletedTime;
    }

    /**
     * Provides the date when the command finished execution
     *
     * @return - The date when the command finished execution
     */
    public Date getTimeCompleted() {
        return timeCompleted;
    }

    /**
     * Sets the exit status
     *
     * @param newExitStatus - A value provided by processes during exit that
     * indicates the last state. This value is used to indicate specific
     * error-states in addition to a normal state
     */
    public void setExitStatus(int newExitStatus) {
        exitStatus = newExitStatus;
    }

    /**
     * Indicates the system dependent exit status provided at process exit
     *
     * Safety Measure: Call hasCompleted, or optionally, catch the
     * NullPointerException that may occur. The integer value associated with
     * this function is not initialized in order to prevent erroneous
     * assumptions about the exit status of the process. This is necessary due
     * to the range of values provided by exiting processes.
     *
     * @return - An integer. The value of the integer varies by process and
     * process state at exit
     * @throws NullPointerException
     */
    public int getExitStatus() throws NullPointerException {
        return exitStatus;
    }

    /**
     * Indicates if this command started.
     *
     * @return True if the command started by analysis time
     */
    public boolean hasStarted() {
        return timeStarted != null;
    }

    /**
     * Indicates if this command completed. The value indicating completion is
     * set during output analysis. As such, the command may have completed after
     * analysis
     *
     * @return True if the command completed at analysis time, otherwise false
     */
    public boolean hasCompleted() {
        return timeCompleted != null;
    }
}
