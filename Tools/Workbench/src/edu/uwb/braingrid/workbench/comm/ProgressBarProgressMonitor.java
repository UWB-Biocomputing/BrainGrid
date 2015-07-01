package edu.uwb.braingrid.workbench.comm;
/////////////////CLEANED

import com.jcraft.jsch.SftpProgressMonitor;
import java.io.File;
import javax.swing.JProgressBar;

/**
 * Provides call back functions to monitor the progress of SecureFileTransfer
 * operations.
 *
 * @author Del Davis
 */
public class ProgressBarProgressMonitor implements SftpProgressMonitor {

    JProgressBar bar;
    File file = null;
    long max;

    /**
     * Responsible for initializing and constructing this monitor object
     *
     * @param bar - The progress bar to update
     * @param f - The file being transferred
     */
    public ProgressBarProgressMonitor(JProgressBar bar, File f) {
        if (f != null && f.exists()) {
            file = f;
        }
        this.bar = bar;
    }

    /**
     * Initializes the progress bar and file transfer progress. Will be called
     * when new operation starts.
     *
     * @param op - a code indicating the direction of transfer, one of PUT and
     * GET
     * @param src - the source file name.
     * @param dest - the destination file name.
     * @param max - the final count (i.e. length of file to transfer).
     */
    @Override
    public void init(int op, String src, String dest, long max) {
        bar.setValue(bar.getMinimum());
        if (max == -1 && file != null) {
            this.max = file.length();
        } else {
            this.max = max;
        }
    }

    /**
     * Updates the progress of the transfer. Will be called periodically as more
     * data is transfered.
     *
     * @param bytes - the number of bytes transferred so far
     * @return true if the transfer should go on, false if the transfer should
     * be cancelled.
     */
    @Override
    public boolean count(long bytes) {
        long percentageComplete = 0;
        if (max > 0) {
            percentageComplete = (long) ((float) bytes / max * 100);
        }
        bar.setValue((int) percentageComplete);
        return true;
    }

    /**
     * Will be called when the transfer ended, either because all the data was
     * transferred, or because the transfer was cancelled.
     */
    @Override
    public void end() {
        bar.setValue(bar.getMaximum());
    }
}
