package edu.uwb.braingrid.workbench.comm;

import com.jcraft.jsch.SftpProgressMonitor;
import java.io.File;
import javax.swing.JProgressBar;

/**
 * Provides call back functions used in SecureFileTransfer transfer operations
 *
 * @author Del Davis
 */
public class ProgressBarProgressMonitor implements SftpProgressMonitor {

    JProgressBar bar;
    File file = null;
    long max;

    public ProgressBarProgressMonitor(JProgressBar bar, File f) {
        if (f != null && f.exists()) {
            file = f;
        }
        this.bar = bar;
    }

    @Override
    public void init(int op, String src, String dest, long max) {
        bar.setValue(bar.getMinimum());
        if (max == -1 && file != null) {
            this.max = file.length();
        } else {
            this.max = max;
        }
    }

    @Override
    public boolean count(long bytes) {
        long percentageComplete = 0;
        if (max > 0) {
            percentageComplete = (long) ((float) bytes / max * 100);
        }
        bar.setValue((int) percentageComplete);
        return true;
    }

    @Override
    public void end() {
        bar.setValue(bar.getMaximum());
    }
}
