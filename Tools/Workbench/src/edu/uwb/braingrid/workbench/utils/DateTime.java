package edu.uwb.braingrid.workbench.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

/**
 * Collection of static helper functions centered around temporal operations.
 *
 * @author Del Davis
 */
public class DateTime {

    public static final long ERROR_TIME = -1L;
    private static final String TIME_PATTERN = "HH:mm:ss";

    /**
     * Converts the number of milliseconds since January 1, 1970, 00:00:00 GMT
     * represented by this date
     *
     * @param millisFromEpoch - The number of milliseconds since January 1,
     * 1970, 00:00:00 GMT represented by this date.
     * @return description of time in the form hours:minutes:seconds (e.g.
     * 14:52:47)
     */
    public static String getTime(Long millisFromEpoch) {
        DateFormat dateFormat = new SimpleDateFormat(TIME_PATTERN);
        String time = dateFormat.format(millisFromEpoch);
        return time;
    }
    
    public static void recordProvTiming(String codeLocation, Long startingMillis) {
        long time = (System.currentTimeMillis() - startingMillis);
        try {
            PrintWriter out = new PrintWriter(new BufferedWriter(
                    new FileWriter("provPerformance.txt", true)));            
            out.println(codeLocation + " " + time);
            out.close();
        } catch (IOException e) {
            System.err.println("Problem in writing to the prov performance file.");
            e.printStackTrace();
        }
    }
}
