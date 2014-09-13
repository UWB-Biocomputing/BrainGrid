package edu.uwb.braingrid.workbench.utils;

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
}
