package edu.uwb.braingrid.general;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LoggerHelperTest {
    @Test
    public void testMinLogLevel() {
        // Assures that this global var exists
        Assertions.assertEquals(LoggerHelper.MIN_LOG_LEVEL, LoggerHelper.MIN_LOG_LEVEL);
    }
}
