package edu.uwb.braingrid.workbenchdashboard.utils;

import java.util.logging.Logger;

enum OSType {
	Windows,
	Linux,
	Mac,
	SunOS,
	FreeBSD,
	UNKNOWN
}

enum OS {
	Windows_XP,
	Windows_2003,
	Linux,
	Windows_2000,
	Mac_OS_X,
	Windows_98,
	SunOS,
	FreeBSD,
	Windows_NT,
	Windows_Me,
	Windows_10,
	UNKNOWN
}

public class SystemProperties {
	
	public static OS getOS() {
		if(OS_ == OS.UNKNOWN) {
			return OS_USER_DEF_;
		}
		return OS_;
	}
	
	public static OSType getOSType() {
		if(OS_TYPE_ == OSType.UNKNOWN) {
			return OS_TYPE_USER_DEF_;
		}
		return OS_TYPE_;
	}
	
	public static SystemProperties getSysProperties() {
		return INIT;
	}

	public static void setOSType(OSType osType) {
		OS_TYPE_USER_DEF_ = osType;
	}
	
	public static void setOS(OS os) {
		OS_USER_DEF_ = os;
	}
	
	private static final Logger LOG = Logger.getLogger(SystemProperties.class.getName());
	private static OSType OS_TYPE_ = OSType.UNKNOWN;
	private static OS OS_ = OS.UNKNOWN;
	private static OS OS_USER_DEF_ =  OS.UNKNOWN;
	private static OSType OS_TYPE_USER_DEF_ = OSType.UNKNOWN;
	private static SystemProperties INIT = new SystemProperties();
	
	private SystemProperties() {
		LOG.info("new SystemProperties");
		String os = System.getProperty("os.name");
//		System.getProperties().list(System.out);
		initOSTypeByString(os);
	}
	
	private static void initOSTypeByString(String os) {
		switch(os) {
		case "Windows 10":	informOSType("Windows 10", OS.Windows_10, OSType.Windows);
			break;
		case "Windows XP":	informOSType("Windows XP", OS.Windows_XP, OSType.Windows);
			break;
		case "Windows 2003": informOSType("Windows 2003", OS.Windows_2003, OSType.Windows);
			break;
		case "Windows NT": informOSType("Windows NT", OS.Windows_NT, OSType.Windows);
			break;
		case "Windows Me": informOSType("Windows NT", OS.Windows_Me, OSType.Windows);
			break;
		case "Linux": informOSType("Linux", OS.Linux, OSType.Linux);
			break;
		case "Mac OS X": informOSType("Mac OS X", OS.Mac_OS_X, OSType.Mac);
			break;
		case "SunOS": informOSType("SunOS", OS.SunOS, OSType.SunOS);
			break;
		case "FreeBSD": informOSType("FreeBSD", OS.FreeBSD, OSType.FreeBSD);
			break;
		}
	}
	
	private static OS informOSType(String name, OS osType, OSType os) {
		LOG.info("OS found - " + name);
		OS_ = osType;
		OS_TYPE_ = os;
		return osType;
	}
}
