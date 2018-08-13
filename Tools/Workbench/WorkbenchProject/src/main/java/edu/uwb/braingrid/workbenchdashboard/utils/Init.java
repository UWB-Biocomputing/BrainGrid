package edu.uwb.braingrid.workbenchdashboard.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.logging.Logger;

import edu.uwb.braingrid.workbench.SystemConfig;

public class Init {
	
	public static void init() {
			try {
				Init.initBaseTemplates();
				Init.initParamsClassTemplateConfig();
				Init.initRootFiles();
			} catch (IOException e) {
				LOG.severe("IO Exception: " + e.getMessage());
			}
	}

	private static void initBaseTemplates() throws IOException {

		String folderName = "BaseTemplates";
		File file = new File("./" + folderName);
		if(!file.exists()) {
			LOG.info("Making Folder: " + file.getAbsolutePath());
		    file.mkdirs();
		}
		
		Init.copyResourceToFile("/init/BaseTemplates/BaseTemplateDefault.xml", file.getAbsolutePath() + "/BaseTemplateDefault.xml");
	}
	
	private static void initParamsClassTemplateConfig() throws IOException {
		String root = "ParamsClassTemplateConfig";
		String connections = "ConnectionsParamsClass";
		String layout = "LayoutParamsClass";
		String neurons = "NeuronsParamsClass";
		String synapses = "SynapsesParamsClass";
		String[] subdirs = {connections, layout, neurons, synapses};
		String outputRootPath = "./" + root + "/";
		for(String dir : subdirs) {
			LOG.info("Making Folder: " + outputRootPath + dir);
			Init.mkdirs(outputRootPath + dir);
		}
		
		String resourceRootPath = "/init/" + root + "/";
		
		// Init ParamsClassTemplateConfig
		LOG.info("Initializing " + root + " resources");
		String[] paramsFiles = {"AllParamsClasses.xml", "AllParamsClasses.xsd"};
		Init.initFiles("/init/" + root, "./" + root, paramsFiles);
		
		// Init ConnectionsParamsClass
		LOG.info("Initializing " + connections + " resources");
		String[] connSubdirFiles = {"ConnGrowth.xml", "ConnStatic.xml"};
		Init.initFiles(resourceRootPath + connections, outputRootPath + connections, connSubdirFiles);
		
		// Init  "LayoutParamsClass";
		LOG.info("Initializing " + layout + " resources");
		String[] laySubdirFiles = {"DynamicLayout.xml", "FixedLayout.xml", "LayoutParamsClass1.xml"};
		Init.initFiles(resourceRootPath + layout, outputRootPath + layout, laySubdirFiles);
		
		// Init "NeuronsParamsClass";
		LOG.info("Initializing " + neurons + " resources");
		String[] neuSubdirFiles = {"AllIZHNeurons.xml", "AllLIFNeurons.xml"};
		Init.initFiles(resourceRootPath + neurons, outputRootPath + neurons, neuSubdirFiles);
		
		// Init "SynapsesParamsClass";
		LOG.info("Initializing " + synapses + " resources");
		String[] synSubdirFiles = {"AllDSSynapses.xml", "AllDynamicSTDPSynapses.xml", "AllSpikingSynapses.xml", "AllSTDPSynapses.xml"};
		Init.initFiles(resourceRootPath + synapses, outputRootPath + synapses, synSubdirFiles);
	}
	
	private static void initRootFiles() throws IOException {
		LOG.info("Initializing Root Files");
		String filename = SystemConfig.BASE_TEMPLATE_INFO_XML_File_URL;
		Init.copyResourceToFile("/init/" + filename, "./" + filename);
		Init.copyResourceToFile("/init/README.TXT", "./README.TXT");
	}
	
	
	
	private static void initFiles(String resourceFolder, String folderOutput, String[] subdirFiles) throws IOException {
			for(String file : subdirFiles) {
				Init.copyResourceToFile(resourceFolder + "/" + file, folderOutput + "/" + file);
			}
	}
	
	private static void mkdirs(String path) {
		File file = new File(path);
		if(!file.exists()) {
			LOG.info("Making Folder(s): " + path);
		    file.mkdirs();
		}
	}
	
	private static void copyResourceToFile(String resourcePath, String filePath) throws IOException  {
		LOG.info("Copying: " + resourcePath + " to: " + filePath);
		String url = "resources" + resourcePath;
		InputStream is = Init.class.getClassLoader().getResourceAsStream(url);
		Files.copy(is, Paths.get(filePath), StandardCopyOption.REPLACE_EXISTING);
	}

	  private static final Logger LOG = Logger.getLogger(Init.class.getName());
}
