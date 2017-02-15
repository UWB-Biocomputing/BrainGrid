BaseTemplateConfig.xml
--------------------------
<?xml version="1.0" encoding="UTF-8"?>
<BaseTemplateConfig templatePath="BaseTemplates/BaseTemplateDefault.xml" allParamsClassesConfigFilePath="ParamsClassTemplateConfig/AllParamsClasses.xml">
    <ParamsClassesType name = "NeuronsParamsClasses" nodePath="/BGSimParams/ModelParams/NeuronsParams"/>
    <ParamsClassesType name = "SynapsesParamsClasses" nodePath="/BGSimParams/ModelParams/SynapsesParams"/>
    <ParamsClassesType name = "ConnectionsParamsClasses" nodePath="/BGSimParams/ModelParams/ConnectionsParams"/>
    <ParamsClassesType name = "LayoutParamsClasses" nodePath="/BGSimParams/ModelParams/LayoutParams"/>
</BaseTemplateConfig>

1. BaseTemplateConfig:templatePath: relative path to the base template which is used as a container for the param classes templates.
2. BaseTemplateConfig:allParamsClassesConfigFilePath: relative path to the params classes configuration file which contains all parameter class types, names and template file paths
3. ParamsClassesType:name: the param class type
4. ParamsClassesType:nodePath: indicate the node location in the base template file to place the param class template


AllParamsClasses.xml
--------------------------
<?xml version="1.0" encoding="UTF-8"?>
<AllParamsClasses
    xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'
    xsi:noNamespaceSchemaLocation='AllParamsClasses.xsd'>
    <NeuronsParamsClasses name="NeuronsParamsClasses" templateDirectory="ParamsClassTemplateConfig/NeuronsParamsClass/">
        <ParamsClass name="ALLLIFNeurons" templateFileName="AllLIFNeurons.xml"/>
        <ParamsClass name="NeuronsParamsClass1" templateFileName="NeuronsParamsClass1.xml"/>
        <ParamsClass name="NeuronsParamsClass2" templateFileName="NeuronsParamsClass2.xml"/>
    </NeuronsParamsClasses>
    <SynapsesParamsClasses name="SynapsesParamsClasses" templateDirectory="ParamsClassTemplateConfig/SynapsesParamsClass/">
        <ParamsClass name="AllDSSynapses" templateFileName="AllDSSynapses.xml"/>
        <ParamsClass name="SynapsesParamsClass1" templateFileName="SynapsesParamsClass1.xml"/>
        <ParamsClass name="SynapsesParamsClass2" templateFileName="SynapsesParamsClass2.xml"/>
    </SynapsesParamsClasses>
    <ConnectionsParamsClasses name="ConnectionsParamsClasses" templateDirectory="ParamsClassTemplateConfig/ConnectionsParamsClass/">
        <ParamsClass name="ConnGrowth" templateFileName="ConnGrowth.xml"/>
        <ParamsClass name="ConnectionsParamsClass1" templateFileName="ConnectionsParamsClass1.xml"/>
        <ParamsClass name="ConnectionsParamsClass2" templateFileName="ConnectionsParamsClass2.xml"/>
    </ConnectionsParamsClasses>
    <LayoutParamsClasses name="LayoutParamsClasses" templateDirectory="ParamsClassTemplateConfig/LayoutParamsClass/">
        <ParamsClass name="FixedLayout" templateFileName="FixedLayout.xml"/>
        <ParamsClass name="LayoutParamsClass1" templateFileName="LayoutParamsClass1.xml"/>
        <ParamsClass name="LayoutParamsClass2" templateFileName="LayoutParamsClass2.xml"/>
    </LayoutParamsClasses>
</AllParamsClasses>

1. templateDirectory: the directory to store all of the param class template files for a params class type
2. ParamsClass:name: the name of the param class
3. ParamsClass:templateFileName: name of the file in the templateDirecotory to store the param class template.
