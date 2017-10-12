package edu.uwb.braingrid.workbench.provvisualizer;

public class ProvVisGlobal {
    public static String RDF_SYNTAX_PREFIX = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    public static String RDF_SCHEME_PREFIX = "http://www.w3.org/2000/01/rdf-schema#";
    public static String PROV_PREFIX = "http://www.w3.org/ns/prov#";
    public static String RDF_TYPE = RDF_SYNTAX_PREFIX + "type";
    public static String RDF_LABEL = RDF_SCHEME_PREFIX + "label";
    public static String PROV_ACTIVITY = PROV_PREFIX + "Activity";
    public static String PROV_SW_AGENT = PROV_PREFIX + "SoftwareAgent";
    public static String PROV_ENTITY = PROV_PREFIX + "Entity";
    public static String PROV_WAS_GENERATED_BY = PROV_PREFIX + "wasGeneratedBy";
    public static String PROV_USED = PROV_PREFIX + "used";
    public static String PROV_GENERATED = PROV_PREFIX + "generated";
    public static String PROV_WAS_ASSOCIATED_WITH = PROV_PREFIX + "wasAssociatedWith";
    public static String PROV_WAS_DERIVED_FROM = PROV_PREFIX + "wasDerivedFrom";
}
