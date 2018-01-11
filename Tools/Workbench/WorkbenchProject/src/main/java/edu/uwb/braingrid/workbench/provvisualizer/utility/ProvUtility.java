package edu.uwb.braingrid.workbench.provvisualizer.utility;

public class ProvUtility {
    public static final String RDF_SYNTAX_PREFIX = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    public static final String RDF_SCHEME_PREFIX = "http://www.w3.org/2000/01/rdf-schema#";
    public static final String PROV_PREFIX = "http://www.w3.org/ns/prov#";
    public static final String RDF_TYPE = RDF_SYNTAX_PREFIX + "type";
    public static final String RDF_LABEL = RDF_SCHEME_PREFIX + "label";
    public static final String PROV_ACTIVITY = PROV_PREFIX + "Activity";
    public static final String PROV_SW_AGENT = PROV_PREFIX + "SoftwareAgent";
    public static final String PROV_ENTITY = PROV_PREFIX + "Entity";
    public static final String PROV_AT_LOCATION = PROV_PREFIX + "atLocation";
    public static final String PROV_WAS_GENERATED_BY = PROV_PREFIX + "wasGeneratedBy";
    public static final String PROV_USED = PROV_PREFIX + "used";
    public static final String PROV_GENERATED = PROV_PREFIX + "generated";
    public static final String PROV_WAS_ASSOCIATED_WITH = PROV_PREFIX + "wasAssociatedWith";
    public static final String PROV_WAS_DERIVED_FROM = PROV_PREFIX + "wasDerivedFrom";
    public static final String PROV_STARTED_AT_TIME = PROV_PREFIX + "startedAtTime";
    public static final String PROV_ENDED_AT_TIME = PROV_PREFIX + "endedAtTime";

    public static final String LABEL_COMMIT = "commit";
    public static final String COMMIT_URI_PREFIX = "https://github.com/UWB-Biocomputing/BrainGrid/commit/";

    public static String getCommitUri(String commitId){
        return COMMIT_URI_PREFIX + commitId;
    }
}
