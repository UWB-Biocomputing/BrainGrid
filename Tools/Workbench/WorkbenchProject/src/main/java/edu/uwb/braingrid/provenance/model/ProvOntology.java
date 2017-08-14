package edu.uwb.braingrid.provenance.model;
/////////////////CLEANED
/**
 * Provides provenance definition URIs for building provenance RDF models.
 *
 * NOTE: This class was designed to provide static functions and fields.
 *
 * @author Del Davis
 * @version 0.1
 */
public class ProvOntology {

    /* Namespaces */
    // see: RDF-CONCEPTS @ http://www.w3.org/TR/prov-o/#bib-RDF-CONCEPTS
    private static final String RDF_NS
            = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    // see: RDF-CONCEPTS @ http://www.w3.org/TR/prov-o/#bib-RDF-CONCEPTS
    private static final String RDFS_NS
            = "http://www.w3.org/2000/01/rdf-schema#";
    // see: XMLSCHEMA11-2 @ http://www.w3.org/TR/prov-o/#bib-XMLSCHEMA11-2
    private static final String XSD_NS = "http://www.w3.org/2000/10/XMLSchema#";
    // see: OWL2-OVERVIEW @ http://www.w3.org/TR/prov-o/#bib-OWL2-OVERVIEW
    private static final String OWL_NS = "http://www.w3.org/2002/07/owl#";
    // PROV-DM @ http://www.w3.org/TR/prov-o/#bib-PROV-DM
    private static final String PROV_NS = "http://www.w3.org/ns/prov#";

    /* Prefixes (used for abbreviating above namespaces in output)*/
    private static final String RDF_PREFIX = "rdf:";
    private static final String RDFS_PREFIX = "rdfs:";
    private static final String XSD_PREFIX = "xsd:";
    private static final String OWL_PREFIX = "owl:";
    private static final String PROV_PREFIX = "prov:";

    /* RDF Property (limited to those commonly used with prov) */
    // see: http://www.w3.org/1999/02/22-rdf-syntax-ns#type
    private static final String RDF_TYPE_PROPERTY = "type";
    // see: http://www.w3.org/2000/01/rdf-schema#label
    private static final String RDFS_LABEL_PROPERTY = "label";


    /* Starting Point Classes */
    // see: http://www.w3.org/TR/prov-o/#Entity
    private static final String ENTITY = "Entity";
    // see: http://www.w3.org/TR/prov-o/#Activity
    private static final String ACTIVITY = "Activity";
    // see: http://www.w3.org/TR/prov-o/#Agent
    private static final String AGENT = "Agent";

    /* Starting Point Properties */
    // see: http://www.w3.org/TR/prov-o/#wasGeneratedBy
    private static final String WAS_GENERATED_BY = "wasGeneratedBy";
    // see: http://www.w3.org/TR/prov-o/#wasDerivedFrom
    private static final String WAS_DERIVED_FROM = "wasDerivedFrom";
    // see: http://www.w3.org/TR/prov-o/#wasAttributedTo
    private static final String WAS_ATTRIBUTED_TO = "wasAttributedTo";
    // see: http://www.w3.org/TR/prov-o/#startedAtTime
    private static final String STARTED_AT_TIME = "startedAtTime";
    // see: http://www.w3.org/TR/prov-o/#used
    private static final String USED = "used";
    // see: http://www.w3.org/TR/prov-o/#wasInformedBy
    private static final String WAS_INFORMED_BY = "wasInformedBy";
    // see: http://www.w3.org/TR/prov-o/#endedAtTime
    private static final String ENDED_AT_TIME = "endedAtTime";
    // see: http://www.w3.org/TR/prov-o/#wasAssociatedWith
    private static final String WAS_ASSOCIATED_WITH = "wasAssociatedWith";
    // see: http://www.w3.org/TR/prov-o/#actedOnBehalfOf
    private static final String ACTED_ON_BEHALF_OF = "actedOnBehalfOf";

    /* Expaded Classes */
    // see: http://www.w3.org/TR/prov-o/#Collection
    private static final String COLLECTION = "Collection";
    // see: http://www.w3.org/TR/prov-o/#EmptyCollection
    private static final String EMPTY_COLLECTION = "EmptyCollection";
    // see: http://www.w3.org/TR/prov-o/#Bundle
    private static final String BUNDLE = "Bundle";
    // see: http://www.w3.org/TR/prov-o/#Person
    private static final String PERSON = "Person";
    // see: http://www.w3.org/TR/prov-o/#SoftwareAgent
    private static final String SOFTWARE_AGENT = "SoftwareAgent";
    // see: http://www.w3.org/TR/prov-o/#Organization
    private static final String ORGANIZATION = "Organization";
    // see: http://www.w3.org/TR/prov-o/#Location
    private static final String LOCATION = "Location";

    /* Expanded Properties */
    // see: http://www.w3.org/TR/prov-o/#alternateOf
    private static final String ALTERNATE_OF = "alternateOf";
    // see: http://www.w3.org/TR/prov-o/#specializationOf
    private static final String SPECIALIZATION_OF = "specializationOf";
    // see: http://www.w3.org/TR/prov-o/#generatedAtTime
    private static final String GENERATED_AT_TIME = "generatedAtTime";
    // see: http://www.w3.org/TR/prov-o/#hadPrimarySource
    private static final String HAD_PRIMARY_SOURCE = "hadPrimarySource";
    // see: http://www.w3.org/TR/prov-o/#value
    private static final String VALUE = "value";
    // see: http://www.w3.org/TR/prov-o/#wasQuotedFrom
    private static final String WAS_QUOTED_FROM = "wasQuotedFrom";
    // see: http://www.w3.org/TR/prov-o/#wasRevisionOf
    private static final String WAS_REVISION_OF = "wasRevisionOf";
    // see: http://www.w3.org/TR/prov-o/#invalidatedAtTime
    private static final String INVALIDATED_AT_TIME = "invalidatedAtTime";
    // see: http://www.w3.org/TR/prov-o/#wasInvalidatedBy
    private static final String WAS_INVALIDATED_BY = "wasInvalidatedBy";
    // see: http://www.w3.org/TR/prov-o/#hadMember
    private static final String HAD_MEMBER = "hadMember";
    // see: http://www.w3.org/TR/prov-o/#wasStartedBy
    private static final String WAS_STARTED_BY = "wasStartedBy";
    // see: http://www.w3.org/TR/prov-o/#wasEndedBy
    private static final String WAS_ENDED_BY = "wasEndedBy";
    // see: http://www.w3.org/TR/prov-o/#invalidated
    private static final String INVALIDATED = "invalidated";
    // see: http://www.w3.org/TR/prov-o/#influenced
    private static final String INFLUENCED = "influenced";
    // see: http://www.w3.org/TR/prov-o/#atLocation
    private static final String AT_LOCATION = "atLocation";
    // see: http://www.w3.org/TR/prov-o/#generated
    private static final String GENERATED = "generated";

    /* Qualified Classes */
    // see: http://www.w3.org/TR/prov-o/#Influence 
    private static final String INFLUENCE = "Influence";
    // see: http://www.w3.org/TR/prov-o/#EntityInfluence 
    private static final String ENTITY_INFLUENCE = "EntityInfluence";
    // see: http://www.w3.org/TR/prov-o/#Usage  
    private static final String USAGE = "Usage";
    // see: http://www.w3.org/TR/prov-o/#Start   
    private static final String START = "Start";
    // see: http://www.w3.org/TR/prov-o/#End   
    private static final String END = "End";
    // see: http://www.w3.org/TR/prov-o/#Derivation   
    private static final String DERIVATION = "Derivation";
    // see: http://www.w3.org/TR/prov-o/#PrimarySource  
    private static final String PRIMARY_SOURCE = "PrimarySource";
    // see: http://www.w3.org/TR/prov-o/#Quotation   
    private static final String QUOTATION = "Quotation";
    // see: http://www.w3.org/TR/prov-o/#Revision   
    private static final String REVISION = "Revision";
    // see: http://www.w3.org/TR/prov-o/#ActivityInfluence   
    private static final String ACTIVITY_INFLUENCE = "ActivityInfluence";
    // see: http://www.w3.org/TR/prov-o/#Generation  
    private static final String GENERATION = "Generation";
    // see: http://www.w3.org/TR/prov-o/#Communication   
    private static final String COMMUNICATION = "Communication";
    // see: http://www.w3.org/TR/prov-o/#Invalidation   
    private static final String INVALIDATION = "Invalidation";
    // see: http://www.w3.org/TR/prov-o/#AgentInfluence   
    private static final String AGENT_INFLUENCE = "AgentInfluence";
    // see: http://www.w3.org/TR/prov-o/#Attribution   
    private static final String ATTRIBUTION = "Attribution";
    // see: http://www.w3.org/TR/prov-o/#Association   
    private static final String ASSOCIATION = "Association";
    // see: http://www.w3.org/TR/prov-o/#Plan   
    private static final String PLAN = "Plan";
    // see: http://www.w3.org/TR/prov-o/#Delegation   
    private static final String DELEGATION = "Delegation";
    // see: http://www.w3.org/TR/prov-o/#InstantaneousEvent   
    private static final String INSTANTATANEOUS_EVENT = "InstantaneousEvent";
    // see: http://www.w3.org/TR/prov-o/#Role 
    private static final String ROLE = "Role";

    /* Qualified Properties */
    // see: http://www.w3.org/TR/prov-o/#wasInfluencedBy
    private static final String WAS_INFLUENCED_BY = "wasInfluencedBy";
    // see: http://www.w3.org/TR/prov-o/#qualifiedInfluence
    private static final String QUALIFIED_INFLUENCE = "qualifiedInfluence";
    // see: http://www.w3.org/TR/prov-o/#qualifiedGeneration 
    private static final String QUALIFIED_GENERATION = "qualifiedGeneration";
    // see: http://www.w3.org/TR/prov-o/#qualifiedDerivation 
    private static final String QUALIFIED_DERIVATION = "qualifiedDerivation";
    // see: http://www.w3.org/TR/prov-o/#qualifiedPrimarySource 
    private static final String QUALIFIED_PRIMARY_SOURCE
            = "qualifiedPrimarySource";
    // see: http://www.w3.org/TR/prov-o/#qualifiedQuotation 
    private static final String QUALIFIED_QUOTATION = "qualifiedQuotation";
    // see: http://www.w3.org/TR/prov-o/#qualifiedRevision
    private static final String QUALIFIED_REVISION = "qualifiedRevision";
    // see: http://www.w3.org/TR/prov-o/#qualifiedAttribution 
    private static final String QAULIFIED_ATTRIBUTION = "qualifiedAttribution";
    // see: http://www.w3.org/TR/prov-o/#qualifiedInvalidation 
    private static final String QAULIFIED_INVALIDATION = "qualifiedInvalidation";
    // see: http://www.w3.org/TR/prov-o/#qualifiedStart 
    private static final String QUALIFIED_START = "qualifiedStart";
    // see: http://www.w3.org/TR/prov-o/#qualifiedUsage 
    private static final String QUALIFIED_USAGE = "qualifiedUsage";
    // see: http://www.w3.org/TR/prov-o/#qualifiedCommunication 
    private static final String QUALIFIED_COMMUNICATION
            = "qualifiedCommunication";
    // see: http://www.w3.org/TR/prov-o/#qualifiedAssociation 
    private static final String QUALIFIED_ASSOCIATION = "qualifiedAssociation";
    // see: http://www.w3.org/TR/prov-o/#qualifiedEnd
    private static final String QUALIFIED_END = "qualifiedEnd";
    // see: http://www.w3.org/TR/prov-o/#qualifiedDelegation 
    private static final String QUALIFIED_DELEGATION = "qualifiedDelegation";
    // see: http://www.w3.org/TR/prov-o/#influencer 
    private static final String INFLUENCER = "influencer";
    // see: http://www.w3.org/TR/prov-o/#entity 
    private static final String ENTITY__QUALIFIED_ = "entity";
    // see: http://www.w3.org/TR/prov-o/#hadUsage 
    private static final String HAD_USAGE = "hadUsage";
    // see: http://www.w3.org/TR/prov-o/#hadGeneration 
    private static final String HAD_GENERATION = "hadGeneration";
    // see: http://www.w3.org/TR/prov-o/#activity 
    private static final String ACTIVITY__QUALIFIED_ = "activity";
    // see: http://www.w3.org/TR/prov-o/#agent 
    private static final String AGENT__QUALIFIED_ = "agent";
    // see: http://www.w3.org/TR/prov-o/#hadPlan 
    private static final String HAD_PLAN = "hadPlan";
    // see: http://www.w3.org/TR/prov-o/#hadActivity 
    private static final String HAD_ACTIVITY = "hadActivity";
    // see: http://www.w3.org/TR/prov-o/#atTime 
    private static final String AT_TIME = "atTime";
    // see: http://www.w3.org/TR/prov-o/#hadRole
    private static final String HAD_ROLE = "hadRole";
    // translations from prov predicate to english sentence-part
    private static final String RDF_TYPE_TRANSLATION_BEFORE_CONSONATE = "is a";
    private static final String RDF_TYPE_TRANSLATION_BEFORE_VOWEL = "is an";
    private static final String AT_TIME_TRANSLATION = "happened on";
    private static final String ASSOCIATION_TRANSLATION = "was responsible for";
    private static final String WAS_ASSOCIATED_WITH_TRANSLATION = "was associated with";
    private static final String RDFS_LABEL_TRANSLATION = "is labeled as";
    private static final String WAS_DERIVED_FROM_TRANSLATION = "was derived from";
    private static final String ENDED_AT_TIME_TRANSLATION = "ended at";
    private static final String STARTED_AT_TIME_TRANSLATION = "started at";
    private static final String USED_TRANSLATION = "used";
    private static final String GENERATED_TRANSLATION = "generated";
    private static final String WAS_GENERATED_BY_TRANSLATION = "was generated by";

    /* Namespace Getters */
    // see: RDF-CONCEPTS @ http://www.w3.org/TR/prov-o/#bib-RDF-CONCEPTS
    public static String getRDFNameSpaceURI() {
        return RDF_NS;
    }

    // see: RDF-CONCEPTS @ http://www.w3.org/TR/prov-o/#bib-RDF-CONCEPTS
    public static String getRDFSNameSpaceURI() {
        return RDFS_NS;
    }

    // see: XMLSCHEMA11-2 @ http://www.w3.org/TR/prov-o/#bib-XMLSCHEMA11-2
    public static String getXSDNameSpaceURI() {
        return XSD_NS;
    }

    // see: OWL2-OVERVIEW @ http://www.w3.org/TR/prov-o/#bib-OWL2-OVERVIEW
    public static String getOWLNameSpaceURI() {
        return OWL_NS;
    }

    // PROV-DM @ http://www.w3.org/TR/prov-o/#bib-PROV-DM
    public static String getPROVNameSpaceURI() {
        return PROV_NS;
    }

    /**
     * Provides prefix for abbreviating RDF namespace in model output
     *
     * @return prefix for abbreviating RDF namespace in model output
     */
    public static String getRDFPrefix() {
        return RDF_PREFIX;
    }

    /**
     * Provides prefix for abbreviating XSD namespace in model output
     *
     * @return prefix for abbreviating XSD namespace in model output
     */
    public static String getXSDPrefix() {
        return XSD_PREFIX;
    }

    /**
     * Provides prefix for abbreviating OWL namespace in model output
     *
     * @return prefix for abbreviating OWL namespace in model output
     */
    public static String getOWLPrefix() {
        return OWL_PREFIX;
    }

    /**
     * Provides prefix for abbreviating PROV namespace in model output
     *
     * @return prefix for abbreviating PROV namespace in model output
     */
    public static String getProvPrefix() {
        return PROV_PREFIX;
    }

    /* RDF Property Getter (limited to those commonly used with prov) */
    public static String getRDFTypePrefixedURI() {
        return RDFS_PREFIX + RDF_TYPE_PROPERTY;
    }

    /* RDF Property Getter (limited to those commonly used with prov) */
    public static String getRDFTypeFullURI() {
        return RDFS_NS + RDF_TYPE_PROPERTY;
    }

    /* RDFS Property Getter (limited to those commonly used with prov) */
    public static String getRDFSLabelPrefixedURI() {
        return RDFS_PREFIX + RDFS_LABEL_PROPERTY;
    }

    /* RDFS Property Getter (limited to those commonly used with prov) */
    public static String getRDFSLabelFullURI() {
        return RDFS_NS + RDFS_LABEL_PROPERTY;
    }

    /* Starting Point Classes */
    // see: http://www.w3.org/TR/prov-o/#Entity
    public static String getEntityStartingPointClassPrefixedURI() {
        return PROV_PREFIX + ENTITY;
    }

    // see: http://www.w3.org/TR/prov-o/#Entity
    public static String getEntityStartingPointClassFullURI() {
        return PROV_NS + ENTITY;
    }

    // see: http://www.w3.org/TR/prov-o/#Activity
    public static String getActivityStartingPointClassPrefixedURI() {
        return PROV_PREFIX + ACTIVITY;
    }

    // see: http://www.w3.org/TR/prov-o/#Activity
    public static String getActivityStartingPointClassFullURI() {
        return PROV_NS + ACTIVITY;
    }

    // see: http://www.w3.org/TR/prov-o/#Agent
    public static String getAgentStartingPointClassPrefixedURI() {
        return PROV_PREFIX + AGENT;
    }

    // see: http://www.w3.org/TR/prov-o/#Agent
    public static String getAgentStartingPointClassFullURI() {
        return PROV_NS + AGENT;
    }

    /* Starting Point Properties */
    // see: http://www.w3.org/TR/prov-o/#wasGeneratedBy
    public static String getWasGeneratedByStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_GENERATED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasGeneratedBy
    public static String getWasGeneratedByStartingPointPropertyFullURI() {
        return PROV_NS + WAS_GENERATED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasDerivedFrom
    public static String getWasDerivedFromStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_DERIVED_FROM;
    }

    // see: http://www.w3.org/TR/prov-o/#wasDerivedFrom
    public static String getWasDerivedFromStartingPointPropertyFullURI() {
        return PROV_NS + WAS_DERIVED_FROM;
    }

    // see: http://www.w3.org/TR/prov-o/#wasAttributedTo
    public static String getWasAttributedToStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_ATTRIBUTED_TO;
    }

    // see: http://www.w3.org/TR/prov-o/#wasAttributedTo
    public static String getWasAttributedToStartingPointPropertyFullURI() {
        return PROV_NS + WAS_ATTRIBUTED_TO;
    }

    // see: http://www.w3.org/TR/prov-o/#startedAtTime
    public static String getStartedAtTimeStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + STARTED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#startedAtTime
    public static String getStartedAtTimeStartingPointPropertyFullURI() {
        return PROV_NS + STARTED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#used
    public static String getUsedStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + USED;
    }

    // see: http://www.w3.org/TR/prov-o/#used
    public static String getUsedStartingPointPropertyFullURI() {
        return PROV_NS + USED;
    }

    // see: http://www.w3.org/TR/prov-o/#wasInformedBy
    public static String getWasInformedByStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_INFORMED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasInformedBy
    public static String getWasInformedByStartingPointPropertyFullURI() {
        return PROV_NS + WAS_INFORMED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#endedAtTime
    public static String getEndedAtTimeStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + ENDED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#endedAtTime
    public static String getEndedAtTimeStartingPointPropertyFullURI() {
        return PROV_NS + ENDED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#wasAssociatedWith
    public static String getWasAssociatedWithStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_ASSOCIATED_WITH;
    }

    // see: http://www.w3.org/TR/prov-o/#wasAssociatedWith
    public static String getWasAssociatedWithStartingPointPropertyFullURI() {
        return PROV_NS + WAS_ASSOCIATED_WITH;
    }

    // see: http://www.w3.org/TR/prov-o/#actedOnBehalfOf
    public static String getActedOnBehalfOfStartingPointPropertyPrefixedURI() {
        return PROV_PREFIX + ACTED_ON_BEHALF_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#actedOnBehalfOf
    public static String getActedOnBehalfOfStartingPointPropertyFullURI() {
        return PROV_NS + ACTED_ON_BEHALF_OF;
    }

    /* Expaded Classes */
    // see: http://www.w3.org/TR/prov-o/#Collection
    public static String getCollectionExpandedClassPrefixedURI() {
        return PROV_PREFIX + COLLECTION;
    }

    // see: http://www.w3.org/TR/prov-o/#Collection
    public static String getCollectionExpandedClassFullURI() {
        return PROV_NS + COLLECTION;
    }

    // see: http://www.w3.org/TR/prov-o/#EmptyCollection
    public static String getEmptyCollectionExpandedClassPrefixedURI() {
        return PROV_PREFIX + EMPTY_COLLECTION;
    }

    // see: http://www.w3.org/TR/prov-o/#EmptyCollection
    public static String getEmptyCollectionExpandedClassFullURI() {
        return PROV_NS + EMPTY_COLLECTION;
    }

    // see: http://www.w3.org/TR/prov-o/#Bundle
    public static String getBundleExpandedClassPrefixedURI() {
        return PROV_PREFIX + BUNDLE;
    }

    // see: http://www.w3.org/TR/prov-o/#Bundle
    public static String getBundleExpandedClassFullURI() {
        return PROV_NS + BUNDLE;
    }

    // see: http://www.w3.org/TR/prov-o/#Person
    public static String getPersonExpandedClassPrefixedURI() {
        return PROV_PREFIX + PERSON;
    }

    // see: http://www.w3.org/TR/prov-o/#Person
    public static String getPersonExpandedClassFullURI() {
        return PROV_NS + PERSON;
    }

    // see: http://www.w3.org/TR/prov-o/#SoftwareAgent
    public static String getSoftwareAgentExpandedClassPrefixedURI() {
        return PROV_PREFIX + SOFTWARE_AGENT;
    }

    // see: http://www.w3.org/TR/prov-o/#SoftwareAgent
    public static String getSoftwareAgentExpandedClassFullURI() {
        return PROV_NS + SOFTWARE_AGENT;
    }

    // see: http://www.w3.org/TR/prov-o/#Organization
    public static String getOrganizationExpandedClassPrefixedURI() {
        return PROV_PREFIX + ORGANIZATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Organization
    public static String getOrganizationExpandedClassFullURI() {
        return PROV_NS + ORGANIZATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Location
    public static String getLocationExpandedClassPrefixedURI() {
        return PROV_PREFIX + LOCATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Location
    public static String getLocationExpandedClassFullURI() {
        return PROV_NS + LOCATION;
    }

    /* Expanded Properties */
    // see: http://www.w3.org/TR/prov-o/#alternateOf
    public static String getAlternateOfExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + ALTERNATE_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#alternateOf
    public static String getAlternateOfExpandedPropertyFullURI() {
        return PROV_NS + ALTERNATE_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#specializationOf
    public static String getSpecializationOfExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + SPECIALIZATION_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#specializationOf
    public static String getSpecializationOfExpandedPropertyFullURI() {
        return PROV_NS + SPECIALIZATION_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#generatedAtTime
    public static String getGeneratedAtTimeExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + GENERATED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#generatedAtTime
    public static String getGeneratedAtTimeExpandedPropertyFullURI() {
        return PROV_NS + GENERATED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#hadPrimarySource
    public static String getHadPrimarySourceExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#hadPrimarySource
    public static String getHadPrimarySourceExpandedPropertyFullURI() {
        return PROV_NS + HAD_PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#value
    public static String getValueExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + VALUE;
    }

    // see: http://www.w3.org/TR/prov-o/#value
    public static String getValueExpandedPropertyFullURI() {
        return PROV_NS + VALUE;
    }

    // see: http://www.w3.org/TR/prov-o/#wasQuotedFrom
    public static String getWasQuotedFromExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_QUOTED_FROM;
    }

    // see: http://www.w3.org/TR/prov-o/#wasQuotedFrom
    public static String getWasQuotedFromExpandedPropertyFullURI() {
        return PROV_NS + WAS_QUOTED_FROM;
    }

    // see: http://www.w3.org/TR/prov-o/#wasRevisionOf
    public static String getWasRevisionOfExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_REVISION_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#wasRevisionOf
    public static String getWasRevisionOfExpandedPropertyFullURI() {
        return PROV_NS + WAS_REVISION_OF;
    }

    // see: http://www.w3.org/TR/prov-o/#invalidatedAtTime
    public static String getInvalidatedAtTimeExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + INVALIDATED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#invalidatedAtTime
    public static String getInvalidatedAtTimeExpandedPropertyFullURI() {
        return PROV_NS + INVALIDATED_AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#wasInvalidatedBy
    public static String getWasInvalidatedByExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_INVALIDATED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasInvalidatedBy
    public static String getWasInvalidatedByExpandedPropertyFullURI() {
        return PROV_NS + WAS_INVALIDATED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#hadMember
    public static String getHadMemberExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_MEMBER;
    }

    // see: http://www.w3.org/TR/prov-o/#hadMember
    public static String getHadMemberExpandedPropertyFullURI() {
        return PROV_NS + HAD_MEMBER;
    }

    // see: http://www.w3.org/TR/prov-o/#wasStartedBy
    public static String getWasStartedByExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_STARTED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasStartedBy
    public static String getWasStartedByExpandedPropertyFullURI() {
        return PROV_NS + WAS_STARTED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasEndedBy
    public static String getWasEndedByExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_ENDED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasEndedBy
    public static String getWasEndedByExpandedPropertyFullURI() {
        return PROV_NS + WAS_ENDED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#invalidated
    public static String getInvalidatedExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + INVALIDATED;
    }

    // see: http://www.w3.org/TR/prov-o/#invalidated
    public static String getInvalidatedExpandedPropertyFullURI() {
        return PROV_NS + INVALIDATED;
    }

    // see: http://www.w3.org/TR/prov-o/#influenced
    public static String getInfluencedExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + INFLUENCED;
    }

    // see: http://www.w3.org/TR/prov-o/#influenced
    public static String getInfluencedExpandedPropertyFullURI() {
        return PROV_NS + INFLUENCED;
    }

    // see: http://www.w3.org/TR/prov-o/#atLocation
    public static String getAtLocationExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + AT_LOCATION;
    }

    // see: http://www.w3.org/TR/prov-o/#atLocation
    public static String getAtLocationExpandedPropertyFullURI() {
        return PROV_NS + AT_LOCATION;
    }

    // see: http://www.w3.org/TR/prov-o/#generated
    public static String getGeneratedExpandedPropertyPrefixedURI() {
        return PROV_PREFIX + GENERATED;
    }

    // see: http://www.w3.org/TR/prov-o/#generated
    public static String getGeneratedExpandedPropertyFullURI() {
        return PROV_NS + GENERATED;
    }

    /* Qualified Classes */
    // see: http://www.w3.org/TR/prov-o/#Influence 
    public static String getInfluenceQualifiedClassPrefixedURI() {
        return PROV_PREFIX + INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#Influence
    public static String getInfluenceQualifiedClassFullURI() {
        return PROV_NS + INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#EntityInfluence 
    public static String getEntityInfluenceQualifiedClassPrefixedURI() {
        return PROV_PREFIX + ENTITY_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#EntityInfluence
    public static String getEntityInfluenceQualifiedClassFullURI() {
        return PROV_NS + ENTITY_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#Usage  
    public static String getUsageQualifiedClassPrefixedURI() {
        return PROV_PREFIX + USAGE;
    }

    // see: http://www.w3.org/TR/prov-o/#Usage
    public static String getUsageQualifiedClassFullURI() {
        return PROV_NS + USAGE;
    }

    // see: http://www.w3.org/TR/prov-o/#Start   
    public static String getStartQualifiedClassPrefixedURI() {
        return PROV_PREFIX + START;
    }

    // see: http://www.w3.org/TR/prov-o/#Start
    public static String getStartQualifiedClassFullURI() {
        return PROV_NS + START;
    }

    // see: http://www.w3.org/TR/prov-o/#End   
    public static String getEndQualifiedClassPrefixedURI() {
        return PROV_PREFIX + END;
    }

    // see: http://www.w3.org/TR/prov-o/#End
    public static String getEndQualifiedClassFullURI() {
        return PROV_NS + END;
    }

    // see: http://www.w3.org/TR/prov-o/#Derivation   
    public static String getDerivationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + DERIVATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Derivation
    public static String getDerivationQualifiedClassFullURI() {
        return PROV_NS + DERIVATION;
    }

    // see: http://www.w3.org/TR/prov-o/#PrimarySource  
    public static String getPrimarySourceQualifiedClassPrefixedURI() {
        return PROV_PREFIX + PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#PrimarySource
    public static String getPrimarySourceQualifiedClassFullURI() {
        return PROV_NS + PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#Quotation   
    public static String getQuotationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + QUOTATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Quotation 
    public static String getQuotationQualifiedClassFullURI() {
        return PROV_NS + QUOTATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Revision   
    public static String getRevisionQualifiedClassPrefixedURI() {
        return PROV_PREFIX + REVISION;
    }

    // see: http://www.w3.org/TR/prov-o/#Revision
    public static String getRevisionQualifiedClassFullURI() {
        return PROV_NS + REVISION;
    }

    // see: http://www.w3.org/TR/prov-o/#ActivityInfluence   
    public static String getActivityInfluenceQualifiedClassPrefixedURI() {
        return PROV_PREFIX + ACTIVITY_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#ActivityInfluence
    public static String getActivityInfluenceQualifiedClassFullURI() {
        return PROV_NS + ACTIVITY_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#Generation  
    public static String getGenerationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Generation
    public static String getGenerationQualifiedClassFullURI() {
        return PROV_NS + GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Communication   
    public static String getCommunicationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + COMMUNICATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Communication
    public static String getCommunicationQualifiedClassFullURI() {
        return PROV_NS + COMMUNICATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Invalidation   
    public static String getInvalidationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + INVALIDATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Invalidation
    public static String getInvalidationQualifiedClassFullURI() {
        return PROV_NS + INVALIDATION;
    }

    // see: http://www.w3.org/TR/prov-o/#AgentInfluence   
    public static String getAgentInfluenceQualifiedClassPrefixedURI() {
        return PROV_PREFIX + AGENT_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#AgentInfluence
    public static String getAgentInfluenceQualifiedClassFullURI() {
        return PROV_NS + AGENT_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#Attribution   
    public static String getAttributionQualifiedClassPrefixedURI() {
        return PROV_PREFIX + ATTRIBUTION;
    }

    // see: http://www.w3.org/TR/prov-o/#Attribution
    public static String getAttributionQualifiedClassFullURI() {
        return PROV_NS + ATTRIBUTION;
    }

    // see: http://www.w3.org/TR/prov-o/#Association   
    public static String getAssociationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + ASSOCIATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Association
    public static String getAssociationQualifiedClassFullURI() {
        return PROV_NS + ASSOCIATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Plan   
    public static String getPlanQualifiedClassPrefixedURI() {
        return PROV_PREFIX + PLAN;
    }

    // see: http://www.w3.org/TR/prov-o/#Plan
    public static String getPlanQualifiedClassFullURI() {
        return PROV_NS + PLAN;
    }

    // see: http://www.w3.org/TR/prov-o/#Delegation   
    public static String getDelegationQualifiedClassPrefixedURI() {
        return PROV_PREFIX + DELEGATION;
    }

    // see: http://www.w3.org/TR/prov-o/#Delegation
    public static String getDelegationQualifiedClassFullURI() {
        return PROV_NS + DELEGATION;
    }

    // see: http://www.w3.org/TR/prov-o/#InstantaneousEvent   
    public static String getInstantaneousEventQualifiedClassPrefixedURI() {
        return PROV_PREFIX + INSTANTATANEOUS_EVENT;
    }

    // see: http://www.w3.org/TR/prov-o/#InstantaneousEvent
    public static String getInstantaneousEventQualifiedClassFullURI() {
        return PROV_NS + INSTANTATANEOUS_EVENT;
    }

    // see: http://www.w3.org/TR/prov-o/#Role 
    public static String getRoleQualifiedClassPrefixedURI() {
        return PROV_PREFIX + ROLE;
    }

    // see: http://www.w3.org/TR/prov-o/#Role
    public static String getRoleQualifiedClassFullURI() {
        return PROV_NS + ROLE;
    }

    /* Qualified Properties */
    // see: http://www.w3.org/TR/prov-o/#wasInfluencedBy
    public static String getWasInfluencedByQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + WAS_INFLUENCED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#wasInfluencedBy
    public static String getWasInfluencedByQualifiedPropertyFullURI() {
        return PROV_NS + WAS_INFLUENCED_BY;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedInfluence
    public static String getQualifiedInfluenceQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedInfluence
    public static String getQualifiedInfluenceQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_INFLUENCE;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedGeneration 
    public static String getQualifiedGenerationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedGeneration
    public static String getQualifiedGenerationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedDerivation 
    public static String getQualifiedDerivationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_DERIVATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedDerivation
    public static String getQualifiedDerivationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_DERIVATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedPrimarySource 
    public static String getQualifiedPrimarySourceQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedPrimarySource
    public static String getQualifiedPrimarySourceQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_PRIMARY_SOURCE;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedQuotation 
    public static String getQualifiedQuotationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_QUOTATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedQuotation
    public static String getQualifiedQuotationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_QUOTATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedRevision
    public static String getQualifiedRevisionQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_REVISION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedRevision
    public static String getQualifiedRevisionQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_REVISION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedAttribution 
    public static String getQualifiedAttributionQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QAULIFIED_ATTRIBUTION;
    }

    public static String getQualifiedAttributionQualifiedPropertyFullURI() {
        return PROV_NS + QAULIFIED_ATTRIBUTION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedInvalidation 
    public static String getQualifiedInvalidationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QAULIFIED_INVALIDATION;
    }

    public static String getQualifiedInvalidationQualifiedPropertyFullURI() {
        return PROV_NS + QAULIFIED_INVALIDATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedStart 
    public static String getQualifiedStartQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_START;
    }

    public static String getQualifiedStartQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_START;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedUsage 
    public static String getQualifiedUsageQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_USAGE;
    }

    public static String getQualifiedUsageQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_USAGE;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedCommunication 
    public static String getQualifiedCommunicationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_COMMUNICATION;
    }

    public static String getQualifiedCommunicationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_COMMUNICATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedAssociation 
    public static String getQualifiedAssociationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_ASSOCIATION;
    }

    public static String getQualifiedAssociationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_ASSOCIATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedEnd
    public static String getQualifiedEndQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_END;
    }

    public static String getQualifiedEndQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_END;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedDelegation 
    public static String getQualifiedDelegationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + QUALIFIED_DELEGATION;
    }

    // see: http://www.w3.org/TR/prov-o/#qualifiedDelegation
    public static String getQualifiedDelegationQualifiedPropertyFullURI() {
        return PROV_NS + QUALIFIED_DELEGATION;
    }

    // see: http://www.w3.org/TR/prov-o/#influencer 
    public static String getInfluencerQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + INFLUENCER;
    }

    // see: http://www.w3.org/TR/prov-o/#influencer 
    public static String getInfluencerQualifiedPropertyFullURI() {
        return PROV_NS + INFLUENCER;
    }

    // see: http://www.w3.org/TR/prov-o/#entity 
    public static String getEntityQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + ENTITY__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#entity
    public static String getEntityQualifiedPropertyFullURI() {
        return PROV_NS + ENTITY__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#hadUsage 
    public static String getHadUsageQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_USAGE;
    }

    // see: http://www.w3.org/TR/prov-o/#hadUsage
    public static String getHadUsageQualifiedPropertyFullURI() {
        return PROV_NS + HAD_USAGE;
    }

    // see: http://www.w3.org/TR/prov-o/#hadGeneration 
    public static String getHadGenerationQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#hadGeneration
    public static String getHadGenerationQualifiedPropertyFullURI() {
        return PROV_NS + HAD_GENERATION;
    }

    // see: http://www.w3.org/TR/prov-o/#activity 
    public static String getActivityQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + ACTIVITY__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#activity
    public static String getActivityQualifiedPropertyFullURI() {
        return ACTIVITY__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#agent 
    public static String getAgentQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + AGENT__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#agent 
    public static String getAgentQualifiedPropertyFullURI() {
        return PROV_NS + AGENT__QUALIFIED_;
    }

    // see: http://www.w3.org/TR/prov-o/#hadPlan 
    public static String getHadPlanQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_PLAN;
    }

    // see: http://www.w3.org/TR/prov-o/#hadPlan 
    public static String getHadPlanQualifiedPropertyFullURI() {
        return PROV_NS + HAD_PLAN;
    }

    // see: http://www.w3.org/TR/prov-o/#hadActivity 
    public static String getHadActivityQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_ACTIVITY;
    }

    // see: http://www.w3.org/TR/prov-o/#hadActivity 
    public static String getHadActivityQualifiedPropertyFullURI() {
        return PROV_NS + HAD_ACTIVITY;
    }

    // see: http://www.w3.org/TR/prov-o/#atTime 
    public static String getAtTimeQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#atTime 
    public static String getAtTimeQualifiedPropertyFullURI() {
        return PROV_NS + AT_TIME;
    }

    // see: http://www.w3.org/TR/prov-o/#hadRole
    public static String getHadRoleQualifiedPropertyPrefixedURI() {
        return PROV_PREFIX + HAD_ROLE;
    }

    // see: http://www.w3.org/TR/prov-o/#hadRole
    public static String getHadRoleQualifiedPropertyFullURI() {
        return PROV_NS + HAD_ROLE;
    }

    /**
     * Provides a human-readable sentence-part for the predicate of a provenance
     * triple
     *
     * @param uri - The PROV-O URI for the predicate
     * @param objectStartWithVowel - Indicates whether or not the object of the
     * triple starts with a vowel
     * @return Human readable text describing the relationship between a subject
     * and object, based on the PROV-O predicate
     */
    public static String translatePredicate(String uri, boolean objectStartWithVowel) {
        String translation = uri;
        if (uri.equals(ProvOntology.getRDFTypeFullURI())) {
            if (!objectStartWithVowel) {
                translation = RDF_TYPE_TRANSLATION_BEFORE_CONSONATE;
            } else {
                translation = RDF_TYPE_TRANSLATION_BEFORE_VOWEL;
            }
        } else if (uri.equals(ProvOntology.getAtTimeQualifiedPropertyFullURI())) {
            translation = AT_TIME_TRANSLATION;
        } else if (uri.equals(ProvOntology.getAssociationQualifiedClassFullURI())) {
            translation = ASSOCIATION_TRANSLATION;
        } else if (uri.equals(ProvOntology.getWasAssociatedWithStartingPointPropertyFullURI())) {
            translation = WAS_ASSOCIATED_WITH_TRANSLATION;
        } else if (uri.equals(ProvOntology.getRDFSLabelFullURI())) {
            translation = RDFS_LABEL_TRANSLATION;
        } else if (uri.equals(ProvOntology.getWasDerivedFromStartingPointPropertyFullURI())) {
            translation = WAS_DERIVED_FROM_TRANSLATION;
        } else if (uri.equals(ProvOntology.getEndedAtTimeStartingPointPropertyFullURI())) {
            translation = ENDED_AT_TIME_TRANSLATION;
        } else if (uri.equals(ProvOntology.getStartedAtTimeStartingPointPropertyFullURI())) {
            translation = STARTED_AT_TIME_TRANSLATION;
        } else if (uri.equals(ProvOntology.getUsedStartingPointPropertyFullURI())) {
            translation = USED_TRANSLATION;
        } else if (uri.equals(ProvOntology.getGeneratedExpandedPropertyFullURI())) {
            translation = GENERATED_TRANSLATION;
        } else if (uri.equals(ProvOntology.getWasGeneratedByStartingPointPropertyFullURI())) {
            translation = WAS_GENERATED_BY_TRANSLATION;
        }
        return translation;
    }
}
