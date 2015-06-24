package edu.uwb.braingrid.provenance;
/////////////////CLEANED
import com.hp.hpl.jena.rdf.model.Literal;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.ModelFactory;
import com.hp.hpl.jena.rdf.model.Property;
import com.hp.hpl.jena.rdf.model.RDFNode;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.rdf.model.Statement;
import com.hp.hpl.jena.rdf.model.StmtIterator;
import edu.uwb.braingrid.provenance.model.ProvOntology;
import edu.uwb.braingrid.workbench.FileManager;
import edu.uwb.braingrid.workbench.project.ProjectMgr;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.InetAddress;
import java.net.URL;
import java.net.URLConnection;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.riot.RiotNotFoundException;

/**
 * <h2>Manages provenance for projects specified within the Brain Grid Toolbox
 * Workbench.</h2>
 *
 * <p>
 * Basic construction requires input files, simulator process, and an output
 * file.</p>
 *
 * <p>
 * Provenance may also be built as a work-in-progress using the add
 * functions.</p>
 *
 * <hr><i>Required Libraries: All libraries included within Apache Jena 2.1</i>
 *
 * @author Del Davis
 * @version 0.1
 */
public class ProvMgr {

    // <editor-fold defaultstate="collapsed" desc="Members">
    /* URI's and labels used to describe the provenance */
    private String provOutputFileURI;
    private static String localNameSpaceURI;
    private static String remoteNameSpaceURI;
    /* flags called prior to an operation through respective query functions */
    /* RDF in-memory representation of the provenance */
    private Model model;

    public static final String ipServiceURL = "http://checkip.amazonaws.com/";
    public static String REMOTE_NS_PREFIX = "remote";
    public static String LOCAL_NS_PREFIX = "local";

    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Construction">
    /**
     * Constructs the Provenance Constructor object from a previously recorded
     * provenance file.
     *
     * @param project - Used as the base-name for the provenance file
     * @param load - True if the provenance should be loaded from an existing
     * file, Otherwise false, in which case a new provenance model will be
     * created which has not yet been persisted to file storage anywhere
     * @throws java.io.IOException, RiotNotFoundException
     */
    public ProvMgr(ProjectMgr project, boolean load) throws IOException {
        if (load) {
            load(project);
        } else {
            init(project);
        }
    }

    /**
     * Empty initialization. Sets safety values for members.
     */
    private void init(ProjectMgr project) throws IOException {
        // create RDF model
        model = ModelFactory.createDefaultModel();
        provOutputFileURI
                = project.determineProvOutputLocation()
                + project.getName() + ".ttl";
        // set prefixes for...
        // RDF syntax
        model.setNsPrefix("rdf", ProvOntology.getRDFNameSpaceURI());
        // RDF schema
        model.setNsPrefix("rdfs", ProvOntology.getRDFSNameSpaceURI());
        // w3 Prov Ontology
        model.setNsPrefix("prov", ProvOntology.getPROVNameSpaceURI());
        // XML schema
        model.setNsPrefix("xsd", ProvOntology.getXSDNameSpaceURI());
        localNameSpaceURI = getLocalNameSpaceURI();
        // BrainGrid Prov
        model.setNsPrefix(LOCAL_NS_PREFIX, localNameSpaceURI);
    }

    /**
     * Loads a model from a previously saved turtle file
     *
     * @param projectName - The base name of the file containing the provenance
     * @return true if the file loaded a model properly, otherwise false
     */
    private void load(ProjectMgr project) throws RiotNotFoundException, IOException {
        String name = project.getName();
        provOutputFileURI = project.determineProvOutputLocation()
                + name + ".ttl";
        model = RDFDataMgr.loadModel(provOutputFileURI);
        localNameSpaceURI = getLocalNameSpaceURI();
        model.setNsPrefix(LOCAL_NS_PREFIX, localNameSpaceURI);
        trimRemoteNS();
    }

    /**
     * Removes the provenance filename from which the model was loaded from all
     * name spaces associated with remote machines.
     */
    private void trimRemoteNS() {
        Map<String, String> nsMap = model.getNsPrefixMap();
        for (Entry entry : nsMap.entrySet()) {
            String nameSpace = (String) entry.getKey();
            String uri = (String) entry.getValue();
            if (nameSpace.startsWith(REMOTE_NS_PREFIX)) {
                remoteNameSpaceURI = uri.substring(uri.lastIndexOf('/') + 1);
                model.setNsPrefix(nameSpace, remoteNameSpaceURI);
            }
        }
    }

    /**
     * Sets a prefix in the model. This may be used to eliminate the file
     * name-spacing that occurs when the provenance model is loaded from a file.
     * A # separator is appended to the end of the URI as a delimiter
     * automatically, do not include it at the end of the URI. URIs in the model
     * which begin with the prefix URI
     *
     * @param prefix - Identifier for the name space
     * @param uri - The URI of the name space
     */
    public void setNsPrefix(String prefix, String uri) {
        // if the model doesn't have a prefix for the uri
        if (model.getNsURIPrefix(uri + "#") == null) {
            model.setNsPrefix(prefix, uri + "#");
        }
    }

    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Getters">
    /**
     * Retrieves the URI for the output file of the provenance model
     *
     * @return The URI of the provenance output file
     */
    public String getProvFileURI() {
        return provOutputFileURI;
    }

    /**
     * Gets the RDF model maintained by the manager
     *
     * @return the resource description framework model maintained by the
     * manager
     */
    public Model getModel() {
        return model;
    }

    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Model Manipulation">
    /**
     * Adds an entity to the provenance model. An entity is a physical, digital,
     * conceptual, or other kind of thing with some fixed aspects; entities may
     * be real or imaginary.
     *
     * Note: This method may return an existing Resource with the correct URI
     * and model, or it may construct a fresh one, as the
     * com.hp.hpl.jena.rdf.model.Model.createResource sees fit. However, you may
     * specify replacement of an existing entity, by providing the replace
     * parameter with a true value.
     *
     * @param uri - the URI of the entity to be created
     * @param label - optional text used in applying a label to the entity
     * resource (used for group queries)
     * @param remote - True if the remote name space prefix should be used
     * @param replace - True if all instances of existing resources with the
     * specified URI should be removed from the model prior to adding this
     * entity resource
     * @return The resource representing the entity (used for method chaining or
     * complex construction)
     */
    public Resource addEntity(String uri, String label, boolean remote, boolean replace) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        if (replace) {
            removeResource(uri);
        }
        // make parts necessary for defining this particular entity in the model
        Resource entityToAdd = createStatement(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getEntityStartingPointClassFullURI());

        // add the label
        if (label != null) {
            labelResource(entityToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return entityToAdd;
    }

    /**
     * Adds an activity to the provenance model. An activity is something that
     * occurs over a period of time and acts upon or with entities; it may
     * include consuming, processing, transforming, modifying, relocating,
     * using, or generating entities.
     *
     * Note: This method may return an existing Resource with the correct URI
     * and model, or it may construct a fresh one, as the
     * com.hp.hpl.jena.rdf.model.Model.createResource sees fit. However, you may
     * specify replacement of an existing entity, by providing the replace
     * parameter with a true value.
     *
     * @param uri - the URI of the activity to be created
     * @param label - optional text used in applying a label to the entity
     * resource (used for group queries)
     * @param remote - True if the remote name space prefix should be used
     * @param replace - True if all instances of existing resources with the
     * specified URI should be removed from the model prior to adding this
     * activity resource
     * @return The resource representing the entity (used for method chaining or
     * complex construction)
     */
    public Resource addActivity(String uri, String label, boolean remote, boolean replace) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        if (replace) {
            removeResource(uri);
        }
        // make parts necessary for defining this activity in the model
        Resource activityToAdd = createStatement(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getActivityStartingPointClassFullURI());
        // add the label if one was provided
        if (label != null) {
            labelResource(activityToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return activityToAdd;
    }

    /**
     * Adds a software agent to the provenance model. A software agent is
     * running software.
     *
     * Note: This method may return an existing Resource with the correct URI
     * and model, or it may construct a fresh one, as the
     * com.hp.hpl.jena.rdf.model.Model.createResource sees fit. However, you may
     * specify replacement of an existing entity, by providing the replace
     * parameter with a true value.
     *
     * @param uri - the URI of the software agent to be created
     * @param label - optional text used in applying a label to the entity
     * resource (used for group queries)
     * @param remote - True if the remote name space prefix should be used
     * @param replace - True if all instances of existing resources with the
     * specified URI should be removed from the model prior to adding this agent
     * resource
     * @return The resource representing the software agent (used for method
     * chaining or complex construction)
     */
    public Resource addSoftwareAgent(String uri, String label, boolean remote, boolean replace) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        if (replace) {
            removeResource(uri);
        }
        // make parts necessary for defining this particular agent in the model
        Resource agentToAdd = createStatement(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getSoftwareAgentExpandedClassFullURI());
        // add the label if one was provided
        if (label != null) {
            labelResource(agentToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return agentToAdd;
    }

    /**
     * Describes the association of an activity to an agent. An activity
     * association is an assignment of responsibility to an agent for an
     * activity, indicating that the agent had a role in the activity. It
     * further allows for a plan to be specified, which is the plan intended by
     * the agent to achieve some goals in the context of this activity.
     *
     * Note: Activity and agent resources must first be described in the model
     * before calling this method.
     *
     * @param activity - A resource description of the activity for which the
     * agent is responsible
     * @param agent - A resource description of the agent that is responsible
     * for activity
     * @return - The statement resource describing this association (used for
     * method chaining or complex construction)
     */
    public Resource wasAssociatedWith(Resource activity, Resource agent) {
        return createStatement(activity.getURI(),
                ProvOntology.getWasAssociatedWithStartingPointPropertyFullURI(),
                agent.getURI());
    }

    /**
     * Describes usage of an entity for an activity. Usage is the beginning of
     * utilizing an entity by an activity. Before usage, the activity had not
     * begun to utilize this entity and could not have been affected by the
     * entity.
     *
     * Note: Activity and entity resources must first be described in the model
     * before calling this method.
     *
     * @param activity - A resource description of the activity for which agent
     * is responsible
     * @param entity
     * @return - The statement resource describing this usage (used for method
     * chaining or complex construction)
     */
    public Resource used(Resource activity, Resource entity) {
        return createStatement(activity.getURI(),
                ProvOntology.getUsedStartingPointPropertyFullURI(),
                entity.getURI());
    }

    /**
     * Specifies the derivation of an destination entity from an existing source
     * entity. A derivation is a transformation of an entity into another, an
     * update of an entity resulting in a new one, or the construction of a new
     * entity based on a pre-existing entity.
     *
     * @param source - An existing entity resource from the provenance record
     * @param dest - A newly derived entity resource. Note: This entity must be
     * generated in the provenance record prior to a call to this function.
     * @return - The derivation statement from created by this function call.
     */
    public Resource wasDerivedFrom(Resource source, Resource dest) {
        return createStatement(source.getURI(),
                ProvOntology.getWasDerivedFromStartingPointPropertyFullURI(),
                dest.getURI());
    }

    /**
     * Qualifies a "generated" statement (see ProvMgr.generated), which
     * specifies the generation of an entity. Generation is the completion of
     * production of a new entity by an activity. This entity did not exist
     * before generation and becomes available for usage after this generation.
     * Note: In terms of the provenance record, a statement of the entity's
     * existence must first be added to the provenance record in order to show
     * generation. However, for the purposes of inference-based queries, the
     * entity does not exist until it has been generated (by this function).
     *
     * @param activity - The activity that generated the specified entity
     * @param entity - The entity that was generated by the specified activity
     * @return The statement that has been added to the provenance record
     * through the invocation of this function.
     */
    private Resource wasGeneratedBy(Resource entity, Resource activity) {
        return createStatement(entity.getURI(),
                ProvOntology.getWasGeneratedByStartingPointPropertyFullURI(),
                activity.getURI());
    }

    /**
     * Specifies the generation of an entity. Generation is the completion of
     * production of a new entity by an activity. This entity did not exist
     * before generation and becomes available for usage after this generation.
     *
     * Note: In terms of the provenance record, a statement of the entity's
     * existence must first be added to the provenance record in order to show
     * generation. However, for the purposes of inference-based queries, the
     * entity does not exist until it has been generated (by calling this
     * function).
     *
     * Note: This function will also add its inverse (ProvMgr.wasGeneratedBy) to
     * the model.
     *
     * @param entity - The entity that was generated by the specified activity
     * @param activity - The activity that generated the specified entity
     * @return The statement that has been added to the provenance record
     * through the invocation of this function.
     */
    public Resource generated(Resource activity, Resource entity) {
        wasGeneratedBy(entity, activity);
        return createStatement(activity.getURI(),
                ProvOntology.getGeneratedExpandedPropertyFullURI(),
                entity.getURI());
    }

    /**
     * Describes when an instantaneous event occurred. The PROV data model is
     * implicitly based on a notion of instantaneous events (or just events),
     * that mark transitions in the world. Events include generation, usage, or
     * invalidation of entities, as well as starting or ending of activities.
     * This notion of event is not first-class in the data model, but it is
     * useful for explaining its other concepts and its semantics.
     *
     * @param activity - The activity that occurred
     * @param instantaneousEvent -
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource if
     * it is available (you can look up its URI by calling getSubject)
     */
    public Resource atTime(Resource activity, Resource instantaneousEvent) {
        return createStatement(activity.getURI(),
                ProvOntology.getAtTimeQualifiedPropertyFullURI(),
                instantaneousEvent.getURI());
    }

    /**
     * Describes when an activity started (use this for activities, use atTime
     * for instantaneous events)
     *
     * @param activity - The activity that occurred
     * @param date - A date representing the time at which the activity started
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource it
     * is available (you can look up its URI by calling getSubject)
     */
    public Resource startedAtTime(Resource activity, Date date) {
        Statement atTime = model.createStatement(activity,
                model.createProperty(ProvOntology.
                        getStartedAtTimeStartingPointPropertyFullURI()),
                getDateLiteral(date));
        model.add(atTime);
        return activity;
    }

    /**
     * Describes when an activity ended (use this for activities, use atTime for
     * instantaneous events)
     *
     * @param activity - The activity that occurred
     * @param date - A date representing the time at which the activity ended
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource it
     * is available (you can look up its URI by calling getSubject)
     */
    public Resource endedAtTime(Resource activity, Date date) {
        Statement atTime = model.createStatement(activity,
                model.createProperty(ProvOntology.
                        getEndedAtTimeStartingPointPropertyFullURI()),
                getDateLiteral(date));
        model.add(atTime);
        return activity;
    }

    /**
     * Creates a definition for the provenance class resource in the model.
     *
     * @param resourceURI - The direct URI to be given to the resource. This
     * means that any prefixing is applied to the URI before this method is
     * called.
     * @param propertyURI - The direct property URI. This is a URI from the prov
     * ontology or RDF schema. Direct means that any prefixing is taken care of
     * prior to this method call.
     * @param definitionURI - The direct definition URI. This must already
     * contain the corresponding prov ontology uri
     * @return The resource that was added to the provenance model.
     */
    private Resource createStatement(String resourceURI, String propertyURI,
            String definitionURI) {
        // create parts
        Resource resource = model.createResource(resourceURI);
        Property property = model.createProperty(propertyURI);
        Resource definition = model.createResource(definitionURI);
        // make a statement out of them
        Statement stmt = model.createStatement(resource, property, definition);
        // add it to the model
        model.add(stmt);
        // provide the resource to the caller for future use
        return resource;
    }

    /**
     * This should only be used when optimistic provenance recording is used and
     * a resource has failed to be generated or be of importance to the
     * provenance record. Do not use this method to overwrite an existing
     * resource. If this method is used, all statements in which the resource
     * specified is the subject of the statement will be removed from the
     * provenance record.
     *
     * @param resourceURI - Identifies the statements that should be removed
     * from the provenance record. All statements whose subject has this URI
     * will be removed from the provenance record.
     */
    public void removeResource(String resourceURI) {
        Resource resource = model.getResource(resourceURI);
        StmtIterator si = model.listStatements(resource, (Property) null,
                (RDFNode) null);
        while (si.hasNext()) {
            Statement s = si.nextStatement();
            model.remove(s);
        }
    }

    /**
     * Adds a label to a resource for ease of query
     *
     * @param resource - The resource to label
     * @param labelText - The text of the literal used as a label
     * @return The same resource that was provided (for method-chaining)
     */
    private Resource labelResource(Resource resource, String labelText) {
        // create parts
        Property labelProperty = model.createProperty(
                ProvOntology.getRDFSLabelFullURI());
        Literal label = model.createLiteral(labelText);
        // add the label to the resource
        resource.addLiteral(labelProperty, label);
        // provide the resource to the caller for method-chaining
        return resource;
    }

    /**
     * Creates a labeled collection resource. This function must be invoked
     * prior to adding resources to a collection. The uri should be a folder
     * location where the input files reside (or at least one of the input files
     * resides)
     *
     * @param uri - The identifier that points to the collection resource
     * @param label - Optional label, for group queries
     * @return The resource that was defined for the collection
     */
    public Resource createCollection(String uri, String label) {
        Resource collection = createStatement(uri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getCollectionExpandedClassFullURI());
        labelResource(collection, label);
        return collection;
    }

    /**
     * Adds a resource to a collection. Members are added to a collection, but
     * the members should first be defined as entities in the model. Use
     * addEntity to accomplish this, prior to adding the member to the
     * collection.
     *
     * @param collection - The collection to add the entity to
     * @param entity - The resource to add to the collection
     * @return - A reference to resource which describes the collection (used
     * for method chaining)
     */
    public Resource addToCollection(Resource collection, Resource entity) {
        Property hadMember = model.createProperty(ProvOntology.
                getHadMemberExpandedPropertyFullURI());
        model.createStatement(collection, hadMember, entity);
        return collection;
    }

    /**
     * Adds a series of statements indicating that an agent created a file at a
     * given location. If the agent does not yet exist, it is first added to the
     * model.
     *
     * @param activityURI
     * @param activityLabel
     * @param agentURI - Identifies the agent responsible for generating the
     * file
     * @param agentLabel - Optional label for the agent created with agentURI
     * @param remoteAgent - Indicates the locale of the agent with respect to
     * the locale where this function was invoked
     * @param fileURI - Identifies the file that was generated
     * @param fileLabel - Optional label for the generated file created with
     * fileURI
     * @param remoteFile - Indicates the locale of the file with respect to the
     * locale where this activity occurred
     * @return The resource object associated with the file that was generated
     */
    public Resource addFileGeneration(String activityURI, String activityLabel,
            String agentURI, String agentLabel, boolean remoteAgent,
            String fileURI, String fileLabel, boolean remoteFile) {
        Resource activity = addActivity(activityURI, activityLabel, remoteAgent, false);
        Resource program = addSoftwareAgent(agentURI, agentLabel, remoteAgent, false);
        Resource file = addEntity(fileURI, fileLabel, remoteFile, false);
        generated(activity, file);
        wasAssociatedWith(activity, program);
        return file;
    }

    /**
     * Adds a series of statements indicating that an agent created a file at a
     * given location.
     *
     * @param activity - An existing resource defining the file generation
     * activity
     * @param agent - An existing agent resource (possibly a software agent)
     * defining the agent that is responsible for generating the file
     * @param file - An existing entity resource for the file
     * @return The resource associated with the file that was generated
     */
    public Resource addFileGeneration(Resource activity, Resource agent,
            Resource file) {
        generated(activity, file);
        wasAssociatedWith(activity, agent);
        return file;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Query Support">
    /**
     * Provides the URIs for all subjects in the provenance model.
     *
     * Note: Full URIs are accumulated in the list parameter that was passed in.
     * Whereas, a list of abbreviated (any parent directories removed) URIs are
     * returned. Abbreviation occurs on web resources, as well as file
     * resources. (e.g. http://www.somesite.com/myFile.xml becomes myFile.xml)
     *
     * @param fullURIs - A List to populate with full URIs of all subjects.
     * @return - A List of abbreviated URIs of all subjects (parent directories
     * removed)
     */
    public List<String> getSubjects(List<String> fullURIs) {
        List<String> abbreviatedURI = new ArrayList<>();
        StmtIterator si = model.listStatements();
        Resource r;
        String uri;
        Statement s;
        while (si.hasNext()) {
            s = si.nextStatement();
            r = s.getSubject();
            uri = r.getURI();
            fullURIs.add(uri);
            abbreviatedURI.add(FileManager.getSimpleFilename(uri));
        }
        return abbreviatedURI;
    }

    /**
     * Provides a collection of all predicate URIs in the provenance model.
     *
     * @return A collection of all predicate URIs in the provenance model.
     */
    public Collection<String> getPredicates() {
        HashSet<String> fullURISet = new HashSet<>();
        StmtIterator si = model.listStatements();
        while (si.hasNext()) {
            fullURISet.add(si.nextStatement().getPredicate().toString());
        }
        return fullURISet;
    }

    /**
     * Provides the URIs for all objects in the provenance model.
     *
     * Note: Full URIs are accumulated in the list parameter that was passed in.
     * Whereas, a list of abbreviated (any parent directories removed) URIs are
     * returned. Abbreviation occurs on web resources, as well as file
     * resources. (e.g. http://www.somesite.com/myFile.xml becomes myFile.xml)
     *
     * @param fullURIs - A List to populate with full URIs of all objects.
     * @return - A List of abbreviated URIs of all objects (parent directories
     * removed)
     */
    public List<String> getObjects(List<String> fullURIs) {
        List<String> abbreviatedURI = new ArrayList<>();
        StmtIterator si = model.listStatements();
        Statement s;
        while (si.hasNext()) {
            s = si.nextStatement();
            fullURIs.add(s.getObject().toString());
            abbreviatedURI.add(FileManager.getSimpleFilename(s.getObject().toString()));
        }
        return abbreviatedURI;
    }

    /**
     * Provides a readable textual representation of provenance statements where
     * subjectURI contains subjectText, predicateURI contains predicateText, and
     * objectURI contains objectText. Any and all of the fields may be used as
     * wildcards by passing a null value for the respective field.
     *
     * @param subjectText - text that should be contained within the subject of
     * a statement if the statement matches
     * @param predicateText - text that should be contained within the predicate
     * of a statement if the statement matches
     * @param objectText - text that should be contained within the object of a
     * statement if the statement matches
     * @param lineDelimiter - separates statements from each other
     * @return statements that match the query
     */
    public String queryProvenance(String subjectText, String predicateText, String objectText, String lineDelimiter) {
        String statements = "";
        Statement stmt;
        String subject, predicate, object;
        RDFNode objectNode;
        boolean isVowel;
        char letter;

        StmtIterator iter = model.listStatements();
        while (iter.hasNext()) {
            stmt = iter.nextStatement();
            subject = stmt.getSubject().getURI();
            predicate = stmt.getPredicate().getURI();
            objectNode = stmt.getObject();
            if (objectNode.isURIResource()) {
                object = objectNode.asResource().getURI();
            } else if (objectNode.isAnon()) {
                object = objectNode.asNode().getURI();
            } else if (objectNode.isLiteral()) {
                object = objectNode.asLiteral().getString();
            } else {
                object = objectNode.asResource().toString();
            }
            if (object.length() > 0) {
                letter = object.charAt(0);
                isVowel = letter == 'a' || letter == 'e' || letter == 'i'
                        || letter == 'o' || letter == 'u' || letter == 'h';
                if (subject.toLowerCase().contains(subjectText.toLowerCase())
                        && predicate.toLowerCase().contains(predicateText.toLowerCase())
                        && object.toLowerCase().contains(objectText.toLowerCase())) {
                    predicate = ProvOntology.translatePredicate(predicate,
                            isVowel);
                    statements += subject + " " + predicate + " " + object;
                    if (iter.hasNext()) {
                        statements += lineDelimiter;
                    }
                }
            }
        }
        return statements;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Utility Functions">
    /**
     * Converts the specified URI to its full form, which includes the
     * associated namespace URI.
     *
     * @param uri - The URI to be converted
     * @return The full form of the URI, which includes the associated namespace
     * URI
     */
    private String getProjectFullLocalURI(String uri) {
        return LOCAL_NS_PREFIX + ":" + uri;
    }

    /**
     * Converts the specified URI to its full form, which includes the
     * associated namespace URI.
     *
     * @param uri - The URI to be converted
     * @return The full form of the URI, which includes the associated namespace
     * URI
     */
    private String getProjectFullRemoteURI(String uri) {
        return REMOTE_NS_PREFIX + ":" + uri;
    }

    /**
     * Uses a web service to determine the external IP of the host machine. If
     * this is not available, the loop back address is used along with the host
     * name of the local machine. If the local host name cannot be ascertained
     * then a default assignment is made.
     *
     * @return A description of the host name along with the most uniquely
     * describing IP available (may or may not come from the InetAddress)
     */
    private String getLocalNameSpaceURI() {
        String localNameSpace = "";

        try {
            URL locator = new URL(ipServiceURL);
            URLConnection connection = locator.openConnection();
            InputStream is = connection.getInputStream();
            InputStreamReader isr = new InputStreamReader(is);
            BufferedReader reader = new BufferedReader(isr);
            String str = null;
            str = reader.readLine();
            if (null == str) {
                str = "127.0.0.1";
            }
            localNameSpace = InetAddress.getLocalHost().getHostName()
                    + "@" + str + "#";
        } catch (IOException e) {
            try {
                localNameSpace = InetAddress.getLocalHost().getHostName()
                        + "@" + InetAddress.getLoopbackAddress() + "#";
            } catch (UnknownHostException ex) {
                localNameSpace = "UnknownHostName@127.0.0.1#";
            }
        }

        return localNameSpace;
    }

    /**
     * Writes the model to the file with the output filename specified during
     * construction or initialization
     *
     * @param project - The project that this provenance is recorded for. The
     * project maintains its name, which is used as the base name for the
     * provenance file
     * @throws java.io.FileNotFoundException
     * @throws java.io.IOException
     */
    public void persist(ProjectMgr project) throws FileNotFoundException, IOException {
        String directory = project.determineProvOutputLocation();
        (new File(directory)).mkdirs();
        model.write(new FileOutputStream(directory + project.getName()
                + ".ttl", false), "TURTLE");
    }

    /**
     * Outputs a constructed model to a stream (may be System.out or a network
     * or file-based stream)
     *
     * Assumption: The caller has checked that the model was assembled properly
     * by calling isAssembled. Model will not be written if not assembled.
     *
     * @param out - The stream to print the model to
     */
    public void outputModel(PrintStream out) {
        if (model != null) {
            model.write(out);
        }
    }

    /**
     * Converts a java date to the format required by Jena's Riot package.
     *
     * @param date - The java date object to convert
     * @return - An xsd formatted date time string
     */
    private Literal getDateLiteral(Date date) {
        Calendar cal = new GregorianCalendar();
        cal.setTime(date);
        return model.createTypedLiteral(cal);
    }
    // </editor-fold>
}
