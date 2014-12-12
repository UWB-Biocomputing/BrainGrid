package edu.uwb.braingrid.provenance;

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
     * Add an entity to the provenance model.
     *
     * @param uri - the identifier that points to the entity resource
     * @param label - optional text used in applying a label to the entity
     * resource (used for ease of future query)
     * @param remote - True if the remote namespace prefix should be used
     * @return The resource representing the entity (used for method chaining or
     * complex construction)
     */
    public Resource addEntity(String uri, String label, boolean remote) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        removeResource(uri);
        // make parts necessary for defining this particular entity in the model
        Resource entityToAdd = createDefinition(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getEntityStartingPointClassFullURI());

        // add the label
        if (label != null) {
            labelResource(entityToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return entityToAdd;
    }

    public Resource addActivity(String uri, String label, boolean remote) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        removeResource(uri);
        // make parts necessary for defining this activity in the model
        Resource activityToAdd = createDefinition(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getActivityStartingPointClassFullURI());
        // add the label if one was provided
        if (label != null) {
            labelResource(activityToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return activityToAdd;
    }

    public Resource addSoftwareAgent(String uri, String label, boolean remote) {
        uri = uri.replaceAll("\\\\", "/");
        String fullUri = remote ? getProjectFullRemoteURI(uri)
                : getProjectFullLocalURI(uri);
        removeResource(uri);
        // make parts necessary for defining this particular agent in the model
        Resource agentToAdd = createDefinition(fullUri,
                ProvOntology.getRDFTypeFullURI(),
                ProvOntology.getSoftwareAgentExpandedClassFullURI());
        // add the label if one was provided
        if (label != null) {
            labelResource(agentToAdd, label);
        }
        // provide the resource to the caller for method-chaining
        return agentToAdd;
    }

    public Resource associateWith(Resource activity, Resource agent) {
        return createDefinition(activity.getURI(),
                ProvOntology.getWasAssociatedWithStartingPointPropertyFullURI(),
                agent.getURI());
    }

    /**
     * Describes when an activity occurred
     *
     * @param activity - The activity that occurred
     * @param time - The time at which the activity occurred
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource if
     * it is available (you can look up its URI by calling getSubject)
     */
    public Resource atTime(Resource activity, String time) {
        return createDefinition(activity.getURI(),
                ProvOntology.getAtTimeQualifiedPropertyFullURI(), time);
    }

    /**
     * Describes when an activity started (use this for activities, use atTime
     * for instantaneous events)
     *
     * @param activity - The activity that occurred
     * @param time - The time at which the activity started
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource it
     * is available (you can look up its URI by calling getSubject)
     */
    public Resource startedAtTime(Resource activity, String time) {
        return createDefinition(activity.getURI(),
                ProvOntology.getStartedAtTimeStartingPointPropertyFullURI(),
                time);
    }

    /**
     * Describes when an activity ended (use this for activities, use atTime for
     * instantaneous events)
     *
     * @param activity - The activity that occurred
     * @param time - The time at which the activity ended
     * @return No uses of the resulting statement are known to be any better
     * than using the activity resource, consider using the activity resource it
     * is available (you can look up its URI by calling getSubject)
     */
    public Resource endedAtTime(Resource activity, String time) {
        return createDefinition(activity.getURI(),
                ProvOntology.getEndedAtTimeStartingPointPropertyFullURI(),
                time);
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
    private Resource createDefinition(String resourceURI, String propertyURI,
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

    private void removeResource(String resourceURI) {
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
     * @return The resource that was defined for the collection
     */
    private Resource createCollection(String uri, String label) {
        Resource collection = createDefinition(uri,
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
     * @param resource - The resource to add to the collection
     * @return - A reference to resource which describes the collection (used
     * for method chaining)
     */
    private Resource addToCollection(Resource collection, Resource resource) {
        Property hadMember = model.createProperty(ProvOntology.
                getHadMemberExpandedPropertyFullURI());
        Statement s = model.createStatement(collection, hadMember, resource);
        return collection;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Query Support">
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

    public Collection<String> getPredicates() {
        HashSet<String> fullURISet = new HashSet<>();
        StmtIterator si = model.listStatements();
        while (si.hasNext()) {
            fullURISet.add(si.nextStatement().getPredicate().toString());
        }
        return fullURISet;
    }

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
    private String getProjectFullLocalURI(String uri) {
        //return localNS + uri;
        return LOCAL_NS_PREFIX + ":" + uri;
    }

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
    // </editor-fold>    
}
