package edu.uwb.braingrid.provenance.model;
/////////////////CLEANED
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.ModelFactory;
import com.hp.hpl.jena.rdf.model.Property;
import com.hp.hpl.jena.rdf.model.RDFNode;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.rdf.model.Statement;
import com.hp.hpl.jena.rdf.model.StmtIterator;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;

/**
 * Encapsulate a statement from an RDF model (or parts of). Provides methods of
 * separation between unique resource identifiers and their respective
 * human-readable counterparts. In addition, provenance statements eliminate
 * abstractness concerns from RDF Model statements. For example, an object may
 * be a String literal. It may not have an associated URI. In this case,
 * querying a provenance statement for its object URI results in a return value
 * of null, which is used as a wildcard in future queries. Whereas, performing
 * the respective operation on an RDF Model statement would result in an
 * exception being thrown.
 *
 * @author Del Davis
 */
public class ProvenanceStatement {

    private Statement statement = null;
    private Model defaultModel = null;
    private Resource subject = null;
    private Property predicate = null;
    private RDFNode objectNode = null;

    /**
     * Constructs this provenance statement as an empty statement
     */
    public ProvenanceStatement() {
        defaultModel = ModelFactory.createDefaultModel();
    }

    /**
     * Constructs this provenance statement based on a statement from a RDF
     * model
     *
     * @param stmt - An existing statement from a RDF model
     */
    public ProvenanceStatement(Statement stmt) {
        statement = stmt;
        defaultModel = ModelFactory.createDefaultModel();
        subject = stmt.getSubject();
        predicate = stmt.getPredicate();
        objectNode = stmt.getObject();
    }

    /**
     * Constructs this provenance statement based on a subject, predicate, and
     * object
     *
     * @param subj - An RDF Resource that represents the subject of this
     * provenance statement
     * @param pred - An RDF Resource that represents the predicate of this
     * provenance statement
     * @param obj - A Resource, anonymous node, or literal RDFNode that
     * represents the object of this provenance statement
     */
    public ProvenanceStatement(Resource subj, Property pred, RDFNode obj) {
        subject = subj;
        predicate = pred;
        objectNode = obj;
    }

    /**
     * Constructs this provenance statement based on URIs of a subject,
     * predicate, and object
     *
     * @param subjURI - Identifier for a RDF Resource that represents the
     * subject of this provenance statement
     * @param predURI - Identifier for a RDF Resource that represents the
     * predicate of this provenance statement
     * @param objURI - Identifier for a Resource, anonymous node, or literal
     * RDFNode that represents the object of this provenance statement
     */
    public ProvenanceStatement(String subjURI, String predURI, String objURI) {
        defaultModel = ModelFactory.createDefaultModel();
        if (subjURI == null || subjURI.isEmpty()) {
            subject = null;
        } else {
            setSubject(subjURI);
        }
        if (predURI == null || predURI.isEmpty()) {
            predicate = null;
        } else {
            setPredicate(predURI);
        }
        if (objURI == null || objURI.isEmpty()) {
            objectNode = null;
        } else {
            setObject(objURI, false);
        }
    }

    /**
     * Determines whether or not this provenance statement was constructed based
     * on a statement from an RDF model.
     *
     * @return True if this provenance statement comes from a model, otherwise
     * false
     */
    public boolean hasNonDefaultModel() {
        return statement != null;
    }

    /**
     * Provides the original model of the RDF statement used to create this
     * provenance statement.
     *
     * @return The original model of the RDF statement used to create this
     * provenance statement or null if no statement was used to construct this
     * provenance statement
     */
    public Model getModel() {
        Model model = hasNonDefaultModel() ? statement.getModel() : null;
        return model;
    }

    /**
     * Provides the statement used to create this provenance statement. May be
     * null.
     *
     * @return The statement used to create this provenance statement or null
     */
    public Statement getStatement() {
        return statement;
    }

    /**
     * Sets the statement of this provenance statement.
     *
     * Warning: The subject, predicate, and object of this provenance statement
     * will be overwritten by the respective resources of the statement provided
     *
     * @param stmt - statement from a model
     */
    public void setStatement(Statement stmt) {
        statement = stmt;
        subject = stmt.getSubject();
        predicate = stmt.getPredicate();
        objectNode = stmt.getObject();
    }

    /**
     * Sets the subject for this provenance statement to the subject of an
     * existing statement
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param subj - An RDF Resource that represents the subject of this
     * provenance statement
     */
    public void setSubject(Resource subj) {
        subject = subj;
        statement = null;
    }

    /**
     * Sets the predicate for this provenance statement to the predicate of an
     * existing statement
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param pred - An RDF Resource that represents the predicate of this
     * provenance statement
     */
    public void setPredicate(Property pred) {
        predicate = pred;
        statement = null;
    }

    /**
     * Sets the object of this provenance statement to the object of an existing
     * statement
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param obj - A Resource, anonymous node, or literal RDFNode that
     * represents the object of this provenance statement
     */
    public void setObject(RDFNode obj) {
        objectNode = obj;
    }

    /**
     * Sets the subject of this provenance statement from a subject created in
     * the default model based on the provided URI.
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param uri - A unique resource identifier used to construct a RDF
     * resource representing the subject of this provenance statement
     */
    public final void setSubject(String uri) {
        if (uri != null) {
            subject = defaultModel.createResource(uri);
        }
    }

    /**
     * Provides the unique resource identifier of the subject for this
     * provenance statement
     *
     * @return The unique resource identifier for the subject of this provenance
     * statement or the empty string if the subject was not set
     */
    public String getSubjectURI() {
        String subj = "";
        if (subject != null) {
            subj = subject.getURI();
        }
        return subj;
    }

    /**
     * Provides the unique resource identifier of the subject for this
     * provenance statement
     *
     * @return The unique resource identifier for the subject of this provenance
     * statement or the empty string if the subject was not set
     */
    public String getSimpleSubjectURI() {
        String subj = "";
        if (subject != null) {
            subj = subject.getURI();
        }
        return subj;
    }

    /**
     * Sets the predicate of this provenance statement to the predicate of an
     * existing statement
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param uri - A unique resource identifier used to construct a RDF
     * resource representing the predicate of this provenance statement
     */
    public final void setPredicate(String uri) {
        if (uri != null) {
            predicate = defaultModel.createProperty(uri);
        }
    }

    /**
     * Provides the unique resource identifier for the predicate of this
     * provenance statement or the empty string.
     *
     * @return The unique resource identifier for the predicate of this
     * provenance statement or the empty string if the predicate was not set
     */
    public String getPredicate() {
        String pred = "";
        if (predicate != null) {
            pred = predicate.getURI();
        }
        return pred;
    }

    /**
     * Sets the object of this provenance statement to the object of an existing
     * statement
     *
     * Warning: The statement of this provenance statement will be invalidated
     * (resulting in a null reference)
     *
     * @param uri - A unique resource identifier used to construct a RDF
     * resource or RDF literal representing the object of this provenance
     * statement
     * @param literal - If true, a Literal will be constructed, otherwise, a
     * Resource
     */
    public final void setObject(String uri, boolean literal) {
        if (uri != null) {
            if (literal) {
                objectNode = defaultModel.createLiteral(uri);
            } else {
                objectNode = defaultModel.createResource(uri);
            }
        }
    }

    /**
     * Provides a collection of provenance statements from the specified model
     * which closely match the URIs of the subject, predicate, and object of
     * this provenance statement. This function is less efficient than
     * strictQuery since it matches URIs against all statements in the model.
     * However, differences between name space prefixes (which may occur
     * automatically when statements are loaded from disk) will be ignored. More
     * specifically, statements match as long as the URIs from the model
     * statement contain the URIs from this provenance statement. The subject,
     * and/or predicate, and/or object of this provenance statement will be used
     * as wildcards if null.
     *
     * @param model - The model to match statements against this
     * @param unique - Determines whether the collection should only contain
     * unique provenance statements
     * @return A collection of provenance statements from model which match the
     * subject, predicate, and object of this provenance statement.
     */
    public Collection<ProvenanceStatement> relaxedQuery(Model model, boolean unique) {
        Collection<ProvenanceStatement> uniqueCollection = new ArrayList<>();
        Collection<ProvenanceStatement> collection = new HashSet<>();
        Collection<ProvenanceStatement> returnCollection;
        String subjectURI, predicateURI, objectURI;
        StmtIterator iter = model.listStatements();
        Statement stmt;
        while (iter.hasNext()) {
            stmt = iter.nextStatement();
            subjectURI = stmt.getSubject().getURI();
            predicateURI = stmt.getPredicate().getURI();
            objectURI = getObjectURI(stmt.getObject());
            boolean subjectMatch = subject == null
                    || subjectURI.toLowerCase().contains(subject.getURI().toLowerCase());
            boolean predicateMatch = predicate == null
                    || predicateURI.toLowerCase().contains(predicate.getURI().toLowerCase());
            boolean objectMatch = objectNode == null
                    || objectURI.toLowerCase().contains(getObjectURI().toLowerCase());
            if (subjectMatch && predicateMatch && objectMatch) {
                if (unique) {
                    uniqueCollection.add(new ProvenanceStatement(stmt));
                } else {
                    collection.add(new ProvenanceStatement(stmt));
                }
            }
        }
        returnCollection = unique ? uniqueCollection : collection;
        return returnCollection;
    }

    /**
     * Provides a collection of provenance statements from the specified model
     * which exactly match the URIs of the subject, predicate, and object of
     * this provenance statement. This function is more efficient than
     * relaxedQuery since it allows indexes maintained by the implementation to
     * improve performance. However, differences between name space prefixes
     * (which may occur automatically when statements are loaded from disk) will
     * cause many seemingly matching statements to be ignored. The subject,
     * and/or predicate, and/or object of this provenance statement will be used
     * as wildcards if null.
     *
     * @param model - The model to match statements against this
     * @param unique - Determines whether the collection should only contain
     * unique provenance statements
     * @return A collection of provenance statements from model which match the
     * subject, predicate, and object of this provenance statement.
     */
    public Collection<ProvenanceStatement> strictQuery(Model model, boolean unique) {
        Collection<ProvenanceStatement> uniqueCollection = new ArrayList<>();
        Collection<ProvenanceStatement> collection = new HashSet<>();
        Collection<ProvenanceStatement> returnCollection;
        StmtIterator iter = model.listStatements(subject, predicate, objectNode);
        while (iter.hasNext()) {
            if (unique) {
                uniqueCollection.add(new ProvenanceStatement(iter.nextStatement()));
            } else {
                collection.add(new ProvenanceStatement(iter.nextStatement()));
            }
        }
        returnCollection = unique ? uniqueCollection : collection;
        return returnCollection;
    }

    /**
     * Provides the URI of the object of this provenance statement
     *
     * @return The unique resource identifier of the object specified
     */
    public String getObjectURI() {
        return ProvenanceStatement.getObjectURI(objectNode);
    }

    /**
     * Provides the URI of an object
     *
     * @param obj A Resource, anonymous node, or literal RDFNode
     * @return The unique resource identifier of the object specified
     */
    public static String getObjectURI(RDFNode obj) {
        String objectURI = "";
        if (obj != null) {
            if (obj.isURIResource()) {
                objectURI = obj.asResource().getURI();
            } else if (obj.isAnon()) {
                objectURI = obj.asNode().getURI();
            } else if (obj.isLiteral()) {
                objectURI = obj.asLiteral().getString();
            } else {
                objectURI = obj.asResource().toString();
            }
        }
        return objectURI;
    }
}
