package it.cnr.istc.stlab;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.jena.query.ParameterizedSparqlString;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;

public class WordToDomainMatrixCreator {

	private static final String SELECT_WORDS = "SELECT  DISTINCT ?word {\n"
			+ "  GRAPH <https://w3id.org/framester/data/wordnet/wn30> { ?word a <https://w3id.org/framester/wn/wn30/schema/Word> }\n"
			+ "} ";

	private static final String SELECT_DOMAINS = "SELECT  DISTINCT ?domain {\n"
			+ "  GRAPH <https://w3id.org/framester/data/wordnet/wn30> { ?domain a <https://w3id.org/framester/wn/wn30/wndomains/WNDomain> }\n"
			+ "} ";

	private static final String SELECT_WORDS_DOMAINS = "SELECT DISTINCT  ?d \n" + "WHERE{\n"
			+ "  ?synset <https://w3id.org/framester/wn/wn30/wndomains/synsetDomain> ?d .\n"
			+ "  ?synset <https://w3id.org/framester/wn/wn30/schema/containsWordSense> ?wordsense .\n"
			+ "  ?wordsense <https://w3id.org/framester/wn/wn30/schema/word> ?word .\n"
			+ "  #?word <https://w3id.org/framester/wn/wn30/schema/lexicalForm> ?lexicalForm .\n" + "} \n";

	private static final String endpoint = "http://localhost:3030/framester-4/query";

	private static Map<String, Integer> getDomains() throws IOException {
		QueryExecution qexec = QueryExecutionFactory.sparqlService(endpoint, QueryFactory.create(SELECT_DOMAINS));
		Map<String, Integer> wnDomainToID = new HashMap<>();
		int count = 0;
		ResultSet rs = qexec.execSelect();
		FileOutputStream fosWNDomains = new FileOutputStream(new File("wnDomainsIDs"));

		while (rs.hasNext()) {
			String domain = rs.next().get("domain").asResource().getURI();
			wnDomainToID.put(domain, count);
			fosWNDomains.write((domain + "\t" + count + "\n").getBytes());
			count++;
		}
		qexec.close();
		fosWNDomains.flush();
		fosWNDomains.close();

		return wnDomainToID;

	}

	public static void main(String[] args) throws IOException {
		QueryExecution qexec = QueryExecutionFactory.sparqlService(endpoint, QueryFactory.create(SELECT_WORDS));

		ResultSet rs = qexec.execSelect();

		ParameterizedSparqlString pss = new ParameterizedSparqlString(SELECT_WORDS_DOMAINS);
		int wordId = 0;

		Map<String, Integer> wnDomainToID = getDomains();

		FileOutputStream fos = new FileOutputStream(new File("wordIDs"));
		FileOutputStream fosWordToDomainMatrix = new FileOutputStream(new File("wordToDomainMatrix"));

		while (rs.hasNext()) {

			if (wordId % 10000 == 0) {
				System.out.println(wordId);
			}

			QuerySolution querySolution = (QuerySolution) rs.next();
			String wordIRI = querySolution.get("word").asResource().getURI();
			fos.write((wordIRI + "\t" + wordId + "\n").getBytes());
			pss.setIri("word", wordIRI);

			QueryExecution qexecDomains = QueryExecutionFactory.sparqlService(endpoint, pss.asQuery());
			ResultSet rs_Domains = qexecDomains.execSelect();
			while (rs_Domains.hasNext()) {
				QuerySolution qs = (QuerySolution) rs_Domains.next();
				String domain = qs.get("d").asResource().getURI();
				Integer domainId = wnDomainToID.get(domain);
				if (domainId == null) {
					throw new RuntimeException();
				}

				fosWordToDomainMatrix.write((wordId + "\t" + domainId + "\n").getBytes());

			}
			qexecDomains.close();

			wordId++;
		}

		qexec.close();

		fos.flush();
		fos.close();
		fosWordToDomainMatrix.flush();
		fosWordToDomainMatrix.close();

	}

}
