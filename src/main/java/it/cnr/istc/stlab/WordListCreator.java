package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import org.apache.jena.query.ParameterizedSparqlString;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;

public class WordListCreator {

	private static final String SELECT_WORDS = "SELECT  DISTINCT ?lexicalForm {\n"
			+ "  ?word <https://w3id.org/framester/wn/wn30/schema/lexicalForm> ?lexicalForm \n" + "} ";

	private static final String endpoint = "http://localhost:3030/framester-4/query";

	public static void main(String[] args) throws IOException {

		BufferedReader br = new BufferedReader(new FileReader("wordIDs"));
		FileOutputStream fos = new FileOutputStream(new File("lexicaForms"));

		ParameterizedSparqlString pss = new ParameterizedSparqlString(SELECT_WORDS);
		br.lines().forEach(uri -> {
			String wordUri = uri.split("\t")[0];
			int idUri = Integer.parseInt(uri.split("\t")[1]);
			pss.setIri("word", wordUri);
			QueryExecution qexec = QueryExecutionFactory.sparqlService(endpoint, pss.asQuery());
			ResultSet rs = qexec.execSelect();
			if (rs.hasNext()) {
				QuerySolution querySolution = (QuerySolution) rs.next();
				String lexicalForm = querySolution.get("lexicalForm").asLiteral().getValue().toString();
				try {
					fos.write((lexicalForm + "\t" + idUri + "\n").getBytes());
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			qexec.close();
		});

		br.close();

		fos.flush();
		fos.close();

	}

}
