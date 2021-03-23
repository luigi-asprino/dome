package it.cnr.istc.stlab;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.cnr.istc.stlab.lgu.commons.semanticweb.querying.QueryExecutionRetrieverWithLimitAndOffset;

public class App {

	private static Logger logger = LoggerFactory.getLogger(App.class);

	public static void main(String[] args) throws IOException {
		logger.info("Test");
//		FileOutputStream fos = new FileOutputStream(new File("words.txt"));
//		QueryExecutionRetrieverWithLimitAndOffset qer = new QueryExecutionRetrieverWithLimitAndOffset("SELECT DISTINCT * {?word <https://w3id.org/framester/wn/wn30/schema/lexicalForm> ?lexicalForm .}", "http://etna.istc.cnr.it/framester2/sparql", 10000);
//		QueryExecution qexec;
//		while ((qexec = qer.next()) != null) {
//			ResultSet rs = qexec.execSelect();
//			if(!rs.hasNext()) {
//				qexec.close();
//				break;
//			}
//			while (rs.hasNext()) {
//				QuerySolution qs = (QuerySolution) rs.next();
//				fos.write((qs.get("lexicalForm").asLiteral().getString()+"\n").getBytes());
//			}
//			qexec.close();
//		}
//		fos.flush();
//		fos.close();
	}
}
