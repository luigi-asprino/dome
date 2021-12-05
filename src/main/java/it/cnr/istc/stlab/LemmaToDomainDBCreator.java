package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.rocksdb.RocksDBException;

import com.google.common.collect.HashBiMap;

import it.cnr.istc.stlab.lgu.commons.misc.ProgressCounter;
import it.cnr.istc.stlab.rocksmap.RocksMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class LemmaToDomainDBCreator {

	private static final String SELECT_WORDS_DOMAINS = "SELECT DISTINCT  ?lexicalForm ?d \n" + "WHERE{\n"
			+ "  ?synset <https://w3id.org/framester/wn/wn30/wndomains/synsetDomain> ?d .\n"
			+ "  ?synset <https://w3id.org/framester/wn/wn30/schema/containsWordSense> ?wordsense .\n"
			+ "  ?wordsense <https://w3id.org/framester/wn/wn30/schema/word> ?word .\n"
			+ "  ?word <https://w3id.org/framester/wn/wn30/schema/lexicalForm> ?lexicalForm .\n" + "} \n";

	private static final String endpoint = "http://localhost:3030/wordnet/query";

	public static Map<String, String> loadMapFromTSV(String filepath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		Map<String, String> result = new HashMap<>();
		br.lines().forEach(l -> {
			String[] row = l.split("\t");
			result.put(row[0], row[1]);
		});
		br.close();
		return result;
	}

	public static void main(String[] args) throws IOException, RocksDBException {
		Map<String, String> wnDomainToKD = loadMapFromTSV(
				"/Users/lgu/workspace/ekr/dome/resources/wn_resources/wn_domains2kd");

		Map<String, String> id2Domain = loadMapFromTSV(
				"/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/domain2id.tsv");

		Map<String, String> domain2id = HashBiMap.create(id2Domain).inverse();

//		System.out.println(QueryFactory.create(SELECT_WORDS_DOMAINS).toString(Syntax.defaultSyntax));
		QueryExecution qexec = QueryExecutionFactory.sparqlService(endpoint, QueryFactory.create(SELECT_WORDS_DOMAINS));

		RocksMap<String, String> lemmaToKDWeighted = new RocksMap<>(
				"/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_wn",
				new StringRocksTransformer(), new StringRocksTransformer());

		ResultSet rs = qexec.execSelect();
		ProgressCounter pc = new ProgressCounter();

		while (rs.hasNext()) {
			pc.increase();
			QuerySolution qs = rs.next();
			String domain = qs.getResource("d").getURI();
			String lemma = qs.get("lexicalForm").asLiteral().getValue().toString();

			if (wnDomainToKD.containsKey(domain)) {
//				System.out.println(lemma + " " + domain + " " + domain2id.get(wnDomainToKD.get(domain)));
				lemmaToKDWeighted.merge(lemma, (domain2id.get(wnDomainToKD.get(domain)) + " 1.0 "));
			} else {
				System.err.println("Can't find KD for " + domain);
			}

		}
		System.out.println("Number of lemmas "+lemmaToKDWeighted.keySet().size());
		lemmaToKDWeighted.close();
	}
	

}
