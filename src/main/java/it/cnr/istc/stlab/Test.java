package it.cnr.istc.stlab;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.rdf.model.Model;
import org.rocksdb.RocksDBException;

import it.cnr.istc.stlab.rocksmap.RocksMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class Test {

	public static void main(String[] args) throws IOException, RocksDBException {
//		JsonIterator ij = JsonIterator.parse(new byte[1]);
//		ij.reset(new ByteArrayInputStream("[0,2,3]".getBytes()));
//		JsonStream.setMode(EncodingMode.DYNAMIC_MODE);
//		JsonIterator.setMode(DecodingMode.DYNAMIC_MODE_AND_MATCH_FIELD_WITH_HASH);
//		while(ij.readArray()) {
//			Any o = ij.readAny();
//			System.out.println(o.toString());
//		}
//
//		String[] a = new String[] { "BIRO.owl", "MODS.owl", "inferred_ARPENTEUR.ttl", "inferred_CIDOC-CRM.ttl",
//				"inferred_FRAD.ttl", "inferred_HOLDING.ttl", "inferred_POSTDATA_full.ttl", "inferred_VIR.ttl",
//				"CULTURAL-ON.owl", "OAENTRY.owl", "inferred_ATLAS-OF-PATHS.ttl", "inferred_CRM-ARCHAEO.ttl",
//				"inferred_FRBR-CORE.ttl", "inferred_ISBD.ttl", "inferred_PRESS-OO.ttl", "CWORK.owl", "SAWS.owl",
//				"inferred_BIBFRAME.ttl", "inferred_CULTURALIS.ttl", "inferred_FRBR-EXT.ttl", "inferred_MADS.ttl",
//				"inferred_RDA_full.ttl", "DOREMUS.owl", "SNAP.owl", "inferred_BIBLIOTEK-O.ttl", "inferred_EAC-CPF.ttl",
//				"inferred_FRBR-OO.ttl", "inferred_NOMISMA.ttl", "inferred_ROAR.ttl", "FENTRY.owl",
//				"inferred_ARCO_full.ttl", "inferred_BIBO.ttl", "inferred_EDM.ttl", "inferred_GND.ttl",
//				"inferred_OAD.ttl", "inferred_RiC-O.ttl", "LAWD.owl", "inferred_ARM_full.ttl", "inferred_BLTERMS.ttl",
//				"inferred_FABIO.ttl", "inferred_HICO.ttl", "inferred_OCSA.ttl", "inferred_VIAF.ttl" };
//		
//		FileOutputStream fos = new FileOutputStream(new File("/Users/lgu/Desktop/run.sh"));
//
//		for (String s1 : a) {
//			for (String s2 : a) {
//				if(!s1.equals(s2)) {
//					String cmd = "java -jar AgreementMakerLight.jar -s CH_dataset/" + s1 + " -t CH_dataset/" + s2
//							+ " -o Alignments/" + FilenameUtils.getBaseName(s1) + "_" + FilenameUtils.getBaseName(s2)
//							+ ".rdf -a\n";
//					fos.write(cmd.getBytes());
//				}
//			}
//		}
//		fos.flush();
//		fos.close();
		
//		String query = "PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
//				+ "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
//				+ "PREFIX  semiotics: <http://ontologydesignpatterns.org/cp/owl/semiotics.owl#>\n"
//				+ "PREFIX  earmark: <http://www.essepuntato.it/2008/12/earmark#>\n"
//				+ "PREFIX wndomains: <https://w3id.org/framester/wn/wn30/wndomains/>\n"
//				+ "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
//				+ "CONSTRUCT {?wn31 wndomains:synsetDomain ?domain}\n"
//				+ "WHERE {\n"
//				+ "?wn30syn  wndomains:synsetDomain ?domain  .\n"
//				+ "  ?wn31 owl:sameAs ?wn30syn .\n"
//				+ "}";
//		
//		QueryExecution qexec = QueryExecutionFactory.sparqlService("http://localhost:3030/wn/sparql", QueryFactory.create(query));
//		
//		Model m = qexec.execConstruct();
//		m.write(new FileOutputStream(new File("/Users/lgu/Desktop/wn31domains.ttl")),"TTL");
		
		RocksMap<String, String> wnId = new RocksMap<>( "resources/wd_labels_wn_id_1", new StringRocksTransformer(),
				new StringRocksTransformer());
		
		System.out.println(wnId.iterator().next().getValue()+" ");

	}

}
