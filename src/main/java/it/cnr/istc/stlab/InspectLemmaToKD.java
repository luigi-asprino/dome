package it.cnr.istc.stlab;

import org.rocksdb.RocksDBException;

import it.cnr.istc.stlab.rocksmap.RocksMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class InspectLemmaToKD {
	
	public static void main(String[] args) throws RocksDBException {
		RocksMap<String, String> lemmaToKDWeighted = new RocksMap<>(
				"/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_wn",
				new StringRocksTransformer(), new StringRocksTransformer());
		
		System.out.println(lemmaToKDWeighted.get("building"));
		
		lemmaToKDWeighted.close();
	}

}
