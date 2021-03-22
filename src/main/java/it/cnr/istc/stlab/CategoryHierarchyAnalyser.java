package it.cnr.istc.stlab;

import java.io.IOException;

import it.cnr.istc.stlab.edwin.Edwin;
import it.cnr.istc.stlab.edwin.RocksDBBackedEquivalenceSetGraph;

public class CategoryHierarchyAnalyser {

	public static void main(String[] args) throws IOException {
		RocksDBBackedEquivalenceSetGraph esg = Edwin.computeESG("src/main/resources/dbc_tim.properties");
		esg.getStats().toTSVFile("src/main/resources/stats.tsv");
	}

}
