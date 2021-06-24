package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

public class ExtractDatasetIds {

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(
				new FileReader(new File("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod/corpus_index")));
		FileOutputStream fos = new FileOutputStream("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod/datasetIds");
		String line;
		while ((line = br.readLine()) != null) {
			String[] a = line.split("\t");
			String[] b = a[0].split("/");
			fos.write(b[6].getBytes());
			fos.write('\n');
		}
		fos.flush();
		fos.close();
	}

}
