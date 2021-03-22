package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class SelectFiles {

	/**
	 * This script selects 100 virtual documents for running the experiments for
	 * domain classification
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("index.tsv"));
		List<String> lines = br.lines().collect(Collectors.toList());
		Collections.shuffle(lines);
		br.close();

		FileOutputStream fos = new FileOutputStream(new File("vdocs_for_domain_experiments.txt"));
		FileOutputStream fos_index = new FileOutputStream(new File("index_ontologies_domain_experiments.txt"));

		lines.subList(0, 100).forEach(l -> {
			String id = l.split("\t")[1];
//			String uri = l.split("\t")[1];
			System.out.println(l);
			try {
				fos_index.write((l + "\n").getBytes());
				fos.write(("/Users/lgu/workspace/ekr/vdg/src/main/resources/virtualDocuments/" + id
						+ "/virtualdocument.txt.bz2\n").getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
		});

		fos.flush();
		fos.close();

		fos_index.flush();
		fos_index.close();

	}

}
