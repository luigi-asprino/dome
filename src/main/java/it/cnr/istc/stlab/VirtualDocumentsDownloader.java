
package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class VirtualDocumentsDownloader {

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(
				new FileReader(new File("/Users/lgu/Desktop/NOTime/EKR/Laundromat_annotations/ids_annotations")));
		br.lines().forEach(l -> {
			new File(String.format("/Users/lgu/Desktop/NOTime/EKR/Laundromat_annotations/virtual_documents/%s/%s/",
					l.subSequence(0, 2), l)).mkdirs();
			System.out.println(String.format(
					"scp luigi@137.204.221.13:/media/4TB/luigi/LOD_Laundromat_VDs/%s/%s/virtualdocument.txt.bz2 "
							+ "/Users/lgu/Desktop/NOTime/EKR/Laundromat_annotations/virtual_documents/%s/%s/virtualdocument.txt.bz2",
					l.subSequence(0, 2), l, l.subSequence(0, 2), l, l.subSequence(0, 2), l));
		});
		br.close();
	}

}
