
package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class URLCheck {

	public static void main(String[] args) throws Exception {
		
		System.out.println(String.format("%s %d", "http://courseware.rkbexplorer.com/ontologies/courseware", getStatusCode("http://courseware.rkbexplorer.com/ontologies/courseware")));
		
		BufferedReader br = new BufferedReader(new FileReader(new File("voc_list")));
		br.lines().forEach(l -> {
			try {
				System.out.println(String.format("%s %d", l, getStatusCode(l)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}

	public static int getStatusCode(String urlToRead) throws Exception {
		URL url = new URL(urlToRead);
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		
		conn.setRequestMethod("GET");

		return conn.getResponseCode();
	}

}
