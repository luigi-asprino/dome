package it.cnr.istc.stlab;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.jena.ext.com.google.common.collect.Sets;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.riot.RDFDataMgr;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMultimap;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import it.cnr.istc.stlab.lgu.commons.misc.ProgressCounter;
import it.cnr.istc.stlab.rocksmap.RocksMap;
import it.cnr.istc.stlab.rocksmap.RocksMultiMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class LexiconWriter {

	private static final Logger logger = LoggerFactory.getLogger(LexiconWriter.class);

	public static void function() throws RocksDBException, IOException {
		String folder = "resources-new";
		new File(folder).mkdir();

		// Create output databases
		RocksMap<String, String> categories = new RocksMap<>(folder + "/wd_categories_1", new StringRocksTransformer(),
				new StringRocksTransformer());
		RocksMap<String, String> labels = new RocksMap<>(folder + "/wd_labels_1", new StringRocksTransformer(),
				new StringRocksTransformer());
		RocksMultiMap<String, String> mainTopicClassificaationClosure = new RocksMultiMap<>(
				folder + "/mainTopicClassificationClosure", new StringRocksTransformer(), new StringRocksTransformer());
		RocksMap<String, String> bnId = new RocksMap<>(folder + "/wd_labels_bn_id_1", new StringRocksTransformer(),
				new StringRocksTransformer());

		// Load BabelDomains and mappings to Knowledge Domains
		Map<String, Set<String>> babelDomains2EKR = loadBabelDomains2EKR(
				"BabelDomains_full/BabelDomains/BabelDomainsToEKR_Domains.txt");
		Map<String, Set<WeightedCategory>> bnDomains = loadBabelDomains(
				"BabelDomains_full/BabelDomains/babeldomains_babelnet.txt", babelDomains2EKR);

		// Load WordNet domains and mappings to Knowledge Domains
		Map<String, String> wndomains2EKR = loadWNDomains2EKR("resources/wn_resources/wn_domains2kd");
		Map<String, Set<String>> wn31domains = loadWN31domains("resources/wn_resources/wn31domains.ttl", wndomains2EKR);
		RocksMap<String, String> wnId = new RocksMap<>(folder + "/wd_labels_wn_id_1", new StringRocksTransformer(),
				new StringRocksTransformer());

//		System.out.println(wn31domains.keySet().iterator().next());
//		System.out.println(wn31domains.get("08965302-n"));

		Map<String, Integer> cat2id = new HashMap<>();

		String outFolder = folder + "/input_wd_7";
		new File(outFolder).mkdirs();
		String fileDomain2Id = outFolder + "/domain2id";
		String fileWordIDs = outFolder + "/wordIDs";
		String fileWord2id = outFolder + "/word2id";
		String fileWordDomainMatrix = outFolder + "/word_domain_matrix";
		String fileWordDomainMatrixBn = outFolder + "/word_domain_matrix_bn";
		String fileWordDomainMatrixWn = outFolder + "/word_domain_matrix_wn";

		ImmutableMultimap<String, String> subCategory2MainCategory = loadMap(mainTopicClassificaationClosure, cat2id,
				fileDomain2Id);
		Iterator<String> it = labels.keyIterator();
		List<String> labelsItem;
		String wd, cat, lab, wdcat, labelCat, synBnId, synWnId;
		Integer idCat, currentWord;
		JsonArray jsonArray;
		ImmutableCollection<String> mainCats;

		FileOutputStream fosWordsIDsFOS = new FileOutputStream(new File(fileWordIDs));
		FileOutputStream fosWord2IDFOS = new FileOutputStream(new File(fileWord2id));
		FileOutputStream fosWordDomainMatrixFOS = new FileOutputStream(new File(fileWordDomainMatrix));
		FileOutputStream fosWordDomainMatrixBnFOS = new FileOutputStream(new File(fileWordDomainMatrixBn));
		FileOutputStream fosWordDomainMatrixWnFOS = new FileOutputStream(new File(fileWordDomainMatrixWn));

		int max = -1;
		boolean print = max > 0;

		int wordCount = 0;
		int itemCount = 0;

		while (it.hasNext()) {
			
			if (itemCount % 10000 == 0) {
				logger.info("Processed {}", itemCount);
			}
			
			if (max-- == 0) {
				break;
			}

			itemCount++;
			wd = it.next();
			cat = categories.get(wd);
			lab = getEnLabel(labels.get(wd), wd).toLowerCase();
			labelsItem = getLabels(labels.get(wd), wd);

			if (lab == null)
				continue;

			for (String s : lab.split("/")) {
				labelsItem.add(s);
			}
			labelsItem.add(lab);

			// write for BN
			synBnId = bnId.get(wd);
			synWnId = wnId.get(wd);

			if (lab != null && (cat != null || (synBnId != null && bnDomains.containsKey(synBnId))
					|| (synWnId != null && wn31domains.containsKey(synWnId)))) {

				if (print) {
					System.out.println(wd + " " + synBnId + " " + synWnId);
				}

				currentWord = wordCount;
				wordCount++;
				fosWordsIDsFOS.write(String.format("%s\t%d\n", wd, currentWord).getBytes());
//				fosWord2IDFOS.write(String.format("%s\t%d\n", lab, currentWord).getBytes());

				for (String l : labelsItem) {
					if (!l.contains(" ")) {
						fosWord2IDFOS.write(String.format("%s\t%d\n", l, currentWord).getBytes());
					}
				}

				if (synBnId != null) {
					if (bnDomains.containsKey(synBnId)) {
						for (WeightedCategory dbpediaCategory : bnDomains.get(synBnId)) {
							idCat = cat2id.get(dbpediaCategory.getCategory());
							fosWordDomainMatrixBnFOS.write(String
									.format("%d\t%d\t%f\n", currentWord, idCat, dbpediaCategory.getScore()).getBytes());
						}
					}
				}

				if (synWnId != null) {
					if (print)
						System.out.println("WN id " + synWnId + " " + wd);
					if (wn31domains.containsKey(synWnId)) {
						for (String dbpediawncat : wn31domains.get(synWnId)) {
							idCat = cat2id.get(dbpediawncat);
							if (idCat != null) {
								fosWordDomainMatrixWnFOS
										.write(String.format("%d\t%d\n", currentWord, idCat).getBytes());
							} else {
								logger.error("Couldn't find {}", dbpediawncat);
							}
						}
					}
				}

				if (cat != null) {
					jsonArray = JsonParser.parseString(cat).getAsJsonArray();
					for (int i = 0; i < jsonArray.size(); i++) {
						wdcat = getCat(jsonArray.get(i).getAsJsonObject(), wd);
						if (wdcat == null)
							continue;
						labelCat = getEnLabel(labels.get(wdcat), wd);
						if (labelCat == null)
							continue;
						mainCats = subCategory2MainCategory.get(toDBpediaCat(labelCat));
						if (mainCats.size() == 0)
							continue;

						for (String mainCat : mainCats) {
							idCat = cat2id.get(mainCat);
							if (print) {
								System.out.println(String.format("%s\t%s\t%s\t%s", wd, lab, mainCat, labelsItem));
							}
							fosWordDomainMatrixFOS.write(String.format("%d\t%d\n", currentWord, idCat).getBytes());

						}

					}

				}
			}

		}

		fosWordDomainMatrixWnFOS.flush();
		fosWordDomainMatrixWnFOS.close();

		fosWordDomainMatrixBnFOS.flush();
		fosWordDomainMatrixBnFOS.close();

		fosWordsIDsFOS.flush();
		fosWordsIDsFOS.close();

		fosWord2IDFOS.flush();
		fosWord2IDFOS.close();

		categories.close();
		labels.close();
		bnId.close();
		wnId.close();
	}

	private static void rewriteWnCategories(Map<String, String> wndomains2ekr, String domain2id,
			String word_domain_matrix_in, String word_domain_matrix_out, Map<String, Integer> cat2id)
			throws IOException {
		logger.info("Rewriting WN domains");
		final ProgressCounter pc = new ProgressCounter();
		BufferedReader br = new BufferedReader(new FileReader(new File(domain2id)));
		Map<Integer, String> id2DomainWN = new HashMap<>();
		br.lines().forEach(l -> {
			String[] row = l.split("\t");
			id2DomainWN.put(Integer.parseInt(row[1]), row[0]);
		});
		br.close();
		BufferedReader brWordDomainMatrixIn = new BufferedReader(new FileReader(new File(word_domain_matrix_in)));
		FileOutputStream fos = new FileOutputStream(new File(word_domain_matrix_out));
		brWordDomainMatrixIn.lines().forEach(l -> {
			pc.increase();
			String[] row = l.split("\t");
			String ekrDomain = wndomains2ekr.get(id2DomainWN.get(Integer.parseInt(row[1])));
			try {
				fos.write(String.format("%s\t%d\n", row[0], cat2id.get(ekrDomain)).getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
		fos.flush();
		fos.close();

	}

	private static Map<String, Set<String>> loadWN31domains(String file, Map<String, String> wndomains2ekr) {
		Model m = ModelFactory.createDefaultModel();
		RDFDataMgr.read(m, file);
		logger.info("Loading {}: size: {}", file, m.size());
		Map<String, Set<String>> result = new HashMap<String, Set<String>>();
		m.listStatements(null, m.getProperty("https://w3id.org/framester/wn/wn30/wndomains/synsetDomain"),
				(RDFNode) null).forEach(s -> {
					Set<String> domains = result.get(s.getSubject().asResource().getURI());
					if (domains == null) {
						domains = new HashSet<>();
					}

					if (wndomains2ekr.get(s.getObject().asResource().getURI()) != null) {
						domains.add(wndomains2ekr.get(s.getObject().asResource().getURI()));
						result.put(
								s.getSubject().asResource().getURI().replace("https://w3id.org/framester/wn/wn31/", ""),
								domains);
					}
				});
		return result;
	}

	private static Map<String, String> loadWNDomains2EKR(String string) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(string)));
		Map<String, String> result = new HashMap<>();
		br.lines().forEach(l -> {
			String[] row = l.split("\t");
			result.put(row[0], row[1]);
		});
		br.close();
		return result;

	}

	public static String toDBpediaCat(String s) {
		return "http://dbpedia.org/resource/" + s.replaceAll(" ", "_");
	}

	public static String getEnLabel(String s, String id) {
		try {
			return JsonParser.parseString(s).getAsJsonObject().get("en").getAsJsonObject().get("value").getAsString();
		} catch (NullPointerException e) {
//			logger.error("Error while processing {}", id);
			return null;
		}
	}

	public static List<String> getLabels(String s, String id) {
		try {
			JsonObject obj = JsonParser.parseString(s).getAsJsonObject();
			List<String> result = new ArrayList<>(obj.keySet().size());
			for (String lang : obj.keySet()) {
				String l = obj.get(lang).getAsJsonObject().get("value").getAsString().toLowerCase();
				if (!l.contains(" ")) {
					result.add(l);
				}
			}
			return result;
		} catch (NullPointerException e) {
//			logger.error("Error while processing {}", id);
			return null;
		}
	}

	public static String getCat(JsonObject obj, String id) {
		try {
			return obj.getAsJsonObject("mainsnak").getAsJsonObject("datavalue").getAsJsonObject("value").get("id")
					.getAsString();
		} catch (NullPointerException e) {
//			logger.error("Error while processing {}", id);
			return null;
		}
//		return JsonParser.parseString(s).getAsJsonObject().get("en").getAsJsonObject().get("value").getAsString();
	}

	public static ImmutableMultimap<String, String> loadMap(
			RocksMultiMap<String, String> mainTopicClassificaationClosure, Map<String, Integer> cat2id,
			String fileDomain2Id) throws IOException {

		ImmutableMultimap.Builder<String, String> builder = ImmutableMultimap.builder();
		AtomicInteger i = new AtomicInteger();
		FileOutputStream fos = new FileOutputStream(new File(fileDomain2Id));

		mainTopicClassificaationClosure.keyIterator().forEachRemaining(category -> {
			logger.info("Loading {} {}", category, i.get());
			cat2id.put(category, i.get());
			try {
				fos.write(String.format("%s\t%d\n", category, i.get()).getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
			mainTopicClassificaationClosure.get(category).forEach(subcategory -> {
				builder.put(subcategory, category);
			});
			i.incrementAndGet();
		});
		fos.flush();
		fos.close();
		return builder.build();
	}

	public static Map<String, Set<WeightedCategory>> loadBabelDomains(String filepath,
			Map<String, Set<String>> babelDomains2EKR) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		Map<String, Set<WeightedCategory>> result = new HashMap<>();
		br.lines().forEach(l -> {
			String[] row = l.split("\t");
			Set<WeightedCategory> weightedCategories = new HashSet<>();
			babelDomains2EKR.get(row[1]);
			double score = Double.parseDouble(row[2].replace("*", ""));
			for (String c : babelDomains2EKR.get(row[1])) {
				weightedCategories.add(new WeightedCategory(c, score));
			}
			result.put(row[0].replace("bn:", ""), weightedCategories);
		});
		br.close();
		return result;

	}

	public static Map<String, Set<String>> loadBabelDomains2EKR(String filepath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		Map<String, Set<String>> result = new HashMap<>();
		br.lines().forEach(l -> {
			String[] row = l.split("\t");
			Set<String> ekrDomainsSet = Sets.newHashSet(row[1].split(" "));
			result.put(row[0], ekrDomainsSet);
		});
		br.close();
		return result;

	}

	public static void main(String[] args) throws RocksDBException, IOException {
		function();
	}

}
