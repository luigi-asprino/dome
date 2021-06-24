package it.cnr.istc.stlab;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.commons.compress.compressors.CompressorException;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.stream.JsonReader;

import it.cnr.istc.stlab.lgu.commons.io.streams.InputStreamFactory;
import it.cnr.istc.stlab.lgu.commons.misc.ProgressCounter;
import it.cnr.istc.stlab.rocksmap.RocksMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class WDReader {

	private static Logger logger = LoggerFactory.getLogger(WDReader.class);

	public static void main(String[] args) throws IOException, CompressorException, RocksDBException {
		logger.info("Reading");
		String folderOut = "resources";
		new File(folderOut).mkdir();
		RocksMap<String, String> wd_items = new RocksMap<>(folderOut + "/wd_items_1", new StringRocksTransformer(),
				new StringRocksTransformer(), true);
		RocksMap<String, String> categories = new RocksMap<>(folderOut + "/wd_categories_1",
				new StringRocksTransformer(), new StringRocksTransformer());
		RocksMap<String, String> labels = new RocksMap<>(folderOut + "/wd_labels_1", new StringRocksTransformer(),
				new StringRocksTransformer());
		RocksMap<String, String> bnId = new RocksMap<>(folderOut + "/wd_labels_bn_id_1", new StringRocksTransformer(),
				new StringRocksTransformer());
		RocksMap<String, String> wnId = new RocksMap<>(folderOut + "/wd_labels_wn_id_1", new StringRocksTransformer(),
				new StringRocksTransformer());

		computeCategoryMapGSON(wd_items, categories, labels, bnId, wnId, -1);

//		System.out.println(categories.get("Q31"));
//		System.out.println(labels.get("Q31"));
//		System.out.println(labels.get("Q5920298"));
//		System.out.println(labels.get("Q4366768"));

//		search("Q5920298");
	}

	public static void computeCategories() throws RocksDBException, CompressorException, IOException {
//		logger.info("Reading");
//		RocksMap<String, String> categories = new RocksMap<>("wd_categories", new StringRocksTransformer(),
//				new StringRocksTransformer(), true);
//		RocksMap<String, String> labels = new RocksMap<>("wd_labels", new StringRocksTransformer(),
//				new StringRocksTransformer());
//
//		RocksMap<String, String> wd_items = new RocksMap<>("wd_items", new StringRocksTransformer(),
//				new StringRocksTransformer());
//		RocksMap<String, String> bnId = new RocksMap<>("wd_labels_bn_id", new StringRocksTransformer(),
//				new StringRocksTransformer());
//		computeCategoryMapGSON(wd_items, categories, labels, bnId, -1);
//
//		categories.close();
//		labels.close();
	}

	public static void search(String idToSearch) throws CompressorException, IOException {
		InputStream is = InputStreamFactory.getInputStream("/Users/lgu/Desktop/NOTime/latest-all.json.bz2");
		InputStreamReader isr = new InputStreamReader(is);
		JsonReader jr = new JsonReader(isr);
		jr.beginArray();

		GsonBuilder gsonBuilder = new GsonBuilder();
		Gson gson = gsonBuilder.create();
		ProgressCounter pc = new ProgressCounter();
		pc.setPrintRate(true);
		JsonObject obj;
		while ((obj = gson.fromJson(jr, JsonObject.class)) != null) {
			pc.increase();
			try {
				if (getId(obj).equals(idToSearch)) {
					System.out.println(obj.toString());
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void computeCategoryMapGSON(RocksMap<String, String> wd_items, RocksMap<String, String> categories,
			RocksMap<String, String> labels, RocksMap<String, String> bnId, RocksMap<String, String> wnId, int breakAt)
			throws CompressorException, IOException {
		InputStream is = InputStreamFactory.getInputStream("/Users/lgu/Desktop/NOTime/latest-all.json.bz2");
		InputStreamReader isr = new InputStreamReader(is);
		JsonReader jr = new JsonReader(isr);
		jr.beginArray();

		GsonBuilder gsonBuilder = new GsonBuilder();
		Gson gson = gsonBuilder.create();
		ProgressCounter pc = new ProgressCounter();
		pc.setPrintRate(true);
		JsonObject obj;
		String cats;
		String id;
		String bnSyn, wnSyn;
		try {
			while ((obj = gson.fromJson(jr, JsonObject.class)) != null) {
				pc.increase();
				if (pc.currentValue() == breakAt) {
					break;
				}
				id = getId(obj);
//				if(id.equals("Q31")) {
//					FileOutputStream fos = new FileOutputStream("/Users/lgu/Desktop/Q31.json");
//					fos.write(obj.toString().getBytes());
//					fos.flush();
//					fos.close();
//					break;
//				}
//				wd_items.put(id, obj.toString());
				bnSyn = getBNSynset(obj);
				if (bnSyn != null) {
//					System.out.println(id + " - " + bnSyn);
					bnId.put(id, bnSyn);
				}

				wnSyn = getWNSynset(obj);
				if (wnSyn != null) {
//					System.out.println(id + " - " + wnSyn);
					wnId.put(id, wnSyn);
				}

				if (getLabel(obj, "en").length() > 0) {
					if ((cats = getCategories(obj)) != null) {
						categories.put(id, cats);
					}
					labels.put(id, obj.get("labels").getAsJsonObject().toString());

				}
			}
		} catch (IllegalArgumentException e) {
			logger.info("Terminated!");
		}

	}

	private static String getLabel(JsonObject obj, String lang) {
		try {
//			System.out.println(obj.get("labels").getAsJsonObject().toString());
			return obj.get("labels").getAsJsonObject().get(lang).getAsJsonObject().get("value").getAsString();
		} catch (NullPointerException e) {
			return "";
		}
	}

	private static String getId(JsonObject obj) {
		return obj.get("id").getAsString();
	}

	private static String getCategories(JsonObject obj) {
		try {
			return obj.get("claims").getAsJsonObject().get("P910").getAsJsonArray().toString();
		} catch (NullPointerException e) {

		}
		return null;
	}

	private static String getBNSynset(JsonObject obj) {
		try {
			return obj.get("claims").getAsJsonObject().get("P2581").getAsJsonArray().get(0).getAsJsonObject()
					.get("mainsnak").getAsJsonObject().get("datavalue").getAsJsonObject().get("value").getAsString();
		} catch (NullPointerException e) {

		}
		return null;
	}

	private static String getWNSynset(JsonObject obj) {
		try {
			return obj.get("claims").getAsJsonObject().get("P8814").getAsJsonArray().get(0).getAsJsonObject()
					.get("mainsnak").getAsJsonObject().get("datavalue").getAsJsonObject().get("value").getAsString();
		} catch (NullPointerException e) {

		}
		return null;
	}

}
