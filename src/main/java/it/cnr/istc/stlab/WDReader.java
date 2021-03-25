package it.cnr.istc.stlab;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.compress.compressors.CompressorException;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.jsoniter.JsonIterator;
import com.jsoniter.any.Any;
import com.jsoniter.output.EncodingMode;
import com.jsoniter.output.JsonStream;
import com.jsoniter.spi.DecodingMode;
import com.jsoniter.spi.JsonException;

import it.cnr.istc.stlab.lgu.commons.misc.ProgressCounter;
import it.cnr.istc.stlab.lgu.commons.streams.InputStreamFactory;
import it.cnr.istc.stlab.rocksmap.RocksMultiMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class WDReader {

	private static Logger logger = LoggerFactory.getLogger(WDReader.class);

	public static void main(String[] args) throws IOException, CompressorException, RocksDBException {
		logger.info("Reading");
		RocksMultiMap<String, String> categories = new RocksMultiMap<>("wd_categories", new StringRocksTransformer(),
				new StringRocksTransformer());

		computeCategoryMap(categories);

	}

	private static void computeCategoryMap(RocksMultiMap<String, String> categories)
			throws CompressorException, IOException, RocksDBException {
		InputStream is = InputStreamFactory.getInputStream("/Users/lgu/Desktop/NOTime/latest-all.json.bz2");
		byte[] b = new byte[ 1024 * 1024 * 10];
		JsonIterator ij = JsonIterator.parse(b);
		ij.reset(is);
		JsonStream.setMode(EncodingMode.DYNAMIC_MODE);
		JsonIterator.setMode(DecodingMode.DYNAMIC_MODE_AND_MATCH_FIELD_WITH_HASH);
		ProgressCounter pc = new ProgressCounter();
		pc.setSLF4jLogger(logger);
		while (ij.readArray()) {
			pc.increase();
//			System.out.println(pc.currentValue());
			Any obj = ij.readAny();
//			logger.info("{} {} {}", getId(obj), getEnLabel(obj), getCategories(obj));
			List<String> cats = getCategories(obj);
			if (!cats.isEmpty()) {
				categories.putAll(getId(obj), cats);
			}
//			ij.reset(is);
//			System.out.println(pc.currentValue() + " " + ij.currentBuffer().length());
			ij.reset(b);
		}
		categories.close();
	}

	private static String getEnLabel(Any obj) {
		return obj.get("labels").get("en").get("value").toString();
	}

	private static String getId(Any obj) {
		return obj.get("id").toString();
	}

	private static List<String> getCategories(Any obj) {
		List<String> result = new ArrayList<>();
		try {
			List<Any> categories = obj.get("claims").get("P910").asList();
			for (Any a : categories) {
				result.add(a.get("mainsnak").get("datavalue").get("value").get("id").toString());
			}
		} catch (JsonException e) {

		}
		return result;
	}

}
