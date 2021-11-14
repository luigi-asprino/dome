package it.cnr.istc.stlab;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.rocksdb.RocksDBException;

import it.cnr.istc.stlab.edwin.EquivalenceSetGraphLoader;
import it.cnr.istc.stlab.edwin.model.EquivalenceSetGraph;

public class ESGEX {
	public static void main(String[] args) throws RocksDBException, IOException {
		EquivalenceSetGraph esgClasses = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("/Users/lgu/Desktop/NOTime/EKR/ESGs/classes");
		EquivalenceSetGraph esgProperties = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("/Users/lgu/Desktop/NOTime/EKR/ESGs/properties");

		FileOutputStream fos = new FileOutputStream(
				new File("/Users/lgu/Desktop/NOTime/EKR/ESGs/classes_and_properties_with_size_ordered"));

		Map<String, Long> entity2size = new HashMap<>();

		esgClasses.getEntities().forEach(u -> {
			entity2size.put(u, esgClasses.getEntityDirectExtensionalSize(u));
//			try {
//				fos.write(u.getBytes());
//				fos.write('\t');
//				fos.write((esgClasses.getEntityDirectExtensionalSize(u) + "").getBytes());
//				fos.write('\n');
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
		});

		esgProperties.getEntities().forEach(u -> {
			entity2size.put(u, esgProperties.getEntityDirectExtensionalSize(u));
//				try {
//				fos.write(u.getBytes());
//				fos.write('\t');
//				fos.write((esgProperties.getEntityDirectExtensionalSize(u) + "").getBytes());
//				fos.write('\n');
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
		});

		System.out.println("done");

		sortByValueAsList(entity2size).forEach(e -> {
			try {
				fos.write(e.getKey().getBytes());
				fos.write('\t');
				fos.write((e.getValue() + "").getBytes());
				fos.write('\n');
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		});

//		System.out.println(esgClasses.getNumberOfEquivalenceSets());
//		System.out.println(esgProperties.getNumberOfEquivalenceSets());
//
//		System.out.println(esgProperties.getEmptyEquivalenceSets().size());
//		System.out.println(esgClasses.getEmptyEquivalenceSets().size());
//
//		{
//			FileOutputStream fos = new FileOutputStream(
//					new File("/Users/lgu/Desktop/NOTime/EKR/ESGs/emptyEntities_classes"));
//
//			esgClasses.getEmptyEquivalenceSets().forEach(l -> {
//				esgClasses.getEquivalenceSet(l).forEach(u -> {
//					try {
//						fos.write(u.getBytes());
//						fos.write('\n');
//					} catch (IOException e) {
//						e.printStackTrace();
//					}
//				});
//			});
//
//			fos.flush();
//			fos.close();
//		}
//
//		{
//			FileOutputStream fos = new FileOutputStream(
//					new File("/Users/lgu/Desktop/NOTime/EKR/ESGs/emptyEntities_properties"));
//
//			esgProperties.getEmptyEquivalenceSets().forEach(l -> {
//				esgProperties.getEquivalenceSet(l).forEach(u -> {
//					try {
//						fos.write(u.getBytes());
//						fos.write('\n');
//					} catch (IOException e) {
//						e.printStackTrace();
//					}
//				});
//			});
//
//			fos.flush();
//			fos.close();
//		}

	}

	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
		List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
		list.sort(Entry.comparingByValue());

		Map<K, V> result = new LinkedHashMap<>();
		for (Entry<K, V> entry : list) {
			result.put(entry.getKey(), entry.getValue());
		}

		return result;
	}

	public static <K, V extends Comparable<? super V>> List<Entry<K, V>> sortByValueAsList(Map<K, V> map) {
		List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
		list.sort(Entry.comparingByValue());
		return list;
	}
}
