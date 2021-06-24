package it.cnr.istc.stlab;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.jena.ext.com.google.common.collect.Sets;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.cnr.istc.stlab.edwin.Edwin;
import it.cnr.istc.stlab.edwin.EquivalenceSetGraphBuilderParameters;
import it.cnr.istc.stlab.edwin.EquivalenceSetGraphLoader;
import it.cnr.istc.stlab.edwin.model.EquivalenceSetGraph;
import it.cnr.istc.stlab.lgu.commons.io.FileUtils;
import it.cnr.istc.stlab.rocksmap.RocksMultiMap;
import it.cnr.istc.stlab.rocksmap.transformer.StringRocksTransformer;

public class CategoryHierarchyAnalyser {

	private static final Logger logger = LoggerFactory.getLogger(CategoryHierarchyAnalyser.class);

	public static void main(String[] args) throws IOException, RocksDBException, CompressorException,
			InstantiationException, IllegalAccessException, ClassNotFoundException, IllegalArgumentException,
			InvocationTargetException, NoSuchMethodException, SecurityException {

//		computeESG();

//		EquivalenceSetGraph esg = EquivalenceSetGraphLoader
//				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat");
//		System.out.println(esg.getEquivalenceSetIdOfIRI("http://dbpedia.org/resource/Category:Industry"));
//		System.out.println(esg.getEquivalenceSet("http://dbpedia.org/resource/Category:Industry"));
//		System.out.println(esg.getEquivalenceSetIdOfIRI("http://dbpedia.org/resource/Category:10th-century_Doges_of_Venice"));
//		
//		Set<String> entities = new HashSet<>();
//		AtomicInteger i = new AtomicInteger();
//		esg.getEquivalenceSetIds().forEach(esId->{
//			entities.addAll(esg.getEquivalenceSet(esId));
//			if(esg.getEquivalenceSet(esId).size()>1) {
//				i.incrementAndGet();
//			}
//		});
//		
//		System.out.println(entities.size());
//		System.out.println(esg.getNumberOfObservedEntities());
//		System.out.println(i.get());
//		System.out.println(esg.getNumberOfEquivalenceSets());
//		computeESG();

//		getHierarchyMainCategories(0);
		
//		computeCategoryClosureForTopCategories();
		
//		computeCategoryClosureForTopCategories();
		computeCategoryClosureForTopCategoriesFinal(6);
	}

	public static void computeCategoryClosureForTopCategories() throws RocksDBException {
		EquivalenceSetGraph esg = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210401");
		
		EquivalenceSetGraph esgMarch = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210301");
		
		Set<String> industries = esgMarch.getEntitiesImplicityEquivalentToOrSubsumedBy("http://dbpedia.org/resource/Category:Industry_(economics)", false, 0);

		RocksMultiMap<String, String> mainTopicClassificaationClosure = new RocksMultiMap<>(
				"mainTopicClassificationClosure", new StringRocksTransformer(), new StringRocksTransformer());

		System.out.println(esg.getEquivalenceSetIdOfIRI("http://dbpedia.org/resource/Category:Academic_disciplines"));
		List<String> mainCategories = FileUtils.readFileToList("src/main/resources/aligned_categories");

//		System.out.println(esg.getTopLevelEquivalenceSets().size());

//		System.out.println(esg.getNumberOfEquivalenceSets());
//		System.out.println(esg.getNumberOfObservedEntities());

		final long numberOfObservedEntities = esg.getNumberOfObservedEntities();

		final Set<String> entities = new HashSet<>();
		AtomicInteger i = new AtomicInteger();

		while (entities.size() < (numberOfObservedEntities * 0.8)) {
			entities.clear();
			logger.info("Depth {}", i.get());
			mainCategories.forEach(cat -> {
				
				logger.info("Computing closure for {}", cat);
				industries.addAll(esgMarch.getEntitiesImplicityEquivalentToOrSubsumedBy("http://dbpedia.org/resource/Category:Industry_(economics)", false, i.get()));
				Set<String> closure = esg.getEntitiesImplicityEquivalentToOrSubsumedBy(cat, false, i.get());
				
				if(cat.equals("http://dbpedia.org/resource/Category:Industry_(economics)")) {
					logger.info("Number of implicit sub categories {}: {}", industries.size(), i.get());
					
				}else {
					
					logger.info("Number of implicit sub categories {}: {}", closure.size(), i.get());
				}
				
				mainTopicClassificaationClosure.putAll(cat, closure);
				entities.addAll(closure);
				entities.addAll(industries);
			});
			i.incrementAndGet();
			logger.info("Gathered entities: {} ", entities.size());
		}

		mainTopicClassificaationClosure.close();
	}

	public static void getHierarchyMainCategories(int depth) throws RocksDBException {
		EquivalenceSetGraph esgApril = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210401");
//		EquivalenceSetGraph esgMarch = EquivalenceSetGraphLoader
//				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210301");

//		RocksMultiMap<String, String> mainTopicClassificaationClosure = new RocksMultiMap<>(
//				"mainTopicClassificationClosure", new StringRocksTransformer(), new StringRocksTransformer());

//		Set<String> toDiscard = Sets.newHashSet("http://dbpedia.org/resource/Category:Ethics","http://dbpedia.org/resource/Category:Health");
		
		List<String> mainCategories = FileUtils.readFileToList("src/main/resources/main_topic_classifications");
//		mainCategories.removeAll(toDiscard);
		List<String> firstLevel = FileUtils.readFileToList("src/main/resources/main_topic_classifications_firstlevel");
//		firstLevel.removeAll(toDiscard);

		Set<String> mainCategoriesNarrowerLevels = Sets.newHashSet(mainCategories);
		mainCategoriesNarrowerLevels.removeAll(firstLevel);

		System.out.println(mainCategoriesNarrowerLevels);

		Map<String, Set<String>> h = new HashMap<>();
		
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Universe", "http://dbpedia.org/resource/Category:Nature");
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Objects", "http://dbpedia.org/resource/Category:Entities");
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Religion", "http://dbpedia.org/resource/Category:Culture");
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Knowledge", "http://dbpedia.org/resource/Category:Culture");
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Philosophical_concepts", "http://dbpedia.org/resource/Category:Philosophy");
//		esgApril.addSpecialization("http://dbpedia.org/resource/Category:Philosophical_concepts", "http://dbpedia.org/resource/Category:Philosophy");
		
		
		System.out.println(esgApril.getSuperEquivalenceSets("http://dbpedia.org/resource/Category:Universe"));

		mainCategoriesNarrowerLevels.forEach(subCat -> {
			Set<String> superEquivalenceEntities = Sets.intersection(esgApril.getSuperEquivalenceSets(subCat), Sets.newHashSet(mainCategories));
			h.put(subCat, superEquivalenceEntities);
		});
		
		Set<String> checkSet = new HashSet<>();
//		System.out.println(h);
		h.forEach((k,v)->{
			System.out.println(k+" "+v);
			if(!v.isEmpty()) {
				checkSet.add(k);
				checkSet.addAll(v);
			}
		});
		System.out.println(checkSet.size());

//		mainCategories.forEach(cat -> {
//			logger.info("Computing closure for {}", cat);
//			Set<String> closure = esgApril.getEntitiesImplicityEquivalentToOrSubsumedBy(cat, false, depth);
//			logger.info("Number of implicit sub categories {}: {}", closure.size(), depth);
//			mainTopicClassificaationClosure.putAll(cat, closure);
//		});
//		
//		logger.info("Overridding http://dbpedia.org/resource/Category:Human_nature");
//		Set<String> closure = esgMarch.getEntitiesImplicityEquivalentToOrSubsumedBy("http://dbpedia.org/resource/Category:Human_nature", false, depth);
//		logger.info("Number of implicit sub categories {}: {}", closure.size(), 7);
//		mainTopicClassificaationClosure.putAll("http://dbpedia.org/resource/Category:Human_nature", closure);
//
//		mainTopicClassificaationClosure.close();
	}

	public static void computeCategoryClosureForTopCategoriesFinal(int depth) throws RocksDBException {
		EquivalenceSetGraph esgApril = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210401");
		EquivalenceSetGraph esgMarch = EquivalenceSetGraphLoader
				.loadEquivalenceSetGraphFromFolder("src/main/resources/ESG_DBpedia_Cat_20210301");

		RocksMultiMap<String, String> mainTopicClassificaationClosure = new RocksMultiMap<>(
				"mainTopicClassificationClosure", new StringRocksTransformer(), new StringRocksTransformer());

		List<String> mainCategories = FileUtils.readFileToList("src/main/resources/aligned_categories");
		
		Set<String> categoriesGathered = new HashSet<>();

		mainCategories.forEach(cat -> {
			logger.info("Computing closure for {}", cat);
			Set<String> closure = esgApril.getEntitiesImplicityEquivalentToOrSubsumedBy(cat, false, depth);
			logger.info("Number of implicit sub categories {}: {}", closure.size(), depth);
			mainTopicClassificaationClosure.putAll(cat, closure);
			categoriesGathered.addAll(closure);
		});

		logger.info("Overridding http://dbpedia.org/resource/Category:Industry_(economics)");
		Set<String> closure = esgMarch.getEntitiesImplicityEquivalentToOrSubsumedBy(
				"http://dbpedia.org/resource/Category:Industry_(economics)", false, depth);
		logger.info("Number of implicit sub categories {}: {}", closure.size(), depth);
		mainTopicClassificaationClosure.putAll("http://dbpedia.org/resource/Category:Industry_(economics)", closure);
		categoriesGathered.addAll(closure);
		logger.info("Number of categories : {}",categoriesGathered.size());

		mainTopicClassificaationClosure.close();
	}

	public static void computeWDCategories() throws IOException, RocksDBException, CompressorException {
		WDReader.computeCategories();
	}

	public static void computeESG() throws IOException, RocksDBException, CompressorException, InstantiationException,
			IllegalAccessException, ClassNotFoundException, IllegalArgumentException, InvocationTargetException,
			NoSuchMethodException, SecurityException {
		EquivalenceSetGraphBuilderParameters esgbp = new EquivalenceSetGraphBuilderParameters();
		esgbp.setDatasetPaths(new String[] { "src/main/resources/categories_lang=en_skos2021-04-01.ttl.bz2" });
		esgbp.setEsgFolder("src/main/resources/ESG_DBpedia_Cat_20210401");
		esgbp.setEsgName("ESG_DBpedia_Cat_20210401");
		esgbp.setEsgBaseURI("base/");
		esgbp.setComputeEstimation(false);
		esgbp.setExportInRDFFormat(false);
		esgbp.setSpecializationPropertyToObserve("http://www.w3.org/2004/02/skos/core#broader");
//		esgbp.setEquivalencePropertyToObserve("http://www.w3.org/2004/02/skos/core#broader");
		Edwin.computeESG(esgbp);
	}

}
