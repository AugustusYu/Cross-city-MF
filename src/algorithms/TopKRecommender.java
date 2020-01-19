package algorithms;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import utility.Location;
import utility.POI;
import utils.CommonUtils;
import utils.Printer;
import data_structure.DenseVector;
import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.Crossrating;
import data_structure.DenseMatrix;
import utils.TopKPriorityQueue;

import java.util.Map;
import java.io.FileWriter;
/**
 * This is an abstract class for topK recommender systems.
 * Define some variables to use, and member functions to implement by a topK recommender.
 * 
 * @author HeXiangnan
 * @since 2014.12.03
 */
public abstract class TopKRecommender {
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	public int wordCount;
	/** Rating matrix of training set. Users by Items.*/
	public SparseMatrix trainMatrix;
	/** Test ratings. For showing progress only. */
	public ArrayList<Rating> testRatings;
	
	/** Position to cutoff. */
	public int topK = 100;
	/** Number of threads to run the model (if multi-thread implementation).*/
	public int threadNum = 1;
	
	/** Evaluation for each user (offline eval) or test instance (online eval).*/
	public DenseVector hits;
	public DenseVector ndcgs;
	public DenseVector precs;
	public int maxIterOnline = 1;
	public HashSet<Integer> newUsers = new HashSet<Integer>();
	public boolean ignoreTrain = true; // ignore train items when generating topK list
	public TopKRecommender() {};
	public int [] poicate;
	public int [][] poiwords;
	public double [] early_evaluate = new double [10] ; //ealry stop
	public int early_stop = 0;
	public Location [] poi_location_topk;
	public TopKRecommender(SparseMatrix trainMatrix, 
			ArrayList<Rating> testRatings, int topK, int threadNum) {
		this.trainMatrix = new SparseMatrix(trainMatrix);
		this.testRatings = new ArrayList<Rating>(testRatings);
		this.topK = topK;
		this.threadNum = threadNum;
		
		this.userCount = trainMatrix.length()[0];
		this.itemCount = trainMatrix.length()[1];
		
		for (int u = 0; u < userCount; u ++) {
			if (trainMatrix.getRowRef(u).itemCount() == 0)	newUsers.add(u);
		}
		for (int i = 0 ;i < 10;i++)
			early_evaluate[i] = 0;
		
	}
	
	/**
	 * Get the prediction score of user u on item i. To be overridden. 
	 */
	public abstract double predict(int u, int i);
	
	/**
	 * Build the model.
	 */
	public abstract void buildModel();
	
	public void readcategary(String filename) throws IOException{
		poicate = new int [itemCount];
		int poi = 0;
		int cate = 0;
		int count = 0;
		
		System.out.printf("begin read categary file!\n");
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line;
		while((line = reader.readLine()) != null) {
			String[] arr = line.split("\t");
			poi = Integer.parseInt(arr[0]);
			cate = Integer.parseInt(arr[1]);
			cate = cate/10000;
			if (poi < itemCount) {
				poicate[poi] = cate;
				count ++;
			}		
		}
		reader.close();
		System.out.printf("%d poi has categoty info\n", count);
	}
	
	public void readcategary_location_tencent(String filename,HashMap<Integer,POI> poiset ) throws IOException{
		poicate = new int [itemCount];
		int poi = 0;
		int cate = 0;
		int count = 0;
		poiwords = new int [itemCount][3];
		int [] words_index = new int [1000000];
		int wordnum = 0;
		for ( int i =0;i<1000000;i++)
			words_index[i] = -1;
		
		System.out.printf("begin read categary file!\n");
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line;
		poi_location_topk = new Location[itemCount];
		while((line = reader.readLine()) != null) {
			String[] arr = line.split("\t");
			poi = Integer.parseInt(arr[0]);
			cate = Integer.parseInt(arr[1]);
			cate = cate;
			if (words_index[(int)cate/10000]==-1) {
				words_index[(int)cate/10000] = wordnum;
				wordnum ++;
			}
			poiwords[poi][0] = words_index[(int)cate/10000];
			if (words_index[(int)cate/100]==-1) {
				words_index[(int)cate/100] = wordnum;
				wordnum ++;
			}
			poiwords[poi][1] = words_index[(int)cate/100];	
			if (words_index[(int)cate]==-1) {
				words_index[(int)cate] = wordnum;
				wordnum ++;
			}
			poiwords[poi][2] = words_index[(int)cate];
			
			Location l = new Location();
			l.latitude = Double.parseDouble(arr[2]);
			l.longitude = Double.parseDouble(arr[3]);
			poi_location_topk[poi]  = l;
			if (poi < itemCount) {
				poicate[poi] = cate;
				count ++;
			}		
			
			ArrayList<Integer> list=new ArrayList<Integer>();
			for (int j=0;j<3;j++) {				
				list.add(poiwords[poi][j]);
				}
			 POI p=new POI();
			 p.poi_id=poi;
			 p.location=poi_location_topk[poi];
			 p.wordset=list;
			 poiset.put(poi, p);
			
		}
		reader.close();
		System.out.printf("%d poi has categoty info\n", count);
		System.out.printf("%d categary words has beed found\n", wordnum);
		this.wordCount = wordnum;
	}
	
	public void readcategary_location(String filename) throws IOException{
		poicate = new int [itemCount];
		int poi = 0;
		int cate = 0;
		int count = 0;
		poiwords = new int [itemCount][3];
		int [] words_index = new int [1000000];
		int wordnum = 0;
		for ( int i =0;i<1000000;i++)
			words_index[i] = -1;
		
		System.out.printf("begin read categary file!\n");
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line;
		poi_location_topk = new Location[itemCount];
		while((line = reader.readLine()) != null) {
			String[] arr = line.split("\t");
			poi = Integer.parseInt(arr[0]);
			cate = Integer.parseInt(arr[1]);
			cate = cate;
			if (words_index[(int)cate/10000]==-1) {
				words_index[(int)cate/10000] = wordnum;
				wordnum ++;
			}
			poiwords[poi][0] = words_index[(int)cate/10000];
			if (words_index[(int)cate/100]==-1) {
				words_index[(int)cate/100] = wordnum;
				wordnum ++;
			}
			poiwords[poi][1] = words_index[(int)cate/100];	
			if (words_index[(int)cate]==-1) {
				words_index[(int)cate] = wordnum;
				wordnum ++;
			}
			poiwords[poi][2] = words_index[(int)cate];
			
			Location l = new Location();
			l.latitude = Double.parseDouble(arr[2]);
			l.longitude = Double.parseDouble(arr[3]);
			poi_location_topk[poi]  = l;
			if (poi < itemCount) {
				poicate[poi] = cate;
				count ++;
			}		
		}
		reader.close();
		System.out.printf("%d poi has categoty info\n", count);
		System.out.printf("%d categary words has beed found\n", wordnum);
		this.wordCount = wordnum;
	}
	
	public void readcategary_location_yelp(String filename,HashMap<Integer,POI> poiset) throws IOException{
		poicate = new int [itemCount];
		int poi = 0;
		String cate ;
		int count = 0;

		int [] words_index = new int [1000000];
		int wordnum = 0;
		for ( int i =0;i<1000000;i++)
			words_index[i] = -1;
		
		System.out.printf("begin read categary file!\n");
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line;
		poi_location_topk = new Location[itemCount];
		while((line = reader.readLine()) != null) {
			String[] arr = line.split("\t");
			poi = Integer.parseInt(arr[0]);
			Location l = new Location();
			l.latitude = Double.parseDouble(arr[1]);
			l.longitude = Double.parseDouble(arr[2]);
			poi_location_topk[poi]  = l;
			if (poi < itemCount) {
				count ++;
			}		
			cate = arr[3];
			String[] con = cate.split("\\|");
			ArrayList<Integer> list=new ArrayList<Integer>();
			for (int j=0;j<con.length;j++) {				
				list.add(Integer.parseInt(con[j]));
				if(wordnum<Integer.parseInt(con[j])) 
					wordnum = Integer.parseInt(con[j]);
				}
			 POI p=new POI();
			 p.poi_id=poi;
			 p.location=poi_location_topk[poi];
			 p.wordset=list;
			 poiset.put(poi, p);
		}
		reader.close();
		wordnum ++;
		System.out.printf("%d poi has categoty info\n", count);
		System.out.printf("%d categary words has beed found\n", wordnum);
		
		this.wordCount = wordnum;
	}
	
	public void readcategary(String filename,int mode) throws IOException{
		poicate = new int [itemCount];
		int poi = 0;
		int cate = 0;
		int count = 0;
		
		System.out.printf("begin read categary file!\n");
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line;
		while((line = reader.readLine()) != null) {
			String[] arr = line.split("\t");
			poi = Integer.parseInt(arr[0]);
			cate = Integer.parseInt(arr[1]);
			if (mode == 1)
				cate = cate/10000;
			else if (mode ==2) 
				cate = cate/100;
			else if (mode == 3)
				;
			
			if (poi < itemCount) {
				poicate[poi] = cate;
				count ++;
			}		
		}
		reader.close();
		System.out.printf("%d poi has categoty info\n", count);
	}
	
	
	/**
	 * Update the model with a new observation. 
	 */
	public abstract void updateModel(int u, int i);
	
	/**
	 * Show progress (evaluation) with current model parameters. 
	 * @iter	Current iteration
	 * @start	Starting time of the iteration
	 * @testMatrix	For evaluation purpose
	 */
	public void showProgress(int iter, long start, ArrayList<Rating> testRatings) {
		long end_iter = System.currentTimeMillis();
		if (userCount == testRatings.size())  // leave-1-out eval
			 evaluate(testRatings);
		else	// global split
			 evaluateOnline(testRatings, 100);
		long end_eval = System.currentTimeMillis();
		
		System.out.printf("Iter=%d[%s] <loss, hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f\t %.4f\t [%s]\n",
				iter, Printer.printTime(end_iter - start), loss(),
				hits.mean(), ndcgs.mean(), precs.mean(), Printer.printTime(end_eval - end_iter));
	}
	
	/**
	 * Online evaluation (global split) by simulating the testing stream. 
	 * @param ratings Test ratings that are sorted by time (old -> recent).
	 * @param interval Print evaluation result per X iteration. 
	 */
	public void evaluateOnline(ArrayList<Rating> testRatings, int interval) {
		int testCount = testRatings.size();
		hits = new DenseVector(testCount);
		ndcgs = new DenseVector(testCount);
		precs = new DenseVector(testCount);
		
		// break down the results by number of user ratings of the test pair
		int intervals = 10;
		int[] counts = new int[intervals + 1];
		double[] hits_r = new double[intervals + 1];
		double[] ndcgs_r = new double[intervals + 1];
		double[] precs_r = new double[intervals + 1];
		
		Long updateTime = (long) 0;
		for (int i = 0; i < testCount; i ++) {
			// Check performance per interval:
			if (i > 0 && interval > 0 && i % interval == 0) {
				System.out.printf("%d: <hr, ndcg, prec> =\t %.4f\t %.4f\t %.4f\n", 
						i, hits.sum() / i, ndcgs.sum() / i, precs.sum() / i);
			}
			// Evaluate model of the current test rating:
			Rating rating = testRatings.get(i);
			double[] res = this.evaluate_for_user(rating.userId, rating.itemId);
			hits.set(i, res[0]);
			ndcgs.set(i, res[1]);
			precs.set(i, res[2]);
			
			// statisitcs for break down
			int r = trainMatrix.getRowRef(rating.userId).itemCount();
			r =  r> intervals ? intervals : r;
			counts[r] += 1;
			hits_r[r] += res[0];
			ndcgs_r[r] += res[1];
			precs_r[r] += res[2];
			
			// Update the model
			Long start = System.currentTimeMillis();
			updateModel(rating.userId, rating.itemId);
			updateTime += (System.currentTimeMillis() - start);
		}
		
		System.out.println("Break down the results by number of user ratings for the test pair.");
		System.out.printf("#Rating\t Percentage\t HR\t NDCG\t MAP\n");
		for (int i = 0; i <= intervals; i ++) {
			System.out.printf("%d\t %.2f%%\t %.4f\t %.4f\t %.4f \n", 
					i, (double)counts[i] / testCount * 100, 
					hits_r[i] / counts[i], ndcgs_r[i] / counts[i], precs_r[i] / counts[i]);
		}
		
		System.out.printf("Avg model update time per instance: %.2f ms\n", (float)updateTime/testCount);
	}
	
	protected ArrayList<Integer> threadSplit(int total, int threadNum, int t) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		int start = (total / threadNum) * t;
		int end = (t == threadNum-1) ? total : 
			(total / threadNum) * (t + 1);
		for (int i = start; i < end; i ++)
			res.add(i);
		return res;
	}
	
	/**
	 * Offline evaluation (leave-1-out) for each user.
	 * @param topK position to cutoff
	 * @param testMatrix
	 * @throws InterruptedException 
	 */
	public void evaluate(ArrayList<Rating> testRatings) {
		assert userCount == testRatings.size();
		for (int u = 0; u < userCount; u ++)
			assert u == testRatings.get(u).userId;
		
		hits = new DenseVector(userCount);
		ndcgs = new DenseVector(userCount);
		precs = new DenseVector(userCount);
		
		// Run the evaluation multi-threads splitted by users
		EvaluationThread[] threads = new EvaluationThread[threadNum];
		for (int t = 0; t < threadNum; t ++) {
			ArrayList<Integer> users = threadSplit(userCount, threadNum, t);
			threads[t] = new EvaluationThread(this, testRatings, users);
			threads[t].start();
		}
		
		// Wait until all threads are finished.
		for (int t = 0; t < threads.length; t++) { 
		  try {
				threads[t].join();
			} catch (InterruptedException e) {
				System.err.println("InterruptException was caught: " + e.getMessage());
			}
		}
	}
	
	public double evaluatehr(ArrayList<Rating> testRatings) {
		int num = testRatings.size();
		double hrsum = 0;
		for (int i=0;i<num;i++) {
			int user = testRatings.get(i).userId;
			int item = testRatings.get(i).itemId;
			hrsum += evaluate_for_user(user,item)[0];
		}
		return hrsum/(double)num;
	}
	
	public void extraevaluate(ArrayList<Rating> testRatings,int mytopk) {
		DenseVector map = new DenseVector(userCount);
		DenseVector mrr = new DenseVector(userCount);
		DenseVector ndcg = new DenseVector(userCount);
		for (int u = 0;u<userCount;u++) {
			ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i = 0; i < itemCount; i++) {
				double score = predict(u, i);
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, mytopk, trainMatrix.getRowRef(u).indexList());
			double maps = 0;
			int item = testRatings.get(u).itemId;
			ndcg.set(u, getNDCG( rankList, item));
			mrr.set(u, getPrecision( rankList, item));
			for (int j = 0;j<itemList.size();j++) {
				item = itemList.get(j);
				maps += getPrecision( rankList, item);
			}
			/*
			if (itemList.size()>0)
				map.set(u, maps/itemList.size());
			else
				map.set(u, 0);		*/	
		}
		System.out.printf("extra evaluate  all <ndcg,map,mrr>: %d\t %.4f\t %.4f\n",
				mytopk,ndcg.mean(),mrr.mean());
	}
	
	public void evaluatefor82(ArrayList<Rating> testRatings,long start) {
		//long end_iter = System.currentTimeMillis();
		DenseVector hr = new DenseVector(userCount);
		DenseVector map = new DenseVector(userCount);
		DenseVector mrr = new DenseVector(userCount);
		DenseVector ndcg = new DenseVector(userCount);
		int factnum = userCount;
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
			
		for (int u = 0;u<userCount;u++) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				ndcg.set(u, 0);
				map.set(u, 0);
				mrr.set(u, 0);
				hr.set(u,0);
				continue;
			}		
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i = 0; i < itemCount; i++) {
				double score = predict(u, i);
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, itemCount-trainMatrix.getRowRef(u).indexList().size(), trainMatrix.getRowRef(u).indexList());
			
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				ndcg.set(u, 0);
				map.set(u, 0);
				mrr.set(u, 0);
				hr.set(u,0);
			}
			else {
				double maps = 0;
				double mrrs = 0;
				double ndcgs = 0;
				double hrs = 0;
				double idcg = 0;
				double[] rank = new double [testList.size()];
				
				for (int i = 0;i<testList.size();i++) {
					int item = testList.get(i);
					rank[i] = -1;
					for (int j = 0;j<rankList.size();j++) {
						if (rankList.get(j)==item) {
							ndcgs += Math.log(2) / Math.log(j+2);
							rank[i] = j;
							
						}
					}			
				}
				Arrays.sort(rank);
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK)
						hrs++;
				}			
				mrrs = 1/(rank[0]+1);
				ndcg.set(u, ndcgs/idcg);
				map.set(u, maps/testList.size());
				mrr.set(u, mrrs);
				hr.set(u,hrs);
			}
		}
		long end_eval = System.currentTimeMillis();
		System.out.printf("extra evaluate for all [%s]<hr,ndcg,map,mrr>:%.4f, %.4f, %.4f, %.4f\n",
				Printer.printTime(end_eval - start),hr.sum()/testcount,ndcg.sum()/factnum,map.sum()/factnum,mrr.sum()/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
	}
	
	public void evaluatefor82crosscity_showFactnum(ArrayList<Rating> testRatings,
			HashSet<Integer> Users,HashSet<Integer> Pois) {
		int factnum = Users.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testRatings.size(); i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		for (int u:Users) {
			if (newUsers.contains(u)) {
				factnum -- ;
				continue;
				}
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0)
				factnum --;
		}
		System.out.printf("the factnum (real test user) is : %d\n",factnum);
	}
	
	public void evaluatefor82crosscity(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i : Pois) {
				double score = predict(u, i);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
			ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
			for (int i :trainlist)
				if (Pois.contains(i))
					tourpoilist.add(i);
			
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
			//ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, topK, tourpoilist);
			
			
			if(true) {
				double maps = 0;
				double mrrs = 0;
				double ndcgs = 0;
				double hrs = 0;
				double idcg = 0;
				double[] rank = new double [testList.size()];
				
				for (int i = 0;i<testList.size();i++) {
					int item = testList.get(i);
					rank[i] = -1;
					for (int j = 0;j<rankList.size();j++) {
						if (rankList.get(j)==item) {
							ndcgs += Math.log(2) / Math.log(j+2);
							rank[i] = j;
							break;
						}
					}			
				}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		}
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
	}


	
	public void evaluatefor82multicity(ArrayList<Rating> testRatings,long start,
			int [][] city_pois, int[] ucity, int[] pcity) {
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = userCount;
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		
		for (int u= 0;u<userCount;u++) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			int city = ucity[u];	
			for (int i = 0; i < itemCount; i++) {
				if (pcity[i] == city)
					continue;
				
				double score = predict(u, i);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, itemCount-trainMatrix.getRowRef(u).indexList().size(), trainMatrix.getRowRef(u).indexList());
			
			for (int j =0;j<testList.size();j++) {
				for (int i =0;i<rankList.size();i++) {
					int poi_test = testList.get(j);
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}				
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u= 0;u<userCount;u++) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("evaluate for  multicity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
	}

	public void evaluatefor82multicity_stlda(ArrayList<Rating> testRatings,long start,
			int [][] city_pois, int[] ucity, int[] pcity) {
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = userCount;
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		
		for (int u= 0;u<userCount;u++) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			int choose_poi = testList.get(0);
			int choose_poi_region = get_region(choose_poi);
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			int city = ucity[u];	
			for (int i = 0; i < itemCount; i++) {
				if (pcity[i] == city)
					continue;
				double score = predict_region(u, i,choose_poi_region);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, itemCount-trainMatrix.getRowRef(u).indexList().size(), trainMatrix.getRowRef(u).indexList());
			
			for (int j =0;j<testList.size();j++) {
				for (int i =0;i<rankList.size();i++) {
					int poi_test = testList.get(j);
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}				
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u= 0;u<userCount;u++) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("evaluate for  multicity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
	}
	
	
	public void evaluatefor82crosscity_lda(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
		
		
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i : Pois) {
				double score = predict(u, i);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
			ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
			for (int i :trainlist)
				if (Pois.contains(i))
					tourpoilist.add(i);
			
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
			//ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, topK, tourpoilist);
			
			
			if(true) {
				double maps = 0;
				double mrrs = 0;
				double ndcgs = 0;
				double hrs = 0;
				double idcg = 0;
				double[] rank = new double [testList.size()];
				
				for (int i = 0;i<testList.size();i++) {
					int item = testList.get(i);
					rank[i] = -1;
					for (int j = 0;j<rankList.size();j++) {
						if (rankList.get(j)==item) {
							ndcgs += Math.log(2) / Math.log(j+2);
							rank[i] = j;
							break;
						}
					}			
				}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		}
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
		if (ans[1]<= early_evaluate[1] && ans[2] <= early_evaluate[2])
			early_stop ++;
		else
			early_stop = 0;
		early_evaluate [1] = ans[1];
		early_evaluate [2] = ans[2];
		if (early_stop > 5) {
			System.out.printf("early stop!\n");
			System.exit(0);
		}
			
	}

	public void evaluatefor82crosscity_stlda_region(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
		
		
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			int region = 0;
			for (int j =0;j<testList.size();j++) {
				HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
				int poi_test = testList.get(j);
				region = get_region(poi_test);
				Location userlo = poi_location_topk[testList.get(j)];
				double la = userlo.latitude;
				double lo = userlo.longitude;
				for (int i : Pois) {
					//double score = predict_currentlocal(u, i,userlo);
					double score = predict_region(u, i,region);
//					if (u<2&&i<2)
//						System.out.printf("user:%d,poi:%d,score:%f,predict:%f\n", u,i,score,predict(u,i));
					if (Double.isNaN(score)) {
						System.out.printf("Nan score has been found in evaluate\n");
						System.exit(0);
					}
					map_item_score.put(i, score);
				}
				ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
				for (int i :trainlist)
					if (Pois.contains(i))
						tourpoilist.add(i);
				ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
				for (int i =0;i<rankList.size();i++) {
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
		if (ans[1]<= early_evaluate[1] && ans[2] <= early_evaluate[2])
			early_stop ++;
		else
			early_stop = 0;
		early_evaluate [1] = ans[1];
		early_evaluate [2] = ans[2];
		if (early_stop > 5) {
			System.out.printf("early stop!\n");
			//System.exit(0);
		}
			
	}

	public void evaluatefor82crosscity_stlda_region_output(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois,String outpath) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
		
		
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			for (int j =0;j<testList.size();j++) {
				HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
				int poi_test = testList.get(j);
				Location userlo = poi_location_topk[testList.get(j)];
				double la = userlo.latitude;
				double lo = userlo.longitude;
				for (int i : Pois) {
					double score = predict_currentlocal(u, i,userlo);
//					if (u<2&&i<2)
//						System.out.printf("user:%d,poi:%d,score:%f,predict:%f\n", u,i,score,predict(u,i));
					if (Double.isNaN(score)) {
						System.out.printf("Nan score has been found in evaluate\n");
						System.exit(0);
					}
					map_item_score.put(i, score);
				}
				ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
				for (int i :trainlist)
					if (Pois.contains(i))
						tourpoilist.add(i);
				ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
				for (int i =0;i<rankList.size();i++) {
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
				myout.write(Double.toString(hr[u]));
				myout.write('\t');
				myout.write(Double.toString(ndcg[u]));
				myout.write('\n');
			}
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
		if (ans[1]<= early_evaluate[1] && ans[2] <= early_evaluate[2])
			early_stop ++;
		else
			early_stop = 0;
		early_evaluate [1] = ans[1];
		early_evaluate [2] = ans[2];
		if (early_stop > 5) {
			System.out.printf("early stop!\n");
			System.exit(0);
		}
			
	}

	public void evaluatefororder_output(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois,String outpath) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			for (int j =0;j<testList.size();j++) {
				HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
				int poi_test = testList.get(j);
				for (int i : Pois) {
					double score = predict(u, i);
//					if (u<2&&i<2)
//						System.out.printf("user:%d,poi:%d,score:%f,predict:%f\n", u,i,score,predict(u,i));
					if (Double.isNaN(score)) {
						System.out.printf("Nan score has been found in evaluate\n");
						System.exit(0);
					}
					map_item_score.put(i, score);
				}
				ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
				for (int i :trainlist)
					if (Pois.contains(i))
						tourpoilist.add(i);
				ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
				for (int i =0;i<rankList.size();i++) {
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}				
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
				myout.write('\n');
			}
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
		if (ans[1]<= early_evaluate[1] && ans[2] <= early_evaluate[2])
			early_stop ++;
		else
			early_stop = 0;
		early_evaluate [1] = ans[1];
		early_evaluate [2] = ans[2];
		if (early_stop > 5) {
			System.out.printf("early stop!\n");
			System.exit(0);
		}		
	}
	
	public void evaluatefororder_output_multi(ArrayList<Rating> testRatings,long start,
			int [][] city_pois, int[] ucity,String outpath, int []pcity) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = userCount;
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u= 0;u<userCount;u++) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			int city = ucity[u];	
			for (int i = 0; i < itemCount; i++) {
				if (pcity[i] == city)
					continue;
				
				double score = predict(u, i);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, itemCount-trainMatrix.getRowRef(u).indexList().size(), trainMatrix.getRowRef(u).indexList());
			
			for (int j =0;j<testList.size();j++) {
				for (int i =0;i<rankList.size();i++) {
					int poi_test = testList.get(j);
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}				
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
				myout.write('\n');
			}
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u= 0;u<userCount;u++) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);		
	}
	
	public void evaluatefororder_output_multi_stlda(ArrayList<Rating> testRatings,long start,
			int [][] city_pois, int[] ucity,String outpath, int []pcity) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = userCount;
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u= 0;u<userCount;u++) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			int choose_poi = testList.get(0);
			int choose_poi_region = get_region(choose_poi);
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			int city = ucity[u];	
			for (int i = 0; i < itemCount; i++) {
				if (pcity[i] == city)
					continue;
				
				double score = predict_region(u, i,choose_poi_region);
				if (Double.isNaN(score)) {
					System.out.printf("Nan score has been found in evaluate\n");
					System.exit(0);
				}
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, itemCount-trainMatrix.getRowRef(u).indexList().size(), trainMatrix.getRowRef(u).indexList());
			
			for (int j =0;j<testList.size();j++) {
				for (int i =0;i<rankList.size();i++) {
					int poi_test = testList.get(j);
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}				
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
				myout.write('\n');
			}
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u= 0;u<userCount;u++) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);		
	}
	
	public void evaluatefororder_output_stlda(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois,String outpath) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
			
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			double maps = 0;
			double mrrs = 0;
			double ndcgs = 0;
			double hrs = 0;
			double idcg = 0;
			double[] rank = new double [testList.size()];
			int region = 0;
			for (int j =0;j<testList.size();j++) {
				HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
				int poi_test = testList.get(j);
				for (int i : Pois) {
					region = get_region(poi_test);
					double score = predict_region(u, i, region);
//					if (u<2&&i<2)
//						System.out.printf("user:%d,poi:%d,score:%f,predict:%f\n", u,i,score,predict(u,i));
					if (Double.isNaN(score)) {
						System.out.printf("Nan score has been found in evaluate\n");
						System.exit(0);
					}
					map_item_score.put(i, score);
				}
				ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
				for (int i :trainlist)
					if (Pois.contains(i))
						tourpoilist.add(i);
				ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
				for (int i =0;i<rankList.size();i++) {
					if (rankList.get(i) ==poi_test) {
						rank[j] = i;
						ndcgs += Math.log(2) / Math.log(i+2);
						break;
					}				
				}			
			}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
//				myout.write(Double.toString(hr[u]));
//				myout.write('\t');
//				myout.write(Double.toString(ndcg[u]));
				myout.write('\n');
			}
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
		}		
		
		System.out.printf("extra evaluate for crosscity [%s]<hr,ndcg,map,mrr,ndcg_topk>:%.4f, %.4f, %.4f, %.4f,%.4f\n",
				Printer.printTime(end_eval - start),ans[1]/testcount,ans[2]/factnum,ans[3]/factnum,ans[0]/factnum,ans[4]/factnum);
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
		if (ans[1]<= early_evaluate[1] && ans[2] <= early_evaluate[2])
			early_stop ++;
		else
			early_stop = 0;
		early_evaluate [1] = ans[1];
		early_evaluate [2] = ans[2];
		if (early_stop > 5) {
			System.out.printf("early stop!\n");
			System.exit(0);
		}
			
	}
	
	public  double predict_currentlocal(int u, int i, Location l) {
		return 0;
	};

	public  double predict_region(int u, int i, int l) {
		return 0;
	};
	
	public int get_region(int i ) {
		return 1;
	}
	

	public void output_testrecord(String outpath) {
		FileWriter out1 = null;
		try {
		out1 = new FileWriter(outpath);
		String str_head = "null";
		for (int u =0;u<userCount;u++)
			{

			}	
		out1.close();
		System.out.printf("finish write test data in %s\n",outpath);
		}
		catch (Exception e) {   

            e.printStackTrace();   

        }
	}
		
	public void evaluatefor82crosscity_output(ArrayList<Rating> testRatings,long start,
			HashSet<Integer> Users,HashSet<Integer> Pois,String outpath) {
		//long end_iter = System.currentTimeMillis();
		double []hr = new double [userCount];
		double []map = new double [userCount];
		double []mrr = new double [userCount];
		double []ndcg = new double [userCount];
		double []ndcg_topk = new double [userCount];
		int factnum = Users.size();
		
		
		int testcount = testRatings.size();
		ArrayList<ArrayList<Integer>> test = new ArrayList<ArrayList<Integer>>();
		for (int u = 0 ;u<userCount;u++) {
			ArrayList<Integer> a = new ArrayList<Integer>();;
			test.add(a);
		}
		for (int i = 0 ; i < testcount; i++) {	
				test.get(testRatings.get(i).userId).add(testRatings.get(i).itemId);
		}
		
		FileWriter myout = null;
		try {
		myout = new FileWriter(outpath);
		
		for (int u :Users) {
			if (newUsers.contains(u)) {
				factnum --;
				testcount -= test.get(u).size();
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u]=0;
				continue;
			}		
			ArrayList<Integer> testList = test.get(u);
			if (testList.size()==0) {
				factnum --;
				hr[u]=0;
				map[u]=0;
				mrr[u]=0;
				ndcg[u]=0;
				ndcg_topk[u] = 0;
				continue ; 
			}
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i : Pois) {
				double score = predict(u, i);
				map_item_score.put(i, score);
			}
			ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
			ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
			for (int i :trainlist)
				if (Pois.contains(i))
					tourpoilist.add(i);
			
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, Pois.size()-tourpoilist.size(), tourpoilist);
			//ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score, topK, tourpoilist);
			
			
			if(true) {
				double maps = 0;
				double mrrs = 0;
				double ndcgs = 0;
				double hrs = 0;
				double idcg = 0;
				double[] rank = new double [testList.size()];
				
				for (int i = 0;i<testList.size();i++) {
					int item = testList.get(i);
					rank[i] = -1;
					for (int j = 0;j<rankList.size();j++) {
						if (rankList.get(j)==item) {
							ndcgs += Math.log(2) / Math.log(j+2);
							rank[i] = j;
							break;
						}
					}			
				}
				Arrays.sort(rank);
				double a_topk = 0;
				double b_topk = 0;
				for(int i = 0;i<testList.size();i++) {
					if (rank[i] == -1) {
						System.out.printf("wrong sort list!\n");
						System.exit(0);
					}
					else {
						maps += (i+1)/(1+rank[i]);
					}
					idcg += Math.log(2) / Math.log(i+2);
					if (rank[i]<topK) {
						hrs++;
						a_topk = Math.log(2) / Math.log(rank[i]+2);
						b_topk = Math.log(2) / Math.log(i+2);
					}
						
				}			
				mrrs = 1/(rank[0]+1);
				ndcg[u]=ndcgs/idcg;
				map[u]=maps/testList.size();
				mrr[u]=mrrs;
				hr[u]=hrs;
				
				myout.write(Integer.toString(u));
				myout.write(':');
				for(int i = 0;i<testList.size();i++) {
					myout.write(Double.toString(rank[i]));
					myout.write('\t');
				}
				myout.write(Double.toString(hr[u]));
				myout.write('\t');
				myout.write(Double.toString(ndcg[u]));
				myout.write('\n');
				
				if (b_topk>0) {
					ndcg_topk[u] = a_topk/b_topk;
				}
			}
		}
		long end_eval = System.currentTimeMillis();
		double[] ans = new double [5];
		for(int u : Users) {
			ans[0] += mrr[u];
			ans[1] += hr[u];
			ans[2] += ndcg[u];
			ans[3] += map[u];
			ans[4] += ndcg_topk[u];
			}
		myout.write("this is the final answer:\n");
		myout.write("hr: "+ans[1]/testcount+ " ,ndcg: "+ans[2]/factnum+"\n");
		myout.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		//System.out.printf(" %d users don't have any test\n",userCount - factnum);	
	}

	public void orderoutput(String p,HashSet<Integer> Users,HashSet<Integer> POIs) {
		FileWriter out1 = null;
		try {
		out1 = new FileWriter(p);
		for (int u :Users)
			{
			int testitem =  testRatings.get(u).itemId;
			ArrayList<Integer> trainlist = trainMatrix.getRowRef(u).indexList();
			ArrayList<Integer> tourpoilist = new ArrayList<Integer>();
			for (int i :trainlist)
				if (POIs.contains(i))
					tourpoilist.add(i);
			

			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i:POIs) {
				double score = predict(u, i);
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score,itemCount-tourpoilist.size(), tourpoilist);
			out1.write(Integer.toString(u));
			out1.write("\t");
			out1.write(Integer.toString(rankList.indexOf(testitem)));
			out1.write("\n");
			}	
		out1.close();
		System.out.print("finish write file text order output\n");
		System.out.print(p);
		}
		catch (Exception e) {   

            e.printStackTrace();   

        }
	}
	
	public void extraevaluate(ArrayList<Rating> testRatings) {
		DenseVector map = new DenseVector(userCount);
		DenseVector mrr = new DenseVector(userCount);
		DenseVector ndcg = new DenseVector(userCount);
		DenseVector hr = new DenseVector(userCount);
		for (int u = 0;u<userCount;u++) {
			ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
			for (int i = 0; i < itemCount; i++) {
				double score = predict(u, i);
				map_item_score.put(i, score);
			}
			ArrayList<Integer> rankList = CommonUtils.TopKeysByValue(map_item_score,itemCount-itemList.size(), itemList);
			double maps = 0;
			int item = testRatings.get(u).itemId;
			ndcg.set(u, getNDCG( rankList, item));
			mrr.set(u, getPrecision( rankList, item));
			hr.set(u, getHitRatio( rankList, item));
			for (int j = 0;j<itemList.size();j++) {
				item = itemList.get(j);
				maps += getPrecision( rankList, item);
			}
			/*
			if (itemList.size()>0)
				map.set(u, maps/itemList.size());
			else
				map.set(u, 0);		*/	
		}
		System.out.printf("extra evaluate  all <ndcg,mrr,hr>:\t %.4f\t %.4f\\t %.4f\n",
				ndcg.mean(),mrr.mean(),hr.mean());
	}
	/**
	 * Evaluation for a specific user with given GT item.
	 * @return:
	 * 	 result[0]: hit ratio
	 * 	 result[1]: ndcg
	 * 	 result[2]: precision
	 */
	protected double[] evaluate_for_user(int u, int gtItem) {
		double[] result = new double[3];
		HashMap<Integer, Double> map_item_score = new HashMap<Integer, Double>();
		// Get the score of the test item first.
		double maxScore = predict(u, gtItem);
		
		// Early stopping if there are topK items larger than maxScore.
		int countLarger = 0;
		for (int i = 0; i < itemCount; i++) {
			double score = predict(u, i);
			map_item_score.put(i, score);
			
			if (score > maxScore)	countLarger ++;
			if (countLarger > topK)	return result;	// early stopping
		}
		
		// Selecting topK items (does not exclude train items).
		ArrayList<Integer> rankList = ignoreTrain ? 
				CommonUtils.TopKeysByValue(map_item_score, topK, trainMatrix.getRowRef(u).indexList()) : 
				CommonUtils.TopKeysByValue(map_item_score, topK, null);
		result[0] = getHitRatio(rankList, gtItem);
		result[1] = getNDCG(rankList, gtItem);
		result[2] = getPrecision(rankList, gtItem);
		
		return result;
	}
	
	/**
	 * Compute Hit Ratio.
	 * @param rankList  A list of ranked item IDs
	 * @param gtItem The ground truth item. 
	 * @return Hit ratio.
	 */
	public double getHitRatio(List<Integer> rankList, int gtItem) {
		for (int item : rankList) {
			if (item == gtItem)	return 1;
		}
		return 0;
	}
	
	/**
	 * Compute NDCG of a list of ranked items.
	 * See http://recsyswiki.com/wiki/Discounted_Cumulative_Gain
	 * @param rankList  a list of ranked item IDs
	 * @param gtItem The ground truth item. 
	 * @return  NDCG.
	 */
	public double getNDCG(List<Integer> rankList, int gtItem) {
		for (int i = 0; i < rankList.size(); i++) {
			int item = rankList.get(i);
			if (item == gtItem)
				return Math.log(2) / Math.log(i+2);
		}
		return 0;
	}
	
	public double getPrecision(List<Integer> rankList, int gtItem) {
		for (int i = 0; i < rankList.size(); i++) {
			int item = rankList.get(i);
			if (item == gtItem)
				return 1.0 / (i + 1);
		}
		return 0;
	}
	
	public double getpos(List<Integer> rankList, int gtItem) {
		for (int i = 0; i < rankList.size(); i++) {
			int item = rankList.get(i);
			if (item == gtItem)
				return i;
		}
		return 0;
	}
	
	// remove
	public void runOneIteration() {}
	
	// remove
	public double loss() {return 0;}
	
	// remove
	public void setUV(DenseMatrix U, DenseMatrix V) {};
}

// Thread for running the offline evaluation.
class EvaluationThread extends Thread {
	TopKRecommender model;
	ArrayList<Rating> testRatings;
	ArrayList<Integer> users;

	public EvaluationThread(TopKRecommender model, ArrayList<Rating> testRatings, 
			ArrayList<Integer> users) {
		this.model = model;
		this.testRatings = testRatings;
		this.users = users;
	}
	
	public void run() {
		for (int u : users) {
			double[] res = model.evaluate_for_user(u, testRatings.get(u).itemId);
			model.hits.set(u, res[0]);
			model.ndcgs.set(u, res[1]);
			model.precs.set(u, res[2]);
		}
	}
}