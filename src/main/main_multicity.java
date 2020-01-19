package main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.NavigableMap;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

import algorithms.*;
import utils.DatasetUtil;
import data_structure.*;
import utils.Printer;
import utils.CommonUtils;

import java.util.ArrayList;


public abstract class main_multicity {

	/** Rating matrix for training. */ 
	public static SparseMatrix trainMatrix;
	
	/** Test ratings (sorted by time for global split). */
	public static ArrayList<Rating> testRatings;
	
	public static int [] user_city;   //0
	public static int [] poi_city;    //1
	public static int [] city_code_ori = {300,400,500,600}; 
	public static int [] city_code_re = {0,1,2,3}; 
	
	public static HashSet<Integer> nativeUsers = new HashSet<Integer>();
	public static HashSet<Integer> tourUsers = new HashSet<Integer>();
	public static HashSet<Integer> nativePois = new HashSet<Integer>();
	public static HashSet<Integer> tourPois = new HashSet<Integer>();
	
	public static HashMap<Integer, Integer> City_users = new HashMap<Integer, Integer>();
	public static HashMap<Integer, Integer> City_pois = new HashMap<Integer, Integer>();
	
	public static int [] [] city_users;
	public static int [] city_users_len;
	public static int [] [] city_pois;
	public static int [] city_pois_len;
	
	public static int topK = 100;
	public static int threadNum = 6;
	
	public static int userCount;
	public static int itemCount;
	
	public static int strangerCount;
	public static int localCount;
	// imput (user,item,ratings)
	public static void ReadRatings_GlobalSplit_multicity(String ratingFile, double testRatio,int citynum)
			throws IOException {

		userCount = itemCount = 0;
		System.out.println("Global splitting with testRatio " + testRatio);
		// Step 1. Construct data structure for sorting.
		System.out.print("Read ratings and sort.");
		long startTime = System.currentTimeMillis();
		ArrayList<Crossrating> ratings = new ArrayList<Crossrating>();
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(ratingFile)));
		String line;
		while((line = reader.readLine()) != null) {
			Crossrating rating = new Crossrating(line);
			if (true)
				ratings.add(rating);	
			userCount = Math.max(userCount, rating.uid);
			itemCount = Math.max(itemCount, rating.pid);
		}
		reader.close();
		userCount ++;
		itemCount ++;
		System.out.printf("[%s]\n", Printer.printTime(
				System.currentTimeMillis() - startTime));
		
		// Step 3. Generate trainMatrix and testStream
		user_city = new int[userCount];
		poi_city = new int[itemCount];
		System.out.printf("Generate trainMatrix and testStream.");
		startTime = System.currentTimeMillis();
		
		trainMatrix = new SparseMatrix(userCount, itemCount);
		testRatings = new ArrayList<Rating>();
		
		city_users = new int [citynum][];   // 0 is the len
		city_users_len = new int [citynum];
		city_pois = new int [citynum][];
		city_pois_len = new int [citynum];	
		
		
		int[] user_num = new int [2];
		int[] poi_num = new int [2];
		int[] record_num = new int [3];
		ArrayList<Crossrating> newratings = new ArrayList<Crossrating>();
		for (Crossrating rating : ratings) {	
			if (rating.ucity == rating.pcity) {
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				record_num[0] ++ ;
				}
			else {
				newratings.add(rating);
				record_num[1]++;
				}
			user_city[rating.uid] = rating.ucity;
			poi_city[rating.pid] = rating.pcity;
		}
		ratings = newratings;
		for (int i = 0;i<citynum;i++) {
			city_users_len[i] = 0;
			city_pois_len[i] = 0;
		}
		for (int u =0;u<userCount;u++) {
			int re = (user_city[u]/100)-3;
			City_users.put(u, re);
			city_users_len[re] ++;
		}
		for (int u =0;u<itemCount;u++) {
			int re = (poi_city[u]/100)-3;
			City_pois.put(u, re);
			city_pois_len[re] ++;
		}
		for (int i = 0;i<citynum;i++) {
			city_users[i] = new int [city_users_len[i]+1];
			city_users[i][0] = city_users_len[i];
			city_pois[i] = new int [city_pois_len[i]+1];
			city_pois[i][0] = city_pois_len[i];
			city_users_len[i] = 0;
			city_pois_len[i] = 0;
		}
		for (int u =0;u<userCount;u++) {
			int re = (user_city[u]/100)-3;
			user_city[u] = re;
			city_users_len[re] ++ ;
			city_users[re][city_users_len[re]] = u;
			
		}
		for (int u =0;u<itemCount;u++) {
			int re = (poi_city[u]/100)-3;
			poi_city[u] = re;
			city_pois_len[re] ++ ;
			city_pois[re][city_pois_len[re]] = u;
			
		}
		// check 
		boolean flag_city = true;
		for (int i = 0;i<citynum;i++) {
			if (city_users[i][0] != city_users_len[i] )
				flag_city = false;
			if (city_pois[i][0] != city_pois_len[i] )
				flag_city = false;
		}
		if (flag_city == false) {
			System.out.printf("city flag is false!! \ncheck main for more info!\n ");
			System.exit(0);
			}
		else {
			System.out.printf("city flag is true and algorithm is continuing ... \n");
		}
		
		System.out.printf("statistics:\n");
		System.out.printf("native records:%d, tour records:%d\n",record_num[0], record_num[1]);
		
		strangerCount = record_num[2];
		localCount = record_num[1];
		
		int[] user_number = new int [userCount];
		int[] user_test = new int[userCount];		
		for (Crossrating rating : ratings) {
			user_number[rating.uid] ++;
		}
		int singleuser = 0;

		for (int user =0;user<userCount;user++) {
			user_test[user] = (int)Math.round((float)user_number[user]*testRatio);
			if (user_test[user] == 0 && user_number[user] > 1)
				user_test[user] = 1;
			else if (user_test[user] == 0 )
				singleuser ++;					
			if (user_number[user] == 0) {
			}
		}
		System.out.printf("%d user has just a record!\n",singleuser);
			
		for (int i = ratings.size()-1; i>=0;i--) {
			Crossrating rating = ratings.get(i);
			if (user_test[rating.uid] > 0) {
				user_test[rating.uid] --;
				testRatings.add(new Rating(rating.uid,rating.pid,(float)rating.count,(long)1));
			}
			else
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);				
		}
		
		// Count number of new users/items/ratings in the test data

		int newuser = 0;
		int newpoi = 0;
		for (int u = 0; u < userCount; u ++) {
			if (trainMatrix.getRowRef(u).itemCount() == 0 )	
				newuser++;
		}
		for (int i=0;i<itemCount;i++) {
			if (trainMatrix.getColRef(i).itemCount() == 0 )	
				newpoi ++;
		}
		
		System.out.printf("[%s]\n", Printer.printTime(
				System.currentTimeMillis() - startTime));
		
		// Print some basic statistics of the dataset.
		System.out.println ("Data\t" + ratingFile);
		System.out.println ("#Users\t" + userCount + ", #newUser: " + newuser);
		System.out.println ("#Items\t" + itemCount + ", #newpoi: " + newpoi);
		System.out.printf("#Ratings\t %d (train), %d(test)\n", 
				trainMatrix.itemCount(),  testRatings.size());
	}


	// Evaluate the model
	public static double[] evaluate_model(TopKRecommender model, String name) {
		long start = System.currentTimeMillis();
		model.buildModel();
		model.evaluate(testRatings);
		
		double[] res = new double[3];
		res[0] = model.hits.mean();
		res[1] = model.ndcgs.mean();
		res[2] = model.precs.mean();
		System.out.printf("%s\t <hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f [%s]\n", 
				name, res[0], res[1], res[2],
				Printer.printTime(System.currentTimeMillis() - start));
		return res;
	}
	

}


