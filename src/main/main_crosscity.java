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


public abstract class main_crosscity {

	/** Rating matrix for training. */ 
	public static SparseMatrix trainMatrix;
	
	/** Test ratings (sorted by time for global split). */
	public static ArrayList<Rating> testRatings;
	
	public static boolean [] user_city;   // 1=targetcity 0=
	public static boolean [] poi_city;    //1=tourcity
	
	public static HashSet<Integer> nativeUsers = new HashSet<Integer>();
	public static HashSet<Integer> tourUsers = new HashSet<Integer>();
	public static HashSet<Integer> nativePois = new HashSet<Integer>();
	public static HashSet<Integer> tourPois = new HashSet<Integer>();
	
	public static int topK = 100;
	public static int threadNum = 6;
	
	public static int userCount;
	public static int itemCount;
	
	public static int strangerCount;
	public static int localCount;
	
	
	// imput (user,item,ratings)
	public static void ReadRatings_GlobalSplit_notimestamp(String ratingFile, double testRatio,int nativecity,int tourcity)
			throws IOException {
		if (nativecity == tourcity) {
			System.out.println("mismatch native city and tour city");
			System.exit(1);
		}
		
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
			if (rating.ucity == nativecity || rating.pcity == tourcity)
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
		user_city = new boolean[userCount];
		poi_city = new boolean[itemCount];
		System.out.printf("Generate trainMatrix and testStream.");
		startTime = System.currentTimeMillis();
		
		trainMatrix = new SparseMatrix(userCount, itemCount);
		testRatings = new ArrayList<Rating>();
		int[] user_num = new int [2];
		int[] poi_num = new int [2];
		int[] record_num = new int [3];
		ArrayList<Crossrating> newratings = new ArrayList<Crossrating>();
		for (Crossrating rating : ratings) {	
			if (rating.ucity == nativecity && rating.pcity == nativecity) {
				record_num[0] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == tourcity && rating.pcity == tourcity) {
				record_num[1] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == nativecity && rating.pcity == tourcity) {
				record_num[2] ++;
				newratings.add(rating);
			}
			else {
				System.out.println("mismatch native city and tour city");
				System.exit(1);
			}
			if (rating.ucity == nativecity)
				user_city[rating.uid] = true;
			else
				user_city[rating.uid] = false;
			if (rating.pcity == tourcity)
				poi_city[rating.pid] = true;
			else
				poi_city[rating.pid] = false;
		}
		ratings = newratings;
		for (int u =0;u<userCount;u++) {
			boolean i = user_city[u];
			if (i==true) {
				user_num[1] ++;
				nativeUsers.add(u);
			}
			else {
				user_num[0] ++;
				tourUsers.add(u);
			}
		}
		for (int u =0;u<itemCount;u++) {
			boolean i = poi_city[u];
			if (i==true) {
				poi_num[1] ++;
				tourPois.add(u);
			}
			else {
				poi_num[0] ++;
				nativePois.add(u);
			}
		}
		System.out.printf("statistics:\n");
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				nativecity,record_num[0],user_num[1],poi_num[0]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",tourcity,
				tourcity,record_num[1],user_num[0],poi_num[1]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				tourcity,record_num[2],user_num[1],poi_num[1]);
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
			else if (user_test[user] == 0 && nativeUsers.contains(user))
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
			if (trainMatrix.getRowRef(u).itemCount() == 0 && nativeUsers.contains(u))	newuser++;
		}
		for (int i=0;i<itemCount;i++) {
			if (trainMatrix.getColRef(i).itemCount() == 0 && tourPois.contains(i))	newpoi ++;
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

	
	
	
	public static void ReadRatings_GlobalSplit_notimestamp_forLCE(String ratingFile, double testRatio,int nativecity,int tourcity)
			throws IOException {
		if (nativecity == tourcity) {
			System.out.println("mismatch native city and tour city");
			System.exit(1);
		}
		
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
			//if (rating.ucity == nativecity || rating.pcity == tourcity)
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
		user_city = new boolean[userCount];
		poi_city = new boolean[itemCount];
		System.out.printf("Generate trainMatrix and testStream.");
		startTime = System.currentTimeMillis();
		
		trainMatrix = new SparseMatrix(userCount, itemCount);
		testRatings = new ArrayList<Rating>();
		int[] user_num = new int [2];
		int[] poi_num = new int [2];
		int[] record_num = new int [4];
		ArrayList<Crossrating> newratings = new ArrayList<Crossrating>();
		for (Crossrating rating : ratings) {	
			if (rating.ucity == nativecity && rating.pcity == nativecity) {
				record_num[0] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == tourcity && rating.pcity == tourcity) {
				record_num[1] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == nativecity && rating.pcity == tourcity) {
				record_num[2] ++;
				newratings.add(rating);
				
			}
			else if (rating.ucity == tourcity && rating.pcity == nativecity){
				record_num[3] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else {
				System.out.println("mismatch native city and tour city");
				System.exit(1);
			}
			if (rating.ucity == nativecity)
				user_city[rating.uid] = true;
			else
				user_city[rating.uid] = false;
			if (rating.pcity == tourcity)
				poi_city[rating.pid] = true;
			else
				poi_city[rating.pid] = false;
		}
		ratings = newratings;
		for (int u =0;u<userCount;u++) {
			boolean i = user_city[u];
			if (i==true) {
				user_num[1] ++;
				nativeUsers.add(u);
			}
			else {
				user_num[0] ++;
				tourUsers.add(u);
			}
		}
		for (int u =0;u<itemCount;u++) {
			boolean i = poi_city[u];
			if (i==true) {
				poi_num[1] ++;
				tourPois.add(u);
			}
			else {
				poi_num[0] ++;
				nativePois.add(u);
			}
		}
		System.out.printf("statistics:\n");
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				nativecity,record_num[0],user_num[1],poi_num[0]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",tourcity,
				tourcity,record_num[1],user_num[0],poi_num[1]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				tourcity,record_num[2],user_num[1],poi_num[1]);
		System.out.printf("ucity:%d pcity:%d records:%d\n",tourcity, nativecity, record_num[3]);
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
			else if (user_test[user] == 0 && nativeUsers.contains(user))
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
			if (trainMatrix.getRowRef(u).itemCount() == 0 && nativeUsers.contains(u))	newuser++;
		}
		for (int i=0;i<itemCount;i++) {
			if (trainMatrix.getColRef(i).itemCount() == 0 && tourPois.contains(i))	newpoi ++;
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

	public static void ReadRatings_GlobalSplit_notimestamp_four(String ratingFile, double testRatio,int nativecity,int tourcity)
			throws IOException {
		if (nativecity == tourcity) {
			System.out.println("mismatch native city and tour city");
			System.exit(1);
		}
		
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
			//if (rating.ucity == nativecity || rating.pcity == tourcity)
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
		user_city = new boolean[userCount];
		poi_city = new boolean[itemCount];
		System.out.printf("Generate trainMatrix and testStream.");
		startTime = System.currentTimeMillis();
		
		trainMatrix = new SparseMatrix(userCount, itemCount);
		testRatings = new ArrayList<Rating>();
		int[] user_num = new int [2];
		int[] poi_num = new int [2];
		int[] record_num = new int [3];
		int fourrating = 0;
		ArrayList<Crossrating> newratings = new ArrayList<Crossrating>();
		for (Crossrating rating : ratings) {	
			if (rating.ucity == nativecity && rating.pcity == nativecity) {
				record_num[0] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == tourcity && rating.pcity == tourcity) {
				record_num[1] ++;
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				}
			else if (rating.ucity == nativecity && rating.pcity == tourcity) {
				record_num[2] ++;
				newratings.add(rating);
			}
			else {
				trainMatrix.setValue(rating.uid, rating.pid, rating.count);
				fourrating ++;
				//System.out.println("mismatch native city and tour city");
				//System.exit(1);
			}
			if (rating.ucity == nativecity)
				user_city[rating.uid] = true;
			else
				user_city[rating.uid] = false;
			if (rating.pcity == tourcity)
				poi_city[rating.pid] = true;
			else
				poi_city[rating.pid] = false;
		}
		
		ratings = newratings;
		for (int u =0;u<userCount;u++) {
			boolean i = user_city[u];
			if (i==true) {
				user_num[1] ++;
				nativeUsers.add(u);
			}
			else {
				user_num[0] ++;
				tourUsers.add(u);
			}
		}
		for (int u =0;u<itemCount;u++) {
			boolean i = poi_city[u];
			if (i==true) {
				poi_num[1] ++;
				tourPois.add(u);
			}
			else {
				poi_num[0] ++;
				nativePois.add(u);
			}
		}
		System.out.printf("statistics:\n");
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				nativecity,record_num[0],user_num[1],poi_num[0]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",tourcity,
				tourcity,record_num[1],user_num[0],poi_num[1]);
		System.out.printf("ucity:%d pcity:%d records:%d usernum:%d poinum:%d\n",nativecity,
				tourcity,record_num[2],user_num[1],poi_num[1]);
		System.out.printf("ucity:%d pcity:%d records:%d\n",tourcity,
				nativecity,fourrating);
		
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
			else if (user_test[user] == 0)
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
			if (trainMatrix.getRowRef(u).itemCount() == 0 && nativeUsers.contains(u))	newuser++;
		}
		for (int i=0;i<itemCount;i++) {
			if (trainMatrix.getColRef(i).itemCount() == 0 && tourPois.contains(i))	newpoi ++;
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


