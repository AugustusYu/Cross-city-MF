package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import utils.Printer;

/**
 * ALS algorithm of the ICDM'09 paper:
 * Yifan Hu etc. Collaborative Filtering for Implicit Feedback Datasets. 
 * @author xiangnanhe
 */
public class MF_ALS_quickiter_multi extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double w0 = 0.01;	// weight for 0s
	double reg = 0.01; 	// regularization parameters
  double init_mean = 0;  // Gaussian mean for init V
  double init_stdev = 0.01; // Gaussian std-dev for init V
	int update_mode = 5;
	int showbound = 0;
	int showcount = 10;
  /** Model parameters to learn */
  DenseMatrix U;	// latent vectors for users
  DenseMatrix V;	// latent vectors for items
  Double []rui ;
  Double []rui_u;
  /** Caches */
	DenseMatrix SU;
	DenseMatrix SV;
	
	boolean showProgress;
	boolean showLoss;
	int citynum = 2;
	
	public Integer [] [] buy_records;
	public int [] [] city_users;
	public int [] city_users_len;
	public int [] [] city_pois;
	public int [] city_pois_len;
	public int [] user_city;   //0
	public int [] poi_city;    //1
	
	public MF_ALS_quickiter_multi(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
			int topK, int threadNum, int factors, int maxIter, double w0, double reg, 
			double init_mean, double init_stdev, boolean showProgress, boolean showLoss,int showbound,int showcount
			, int citynum) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0 ;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showLoss = showLoss;
		this.initialize();
		this.showbound = showbound;
		this.showcount = showcount;
		this.citynum = citynum;
	}
	
	public void setintarray(int [] [] A, int [][]B, int []C, int[]D ) {
		//System.out.printf("check %d %d %d %d",A[0][0],B[0][0],C[0],D[0]);
		city_users = A.clone();
		city_pois = B.clone();
		user_city = C.clone();
		poi_city = D.clone();
		city_users_len = new int [citynum];
		city_pois_len = new int [citynum];
		for (int i = 0;i<citynum;i++) {
			city_users_len[i] = city_users[i][0];
			city_pois_len[i] = city_pois[i][0];
		}
	}
	
	//remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		SU = U.transpose().mult(U);
		SV = V.transpose().mult(V);
	}
	
	private void initialize() {
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		
		SU = U.transpose().mult(U);
		SV = V.transpose().mult(V);
		rui = new Double [itemCount];
		rui_u = new Double [userCount];
		
	}
	
	// Implement the ALS algorithm of the ICDM'09 paper
	public void buildModel() {
		System.out.println("Run for MF_ALS_quickiter");
		
		double loss_pre = Double.MAX_VALUE;
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			
			// Update user factors
			for (int u = 0; u < userCount; u ++) {
				update_user(u,update_mode);
			}
			
			// Update item factors
			for (int i = 0; i < itemCount; i ++) {
				update_item(i,update_mode);
			}		
			// Show progress
			if (showProgress && (iter > showbound || iter % showcount == 0)) {
				long end_iter = System.currentTimeMillis();
				System.out.printf("iter = %d [%s]  ",iter,Printer.printTime(end_iter - start));
				evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
			}
			//showProgress(iter, start, testRatings);			
			// Show loss
			if (showLoss)
				loss_pre = showLoss(iter, start, loss_pre);	
		}
	}
	
	public void buildmulticityModel() {
		System.out.println("Run for MF_ALS_quickiter");
		
		double loss_pre = Double.MAX_VALUE;
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			
			// Update user factors
			for (int u = 0; u < userCount; u ++) {
				update_user_multi(u);
			}
			
			// Update item factors
			for (int i = 0; i < itemCount; i ++) {
				update_item_multi(i);
			}		
			// Show progress
			if (showProgress && (iter > showbound || iter % showcount == 0)) {
				long end_iter = System.currentTimeMillis();
				System.out.printf("iter = %d [%s]  ",iter,Printer.printTime(end_iter - start));
				evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
			}
			//showProgress(iter, start, testRatings);			
			// Show loss
			if (showLoss)
				loss_pre = showLoss(iter, start, loss_pre);	
		}
	}
	
	
	public void buildcrosscityModel_runtime_test(double r) {
		double avertime = 0;
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			// Update user factors
			for (int u = 0; u < userCount*r; u ++) {
				update_user(u,update_mode);
			}
			
			// Update item factors
			for (int i = 0; i < itemCount*r; i ++) {
				update_item(i,update_mode);
			}		
			if (true) {
				long end_iter = System.currentTimeMillis();
				System.out.printf("iter = %d [%s]  \n",iter,Printer.printTime(end_iter - start));			
				//evaluatefor82crosscity(testRatings,start,targetUsers,targetPois);
				long thistime = end_iter - start;
				double this_second = (double)(thistime)/1000;
				avertime = (avertime * iter + this_second)/(iter+1);
				System.out.printf("iter = %d,this time:%f, average time:%f  \n",iter,this_second,avertime);	
			}
		}
	}
		
	// Run model for one iteration
	public void runOneIteration() {
		// Update user latent vectors
		for (int u = 0; u < userCount; u ++) {
			update_user(u);
		}
		
		// Update item latent vectors
		for (int i = 0; i < itemCount; i ++) {
			update_item(i);
		}
	}
	
	private void update_user(int u) {
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		// Get matrix Au
		DenseMatrix Au = SU.scale(w0);
		for (int k1 = 0; k1 < factors; k1 ++) {
			for (int k2 = 0; k2 < factors; k2 ++) {
				for (int i : itemList)
					Au.add(k1, k2, V.get(i, k1) * V.get(i, k2) * (1 - w0));
			}
		} 
		// Get vector du
		DenseVector du = new DenseVector(factors);
		for (int k = 0; k < factors; k ++) {
			for (int i : itemList)
				//du.add(k, V.get(i, k) * trainMatrix.getValue(u, i));
				du.add(k, V.get(i, k) * 1);
		}
		// Matrix inversion to get the new embedding
		for (int k = 0; k < factors; k ++) { // consider the regularizer
			Au.add(k, k, reg);
		}
		DenseVector newVector = Au.inv().mult(du);
		
		// Update the SU cache
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SU.get(f, k) - U.get(u, f) * U.get(u, k)
						+ newVector.get(f) * newVector.get(k);
				SU.set(f, k, val);
				SU.set(k, f, val);
			}
		}
		// Update parameters
		for (int k = 0; k < factors; k ++) {
			U.set(u, k, newVector.get(k));
		}
	}
	
	private void update_item(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		// Get matrix Ai
		DenseMatrix Ai = SV.scale(w0);
		for (int k1 = 0; k1 < factors; k1 ++) {
			for (int k2 = 0; k2 < factors; k2 ++) {
				for (int u : userList)
					Ai.add(k1, k2, U.get(u, k1) * U.get(u, k2) * (1 - w0));
			}
		}
		// Get vector di
		DenseVector di = new DenseVector(factors);
		for (int k = 0; k < factors; k ++) {
			for (int u : userList)
				//di.add(k, U.get(u, k) * trainMatrix.getValue(u, i));
				di.add(k, U.get(u, k) * 1);
		}
		// Matrix inversion to get the new embedding
		for (int k = 0; k < factors; k ++) { // consider the regularizer
			Ai.add(k, k, reg);
		}
		DenseVector newVector = Ai.inv().mult(di);
		
		// Update the SV cache
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SV.get(f, k) - V.get(i, f) * V.get(i, k)
						+ newVector.get(f) * newVector.get(k);
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
		
		// Update parameters
		for (int k = 0; k < factors; k ++) {
			V.set(i, k, newVector.get(k));
		}
	}
	private void update_user_multi(int u) {
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		DenseVector oldVector = U.row(u);		
		for (int i:itemList)
			rui[i] = U.row(u, false).inner(V.row(i, false));
		int acount = 0;
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = w0*SV.get(k,k)+reg;
			for (int f = 0;f<factors;f++) {
				numerator += -w0*SV.get(k, f)*U.get(u, f);
			}
			numerator -= -w0*SV.get(k, k)*U.get(u, k);
			for(int i:itemList) {
				denominator += (1-w0)* V.get(i,k)*V.get(i, k);
				numerator += V.get(i, k)-(1-w0)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
			}
			double val = 0;
			double a = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val)) {
				System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
				val = 1;
				acount +=1;
			}
			for (int i:itemList) {
				rui[i] += (val-U.get(u, k))*V.get(i, k);
			}		
			U.set(u, k, val);	
		}
		if (acount > 3) System.exit(0);
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k)
						+ U.get(u, f) * U.get(u, k);
				SU.set(f, k, val);
				SU.set(k, f, val);
			}
		} // end for f	
	}
	
	
	private void update_user(int u,int mode ) {
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		
		switch(mode) {
		case 0:{
			System.exit(1);
		}break;
		
		case 1:{
			System.exit(1);
		}break;
			
		case 2:{
			System.exit(1);
		}break;
		
		//element wise
		case 3 :{
			System.exit(1);
		}break;
			
		//element wise 
		case 4 :{
			Double []rui = new Double [itemCount];
			for (int i =0;i<itemCount;i++)
				rui[i] = U.row(u, false).inner(V.row(i, false));
			int acount = 0;
			for(int k =0;k<factors;k++) {
				double numerator = 0;
				double denominator = reg;
				for (int i =0;i<itemCount;i++) {
					if (trainMatrix.getValue(u, i)!=0) {
						numerator += V.get(i, k)*(1-rui[i]+U.get(u, k)*V.get(i, k));
						denominator += V.get(i,k)*V.get(i, k);
					}
					else {
						numerator += V.get(i, k)*(0-rui[i]+U.get(u, k)*V.get(i, k))*w0;
						denominator += V.get(i,k)*V.get(i, k)*w0;
					}
				}
				double val = 0;
				if (denominator!= 0)
					val = numerator/denominator;
				if (Double.isNaN(val)) {
					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
					val = 1;
					acount +=1;
				}
				for (int i =0;i<itemCount;i++) {
					rui[i] += (val-U.get(u, k))*V.get(i, k);
				}		
				U.set(u, k, val);	
			}
			if (acount > 3) System.exit(0);
		}break;
		//element wise with cache poi_cache
		case 5 :{
			DenseVector oldVector = U.row(u);		
			for (int i:itemList)
				rui[i] = U.row(u, false).inner(V.row(i, false));
			int acount = 0;
			for(int k =0;k<factors;k++) {
				double numerator = 0;
				double denominator = 0;
				denominator = w0*SV.get(k,k)+reg;
				for (int f = 0;f<factors;f++) {
					numerator += -w0*SV.get(k, f)*U.get(u, f);
				}
				numerator -= -w0*SV.get(k, k)*U.get(u, k);
				for(int i:itemList) {
					denominator += (1-w0)* V.get(i,k)*V.get(i, k);
					numerator += V.get(i, k)-(1-w0)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
				}
				double val = 0;
				double a = 0;
				if (denominator!= 0)
					val = numerator/denominator;
				if (Double.isNaN(val)) {
					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
					val = 1;
					acount +=1;
				}
				for (int i:itemList) {
					rui[i] += (val-U.get(u, k))*V.get(i, k);
				}		
				U.set(u, k, val);	
			}
			if (acount > 3) System.exit(0);
			for (int f = 0; f < factors; f ++) {
				for (int k = 0; k <= f; k ++) {
					double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k)
							+ U.get(u, f) * U.get(u, k);
					SU.set(f, k, val);
					SU.set(k, f, val);
				}
			} // end for f	
		}break;
		}	
	}
	
	private void update_item_multi(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		DenseVector oldVector = V.row(i);
		double []rui_u = new double [userCount];
		for (int u :userList)
			rui_u[u] = U.row(u, false).inner(V.row(i, false));
		
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = SU.get(k, k)*w0+reg;
			for (int f = 0;f<factors;f++) {
				numerator += -w0*SU.get(k, f)*V.get(i, f);
			}
			numerator -= -w0*SU.get(k, k)*V.get(i, k);					
			for(int u :userList) {
				denominator += (1-w0)*U.get(u, k)*U.get(u, k);
				numerator += U.get(u, k)-(1-w0)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
			}
			double val = 0;
			double a =0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				rui_u[u] += (val - V.get(i, k))*U.get(u,k);
			}	
			V.set(i, k, val);				
		}
		// Update the SV cache
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k)
						+ V.get(i, f) * V.get(i, k);
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}	
	}
	
	private void update_item(int i,int mode ) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		switch(mode) {
		case 0:{
			System.exit(1);
		}break;
		
		case 1:{
			System.exit(1);
		}break;
			
		case 2:{
			System.exit(1);
		}break;
		
		//element wise
		case 3 :{
			System.exit(1);
		}break;
		
		//element wise
		case 4 :{
			double []rui = new double [userCount];
			for (int u =0;u<userCount;u++)
				rui[u] = U.row(u, false).inner(V.row(i, false));
			
			for(int k =0;k<factors;k++) {
				double numerator = 0;
				double denominator = reg;
				for (int u =0;u<userCount;u++) {
					if (trainMatrix.getValue(u, i)!=0) {
						numerator += U.get(u, k)*(1-rui[u]+V.get(i, k)*U.get(u, k));
						denominator += U.get(u,k)*U.get(u, k);
					}
					else {
						numerator += U.get(u, k)*(0-rui[u]+U.get(u, k)*V.get(i, k))*w0;
						denominator += U.get(u,k)*U.get(u, k)*w0;
					}
				}
				double val = 0;
				if (denominator!= 0)
					val = numerator/denominator;
				if (Double.isNaN(val))
					val = 1;
				for (int u =0;u<userCount;u++) {
					rui[u] += (val - V.get(i, k))*U.get(u,k);
				}
				V.set(i, k, val);
				
			}
		}break;
		
		//element wise with cahce
		case 5 :{
			DenseVector oldVector = V.row(i);
			double []rui_u = new double [userCount];
			for (int u :userList)
				rui_u[u] = U.row(u, false).inner(V.row(i, false));
			
			for(int k =0;k<factors;k++) {
				double numerator = 0;
				double denominator = 0;
				denominator = SU.get(k, k)*w0+reg;
				for (int f = 0;f<factors;f++) {
					numerator += -w0*SU.get(k, f)*V.get(i, f);
				}
				numerator -= -w0*SU.get(k, k)*V.get(i, k);					
				for(int u :userList) {
					denominator += (1-w0)*U.get(u, k)*U.get(u, k);
					numerator += U.get(u, k)-(1-w0)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
				}
				double val = 0;
				double a =0;
				if (denominator!= 0)
					val = numerator/denominator;
				if (Double.isNaN(val))
					val = 1;
				for (int u :userList) {
					rui_u[u] += (val - V.get(i, k))*U.get(u,k);
				}	
				V.set(i, k, val);				
			}
			// Update the SV cache
			for (int f = 0; f < factors; f ++) {
				for (int k = 0; k <= f; k ++) {
					double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k)
							+ V.get(i, f) * V.get(i, k);
					SV.set(f, k, val);
					SV.set(k, f, val);
				}
			}		
		}break;
		}
	}
	
	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		double loss_cur = loss();
		String symbol = loss_pre >= loss_cur ? "-" : "+";
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s]\n", iter, 
				Printer.printTime(start1 - start), symbol, loss_cur, 
				Printer.printTime(System.currentTimeMillis() - start1));
		return loss_cur;
	}
	
	// Fast way to calculate the loss function
	public double loss() {
		// Init the SV cache for fast calculation
		DenseMatrix SV = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = 0;
				for (int i = 0; i < itemCount; i ++)
					val += V.get(i, f) * V.get(i, k);
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
		
		double L = reg * (U.squaredSum() + V.squaredSum());
		for (int u = 0; u < userCount; u ++) {
			double l = 0;
			for (int i : trainMatrix.getRowRef(u).indexList()) {
				//l += Math.pow(trainMatrix.getValue(u, i) - predict(u, i), 2);
				l += Math.pow(1 - predict(u, i), 2);
			}
			l *= (1 - w0);
			l += w0 * SV.mult(U.row(u, false)).inner(U.row(u, false));
			L += l;
		}
		return L;
	}
	
	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}

	@Override
	public void updateModel(int u, int i) {
		trainMatrix.setValue(u, i, 1);
		
		for (int iter = 0; iter < maxIterOnline; iter ++) {
			update_user(u);
			
			update_item(i);
		}
	}
}
