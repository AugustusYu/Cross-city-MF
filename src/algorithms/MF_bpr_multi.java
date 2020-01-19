package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import utils.CommonUtils;
import utils.Printer;


public class MF_bpr_multi extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double w0 = 0.01;	// weight for 0s
	double w0_c = 0;
	double lr = 0.01;
	double reg = 0.01; 	
	double init_mean = 0; 
	double init_stdev = 0.01;
	int update_mode = 5;
	int showbound = 0;
	int showcount = 10;
	/** Model parameters to learn */
	DenseMatrix U;	// latent vectors for users
	DenseMatrix V;	// latent vectors for items
	//DenseMatrix Ufinal;
	//DenseMatrix Vfinal;
	double []rui ;
	double []rui_u;   
	int citynum = 2;
	/** Caches */
//	double [][] SUtarget;
//	double [][] SVtarget;
//	double [][] SUextra;
//	double [][] SVextra;
//	double [][] SUfinal;
//	double [][] SVfinal;
	int sharefactor = 0;
	int [] user_index;    // user_index[u] =  i --- Ufinal[i] = U[u]
	int [] poi_index;
	int [] user_value;  // user_value[i] = u
	int [] poi_value;
	double bigalpha = 0.5; // for user reg
	double bigbeta = 0.5;  // for poi reg 
	Random rand = new Random();
	boolean loss_flag = false;
	
	boolean showProgress;
	boolean showLoss;
	
//	public Integer [][] buy_stranger_target;
//	public Integer [][] buy_stranger_source;
//	public Integer [][] buy_local_target;

	public Integer [] [] buy_records;
	public Integer [] [] buy_records_home;
	public Integer [] [] buy_records_tour;
	public HashMap<Integer,Integer> stranger_index = new HashMap<Integer,Integer>();
	public HashMap<Integer,Integer> local_index = new HashMap<Integer,Integer>();
	
	//public  HashSet<Integer> targetUsers ;
	//public  HashSet<Integer> targetPois ;
	//public  HashSet<Integer> extraUsers = new HashSet<Integer>();
	//public  HashSet<Integer> extraPois = new HashSet<Integer>();
	
	public int [] [] city_users;
	public int [] city_users_len;
	public int [] [] city_pois;
	public int [] city_pois_len;
	public int [] user_city;   //0
	public int [] poi_city;    //1
	
	public MF_bpr_multi(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
			int topK, int threadNum, int factors, int maxIter, double lr, double reg, 
			double init_mean, double init_stdev, boolean showProgress, boolean showLoss,int showbound,int showcount,
			int citynum) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr ;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showLoss = showLoss;
		this.citynum = citynum;
		//this.setextrainfo(new int [3]); 
		//this is not finished
		this.showbound = showbound;
		this.showcount = showcount;
	}
	
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		//SU = U.transpose().mult(U);
		//SV = V.transpose().mult(V);
	}
	
//	public void sethashset(HashSet<Integer> A,HashSet<Integer> B) {
//		this.targetUsers = A;
//		this.targetPois = B;	
//	}
	
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
	
	public void setintpara(int [] paras) {
		this.sharefactor = paras[0];
	}
	
	public void setbigw(double [] arr) {
		this.bigalpha = arr[0];
		this.bigbeta = arr[1];
		System.out.printf("bigalpha:%f,bigbeta:%f\n",bigalpha,bigbeta);
	}
	
	public void initialize() {
		rui = new double [itemCount];
		rui_u = new double [userCount];	
		//this.w0_c = this.w0 * targetPois.size()/(itemCount - targetPois.size());
		this.w0_c = this.w0;
		System.out.printf("w0_c = %f \n",w0_c);
		//System.out.printf("target user:%d target poi:%d \n",targetUsers.size(),targetPois.size());
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		//Ufinal = new DenseMatrix(targetUsers.size(), factors);
		//Vfinal = new DenseMatrix(targetPois.size(), factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);	
		
		user_index = new int [userCount];
		poi_index = new int [itemCount];
	
	}
	

	public void buildModel() {
		//no longer to be used
	}
	
	
	public void buildmulticityModel(int mode) {
		mode = 3;
		System.out.printf("multicity model has been built\n");
		double loss_pre = Double.MAX_VALUE;
		int nonzeros = trainMatrix.itemCount();
		buy_records = new Integer[userCount][];
		
		for (int u = 0 ;u<userCount;u++) {
			if (true) {		
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				buy_records[u] = itemList.toArray(new Integer [itemList.size()]) ;
				}

		}
		switch (mode) {
		case 0:{
			System.out.printf("mode == 0, this algo only use sparse datasets!\n");
			
		}break;
		
		case 1:{
			System.out.printf("mode == 1, this algo adds users extra records!\n");						
		}break;
		
		case 2:{
			System.out.printf("mode == 2, this algo adds POIs extra records!\n");
		}break;	
		case 3:{
			System.out.printf("mode == 3, this algo adds all extra data!\n");
			for (int iter = 0; iter < maxIter; iter ++) {
				Long start = System.currentTimeMillis();
				rand = new Random();
				for (int s = 0; s < nonzeros; s ++) {					
					bpr_update_alluser_allpoi(s);
				}
				if (showProgress && (iter > showbound || iter % showcount == 0)) {
					long end_iter = System.currentTimeMillis();
					System.out.printf("iter = %d [%s]  ",iter,Printer.printTime(end_iter - start));			
					evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
				}			
			}
		}break;
		
		default:{
			System.out.printf("wrong mode! please check imput paras!\n");
			System.exit(0);
		}break;
		
		
		}
			
	}
	
	// Run model for one iteration
	public void runOneIteration() {
		// Update user latent vectors
		for (int u = 0; u < userCount; u ++) {
			
		}
		
		// Update item latent vectors
		for (int i = 0; i < itemCount; i ++) {
			
		}
	}
	
	private void bpr_update_alluser_allpoi(int s) {
		rand = new Random();
		int u = rand.nextInt(userCount);
		int num = 0;
		int i = 0;
		int j = 0;
		num = buy_records[u].length;
		if (num == 0) 
			return ;
		i = buy_records[u][rand.nextInt(num)];
		j = rand.nextInt(itemCount);
		while(trainMatrix.getValue(u, j)!=0)		
			j =  rand.nextInt(itemCount);
		
		//update
		double y_pos = predict(u,i);
		double y_neg = predict(u,j);
		double mult = - partial_loss(y_pos - y_neg);
		double grad_u =0;
		double grad = 0;
		for (int f = 0; f < factors; f ++) {
	    	grad_u = V.get(i, f) - V.get(j, f);
	    	U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));	
	    	grad = U.get(u, f);
	    	V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
	    	V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));      
	    }
		if(Double.isInfinite(grad)||Double.isInfinite(grad_u)) {
			System.out.print("INfinite num has been catched \n\n\n");
			System.exit(0);
		} 
	}
	

	  private double partial_loss(double x) {
		    double exp_x = Math.exp(-x);
		    return exp_x / (1 + exp_x);
	  }
	
	private int [] sample_update_pair(HashSet<Integer> users,HashSet<Integer> pois) {
		int u = 0;
		int i = 0;
		int j = 0;
		rand = new Random();
		u = rand.nextInt(userCount);
		while (!users.contains(u))
			u = rand.nextInt(userCount);	
		return new int [] {u,i,j};
	}
	

	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		int a = 0;
		double [] loss_cur = new double [2];
		loss_cur = loss(a);
		String symbol = loss_pre >= loss_cur[0] ? "-" : "+";
		if (symbol == "+")
			loss_flag = true;
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s], small loss: %.4f\n", iter, 
				Printer.printTime(start1 - start), symbol, loss_cur[0], 
				Printer.printTime(System.currentTimeMillis() - start1),loss_cur[1]);
		return loss_cur[0];
	}
	
	// Fast way to calculate the loss function
	// this is the crosscity loss of all
	public double loss() {
		double L = 0;
		return L;
	}
	
	public double[] loss( int a ) {
		double L = 0;
		return new double [] {L};
	}
	
	
	public double predict(int u ,int i ) {
		return U.row(u, false).inner(V.row(i, false));
	}
	
	
	public double predict_extra(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}
	
	
	@Override
	public void updateModel(int u, int i) {
		trainMatrix.setValue(u, i, 1);
		
		for (int iter = 0; iter < maxIterOnline; iter ++) {
			//update_user(u);
			
			//update_item(i);
		}
	}
}
