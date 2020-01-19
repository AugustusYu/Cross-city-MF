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


public class MF_bpr_uidt extends TopKRecommender {
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
	DenseMatrix Ufinal;
	DenseMatrix Vfinal;
	double []rui ;
	double []rui_u;   
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
	
	public Integer [][] buy_stranger_target;
	public Integer [][] buy_stranger_source;
	public Integer [][] buy_local_target;

	public HashMap<Integer,Integer> stranger_index = new HashMap<Integer,Integer>();
	public HashMap<Integer,Integer> local_index = new HashMap<Integer,Integer>();
	
	public  HashSet<Integer> targetUsers ;
	public  HashSet<Integer> targetPois ;
	public  HashSet<Integer> extraUsers = new HashSet<Integer>();
	public  HashSet<Integer> extraPois = new HashSet<Integer>();
	
	public MF_bpr_uidt(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
			int topK, int threadNum, int factors, int maxIter, double lr, double reg, 
			double init_mean, double init_stdev, boolean showProgress, boolean showLoss,int showbound,int showcount) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr ;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showLoss = showLoss;
		
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
	
	public void sethashset(HashSet<Integer> A,HashSet<Integer> B) {
		this.targetUsers = A;
		this.targetPois = B;	
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
		System.out.printf("target user:%d target poi:%d \n",targetUsers.size(),targetPois.size());
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		Ufinal = new DenseMatrix(targetUsers.size(), factors);
		Vfinal = new DenseMatrix(targetPois.size(), factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		Ufinal.init(init_mean, init_stdev);
		Vfinal.init(init_mean, init_stdev);	
		

//		SUtarget = new double [factors][factors];
//		SVtarget = new double [factors][factors];
//		SUextra = new double [factors][factors];//extra dinmition equals zero 
//		SVextra = new double [factors][factors];
//		SUfinal = new double [factors][factors];
//		SVfinal = new double [factors][factors];		
		
		user_index = new int [userCount];
		poi_index = new int [itemCount];
		user_value = new int [targetUsers.size()];
		poi_value = new int [targetPois.size()];
		
		int pos = 0;
		for (int u = 0;u<userCount;u++) {
			if (targetUsers.contains(u)) {
				user_index[u] = pos;
				user_value[pos] = u;
				pos ++;
				for (int f = 0;f<sharefactor;f++)
					Ufinal.set(user_index[u],f,U.get(u, f));	
//				for (int f=0; f<factors;f++)
//					for (int k=f;k<factors;k++) {
//						double val = U.get(u,f)*U.get(u,k);  
//						SUtarget[f][k] += val;
//						if (f!=k) 
//							SUtarget[k][f] += val;
//						val = Ufinal.get(user_index[u], k) *  Ufinal.get(user_index[u], f);
//						SUfinal[f][k] += val;
//						if (f!=k) 
//							SUfinal[k][f] += val;						
//					}
			}
			else {
				extraUsers.add(u);
//				for (int f=0; f<factors;f++)
//					for (int k=f;k<factors;k++) {
//						double val = U.get(u,f)*U.get(u,k);  
//						SUextra[f][k] += val;
//						if (f!=k) SUextra[k][f] += val;
//					}					
			}
		}
		pos = 0;
		for (int i = 0;i<itemCount;i++) {
			if (targetPois.contains(i)) {
				poi_index[i] = pos;
				poi_value[pos] = i;
				pos ++;
				for (int f=0; f<factors;f++)
					for (int k=f;k<factors;k++) {
						double val = V.get(i,f)*V.get(i,k);  
//						SVtarget[f][k] += val;
//						if (f!=k) SVtarget[k][f] += val;

//						val = Vfinal.get(poi_index[i], k) *  Vfinal.get(poi_index[i], f);
//						SVfinal[f][k] += val;
//						if (f!=k) 
//							SVfinal[k][f] += val;								
					}
			}
			else {
				extraPois.add(i);
				for (int f=0; f<factors;f++)
					for (int k=f;k<factors;k++) {
						double val = V.get(i,f)*V.get(i,k);  
//						SVextra[f][k] += val;
//						if (f!=k) SVextra[k][f] += val;
					}				
			}					
		}
		//System.out.printf("initilize: matrix : %f\n",Vfinal.get(0, 0));
	
	}
	
//	public void refresh_usercache() {
//		for (int k = 0;k<factors;k++)
//			for (int f = 0;f<factors;f++) {
//				SUtarget[k][f] = 0;
//				SUextra[k][f] = 0;
//				SUfinal[k][f] = 0;
//			}
//		double val = 0;
//		double tmp = 0;
//		for(int u = 0;u<userCount;u++)
//			if (targetUsers.contains(u))
//				for (int k = 0;k<factors;k++)
//					for (int f = k;f<factors;f++) {
//						if (k<sharefactor && f<sharefactor) {
//							val = U.get(u, k) * U.get(u, f);
//							tmp = Ufinal.get(user_index[u], k) *  Ufinal.get(user_index[u], f);
//							if (val != tmp ) {
//								System.out.printf("user:%d, index:%d,f=%d,k=%d\n",u,user_index[u],f,k);
//								System.out.printf("user share vector error! %f != %f \n", val,tmp);
//								System.exit(0);
//							}
//							else {
//								SUtarget[k][f] += val;
//								SUtarget[f][k]= SUtarget[k][f];
//								SUfinal[f][k]= SUtarget[k][f];
//								SUfinal[k][f]= SUtarget[k][f];	
//							}		
//						}
//						else {
//							SUtarget[k][f] += U.get(u, k) * U.get(u, f);
//							SUtarget[f][k]= SUtarget[k][f];
//							SUfinal[f][k] += Ufinal.get(user_index[u], k) *  Ufinal.get(user_index[u], f);
//							SUfinal[k][f]= SUfinal[f][k];	
//						}
//					}
//			else for (int k = 0;k<factors;k++)
//				for (int f = k;f<factors;f++) {
//					val = U.get(u, k) * U.get(u, f);						
//					SUextra[k][f] += val;
//					SUextra[f][k] = SUextra[k][f];
//				}
//				
//	}
//	
	
	public void buildModel() {
		//no longer to be userd
	}
	
	public void buildcrosscityModel() {
		double loss_pre = Double.MAX_VALUE;
		int nonzeros = trainMatrix.itemCount();
		buy_stranger_target = new Integer [targetUsers.size()][];
		buy_stranger_source = new Integer [targetUsers.size()][];
		buy_local_target = new Integer [extraUsers.size()][];
		for (int u = 0 ;u<userCount;u++) {
			if (targetUsers.contains(u)) {
				stranger_index.put(u, stranger_index.size());
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> target_list = new ArrayList<Integer>() ;
				ArrayList<Integer> source_list = new ArrayList<Integer>() ;
				for (int i = 0;i<itemList.size();i++) {
					int poi = itemList.get(i);
					if (targetPois.contains(i))
						target_list.add(poi);
					else
						source_list.add(poi);
				}
				buy_stranger_target[stranger_index.get(u)] = target_list.toArray(new Integer [target_list.size()]); 
				buy_stranger_source[stranger_index.get(u)] = source_list.toArray(new Integer [source_list.size()]); 
			}
			else {
				local_index.put(u, local_index.size());
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				buy_local_target[local_index.get(u)] = itemList.toArray(new Integer [itemList.size()]);
			}
		}
		System.out.printf("finish ini in crosscity model, "
				+ "with %d stranger and %d local\n", stranger_index.size(),local_index.size());
			for (int iter = 0; iter < maxIter; iter ++) {
				refresh_usercache();
				Long start = System.currentTimeMillis();
				rand = new Random();
				for (int s = 0; s < nonzeros; s ++) { 
					double p = rand.nextDouble();
					//double p = 0.01;
					bpr_mix_update();
					
				}
				//System.out.printf("iter = %d: matrix : %f\n",iter,Vfinal.get(0, 1));
				//System.out.printf("iter = %d: score : %f\n",iter,predict(0,0));
				if (showProgress && (iter > showbound || iter % showcount == 0)) {
					long end_iter = System.currentTimeMillis();
					System.out.printf("iter = %d [%s]  ",iter,Printer.printTime(end_iter - start));			
					evaluatefor82crosscity_lda(testRatings,start,targetUsers,targetPois);
				}			
			}
			
	}
	
	public void buildcrosscityModel_runtime_test(double r) {
		double avertime = 0;
		int nonzeros = trainMatrix.itemCount();
		buy_stranger_target = new Integer [targetUsers.size()][];
		buy_stranger_source = new Integer [targetUsers.size()][];
		buy_local_target = new Integer [extraUsers.size()][];
		for (int u = 0 ;u<userCount;u++) {
			if (targetUsers.contains(u)) {
				stranger_index.put(u, stranger_index.size());
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				ArrayList<Integer> target_list = new ArrayList<Integer>() ;
				ArrayList<Integer> source_list = new ArrayList<Integer>() ;
				for (int i = 0;i<itemList.size();i++) {
					int poi = itemList.get(i);
					if (targetPois.contains(i))
						target_list.add(poi);
					else
						source_list.add(poi);
				}
				buy_stranger_target[stranger_index.get(u)] = target_list.toArray(new Integer [target_list.size()]); 
				buy_stranger_source[stranger_index.get(u)] = source_list.toArray(new Integer [source_list.size()]); 
			}
			else {
				local_index.put(u, local_index.size());
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				buy_local_target[local_index.get(u)] = itemList.toArray(new Integer [itemList.size()]);
			}
		}
		System.out.printf("finish ini in crosscity model, "
				+ "with %d stranger and %d local\n", stranger_index.size(),local_index.size());
		for (int iter = 0; iter < maxIter; iter ++) {
			refresh_usercache();
			Long start = System.currentTimeMillis();
			rand = new Random();
			for (int s = 0; s < nonzeros*r; s ++) { 
				double p = rand.nextDouble();
				bpr_mix_update();
				
			}
			long end_iter = System.currentTimeMillis();
			System.out.printf("iter = %d [%s]  \n",iter,Printer.printTime(end_iter - start));			
			long thistime = end_iter - start;
			double this_second = (double)(thistime)/1000;
			avertime = (avertime * iter + this_second)/(iter+1);
			System.out.printf("iter = %d,this time:%f, average time:%f  \n",iter,this_second,avertime);	
		}

	}
	
	
	
	
	
	
	
	
	
	
	
	
	private void refresh_usercache() {
		boolean flag = true ;
		for (int u = 0;u<userCount;u++)
			if (targetUsers.contains(u))
				for (int f =0;f<sharefactor;f++) {
					if (U.get(u, f) - Ufinal.get(user_index[u], f) > 0.01) {
						flag = false;
						Ufinal.set(user_index[u], f,U.get(u, f));
					}
				}	
		if (flag == false ) {
			System.out.printf("find unmatch usre cache in refresh_usercache\n");
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
	
	private void bpr_mix_update() {
		rand = new Random();
		int u = rand.nextInt(userCount);
		if (targetUsers.contains(u)) {
			//target user 
			ArrayList<Integer> buylist = trainMatrix.getRowRef(u).indexList();
			if (buylist.size()==0)
				return;
			int i = buylist.get(rand.nextInt(buylist.size()));
			int j = rand.nextInt(itemCount);
			while (trainMatrix.getValue(u, j)!=0)
				j =  rand.nextInt(itemCount);	
			double y_pos = predict(u,i);
			double y_neg = predict(u,j);
			double mult = - partial_loss(y_pos - y_neg);
			double grad_u =0;
			double grad = 0;
			if (targetPois.contains(i)&&targetPois.contains(j)) {
				for (int f = 0; f < factors; f ++) {
			    	grad_u = Vfinal.get(poi_index[i], f) - Vfinal.get(poi_index[j], f);
			    	Ufinal.add(user_index[u], f, -lr * (mult * grad_u + reg * Ufinal.get(user_index[u], f)));
			    	if (f<sharefactor)
			    		U.set(u, f, Ufinal.get(user_index[u], f));
			    	grad = Ufinal.get(user_index[u], f);
			    	Vfinal.add(poi_index[i], f, -lr * (mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[i], f) - bigbeta*V.get(i, f)));
			    	Vfinal.add(poi_index[j], f, -lr * (-mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[j], f)- bigbeta*V.get(j, f)));      
			    }	
			}
			else if ((!targetPois.contains(i))&&(!targetPois.contains(j))){
				for (int f = 0; f < factors; f ++) {
					grad_u = V.get(i, f) - V.get(j, f);
			    	U.add(u, f, -bigalpha*lr * (mult * grad_u + reg * U.get(u, f)));
			    	if (f<sharefactor) {
			    		Ufinal.set(user_index[u], f,U.get(u, f));
			    	}
			    	grad = U.get(u, f);
			    	V.add(i, f, -bigalpha*lr * (mult * grad + reg * V.get(i, f)));
			    	V.add(j, f, -bigalpha*lr * (-mult * grad + reg * V.get(j, f)));      
			    }
			}
			else if ((!targetPois.contains(i))&&(targetPois.contains(j))) {
				for (int f = 0; f < sharefactor; f ++) {
					grad_u = V.get(i, f) - Vfinal.get(poi_index[j], f);
			    	U.add(u, f, -bigalpha*lr * (mult * grad_u + reg * U.get(u, f)));
			    	if (f<sharefactor) {
			    		Ufinal.set(user_index[u], f,U.get(u, f));
			    	}
			    	grad = U.get(u, f);
			    	V.add(i, f, -bigalpha*lr * (mult * grad + reg * V.get(i, f)));
			    	Vfinal.add(poi_index[j], f, -lr * (-mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[j], f)- bigbeta*V.get(j, f)));           
			    }
			}
			else if ((targetPois.contains(i))&&(!targetPois.contains(j))) {
				for (int f = 0; f < sharefactor; f ++) {
			    	grad_u = Vfinal.get(poi_index[i], f) - V.get(j, f);
			    	U.add(u, f, -bigalpha*lr * (mult * grad_u + reg * U.get(u, f)));
			    	if (f<sharefactor) {
			    		Ufinal.set(user_index[u], f,U.get(u, f));
			    	}
			    	grad = U.get(u, f);
			    	Vfinal.add(poi_index[i], f, -lr * (mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[i], f) - bigbeta*V.get(i, f)));
			    	V.add(j, f, -bigalpha*lr * (-mult * grad + reg * V.get(j, f))); 
			    }
			}			
			
		}else {
			//extra user 
			int num = 0;
			int i = 0;
			int j = 0;
			num = buy_local_target[local_index.get(u)].length;
			if (num == 0) 
				return ;
			i = buy_local_target[local_index.get(u)][rand.nextInt(num)];
			while (!targetPois.contains(i))
				i = buy_local_target[local_index.get(u)][rand.nextInt(num)];
			j = rand.nextInt(itemCount);
			while (!targetPois.contains(j) || trainMatrix.getValue(u, j)!=0)
				j =  rand.nextInt(itemCount);
			double y_pos = predict(u,i);
			double y_neg = predict(u,j);
			double mult = - partial_loss(y_pos - y_neg);
			double grad_u =0;
			double grad = 0;
			for (int f = 0; f < factors; f ++) {
		    	grad_u = V.get(i, f) - V.get(j, f);
		    	U.add(u, f, -lr * (mult * grad_u + reg * U.get(u, f)));
		    	
		    	grad = U.get(u, f);
		    	V.add(i, f, -lr * (mult * grad + (reg+bigbeta) * V.get(i, f)-bigbeta*Vfinal.get(poi_index[i], f)));
		    	V.add(j, f, -lr * (-mult * grad + (reg+bigbeta) * V.get(j, f)-bigbeta*Vfinal.get(poi_index[j], f)));       
		    }
			
		}
	}
	
	private void bpr_update_stranger_target(int s) {
		//sample
		rand = new Random();
		int u = rand.nextInt(userCount);
		while (!targetUsers.contains(u))
			u = rand.nextInt(userCount);
		int num = buy_stranger_target[stranger_index.get(u)].length;
		if (num == 0) 
			return ;
		int i = buy_stranger_target[stranger_index.get(u)][rand.nextInt(num)];
		int j = rand.nextInt(itemCount);
		while (!targetPois.contains(j) || trainMatrix.getValue(u, j)!=0)
			j =  rand.nextInt(itemCount);
		
		//update
		double y_pos = predict(u,i);
		double y_neg = predict(u,j);
		double mult = - partial_loss(y_pos - y_neg);
		double grad_u =0;
		double grad = 0;
//		if (s< 10) {
//			System.out.printf("s:%d, u:%d,i:%d,j:%d\n", s,u,i,j);
//			System.out.printf("index,i:%d,j:%d\n", poi_index[i],poi_index[j]);
//			System.out.printf("ufinal: %f\n", Ufinal.get(user_index[u], 0));
//			System.out.printf("vfinal,i:%f,j:%f\n", Vfinal.get(poi_index[i], 0),Vfinal.get(poi_index[j], 0));
//			System.out.printf("v     ,i:%f,j:%f\n", V.get(i, 0),V.get(j, 0));
//			System.out.printf("%f,%f,%f\n",y_pos,y_neg,mult);
//		}
		for (int f = 0; f < factors; f ++) {
	    	grad_u = Vfinal.get(poi_index[i], f) - Vfinal.get(poi_index[j], f);
	    	Ufinal.add(user_index[u], f, -lr * (mult * grad_u + reg * Ufinal.get(user_index[u], f)));
	    	if (f<sharefactor)
	    		U.set(u, f, Ufinal.get(user_index[u], f));
	    	grad = Ufinal.get(user_index[u], f);
	    	Vfinal.add(poi_index[i], f, -lr * (mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[i], f) - bigbeta*V.get(i, f)));
	    	Vfinal.add(poi_index[j], f, -lr * (-mult * grad + (reg+bigbeta) * Vfinal.get(poi_index[j], f)- bigbeta*V.get(j, f)));      
	    }	
//		if (s<10) {
//			System.out.printf("lr:%f,bigbeta:%f,reg:%f\n", lr,bigbeta,reg);
//			System.out.printf("after iter:Ufinal:%f \nvfinal,i:%f,j:%f\n", 
//					Ufinal.get(user_index[u], 0),Vfinal.get(poi_index[i], 0),Vfinal.get(poi_index[j], 0));
//		}
		if(Double.isInfinite(grad)||Double.isInfinite(grad_u)) {
			System.out.print("INfinite num has been catched \n\n\n");
			System.exit(0);
		}  
		
	}
	
	private void bpr_update_stranger_source() {
		//sample
		rand = new Random();
		int u = rand.nextInt(userCount);
		while (!targetUsers.contains(u))
			u = rand.nextInt(userCount);
		int num = buy_stranger_source[stranger_index.get(u)].length;
		if (num == 0) 
			return ;
		int i = buy_stranger_source[stranger_index.get(u)][rand.nextInt(num)];
		int j = rand.nextInt(itemCount);
		while (targetPois.contains(j) || trainMatrix.getValue(u, j)!=0)
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
	    	if (f<sharefactor) {
	    		Ufinal.set(user_index[u], f,U.get(u, f));
	    	}
	    	grad = U.get(u, f);
	    	V.add(i, f, -lr * (mult * grad + reg * V.get(i, f)));
	    	V.add(j, f, -lr * (-mult * grad + reg * V.get(j, f)));      
	    }
		if(Double.isInfinite(grad)||Double.isInfinite(grad_u)) {
			System.out.print("INfinite num has been catched \n\n\n");
			System.exit(0);
		}  
	}
	
	private void bpr_update_local_target() {
		//sample
		rand = new Random();
		int u = rand.nextInt(userCount);
		while (targetUsers.contains(u))
			u = rand.nextInt(userCount);
		int num = buy_local_target[local_index.get(u)].length;
		if (num == 0) 
			return ;
		int i = buy_local_target[local_index.get(u)][rand.nextInt(num)];
		int j = rand.nextInt(itemCount);
		while (!targetPois.contains(j) || trainMatrix.getValue(u, j)!=0)
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
	    	V.add(i, f, -lr * (mult * grad + (reg+bigbeta) * V.get(i, f)-bigbeta*Vfinal.get(poi_index[i], f)));
	    	V.add(j, f, -lr * (-mult * grad + (reg+bigbeta) * V.get(j, f)-bigbeta*Vfinal.get(poi_index[j], f)));      
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
	
//	private void update_user(int u,int mode ) {
//		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
//		
//		switch(mode) {
//		case 0:{ // for extra user 
//			//DenseVector oldVector = U.row(u);
//			for (int i:itemList) {
//				rui[i] = U.row(u, false).inner(V.row(i, false));
//				if (!targetPois.contains(i))
//					System.out.println("illegal pair");
//			}int acount = 0;
//			for(int k =0;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = w0*SVtarget[k][k]+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0*SVtarget[k][f]*U.get(u, f);
//				}
//				numerator -= -w0*SVtarget[k][k]*U.get(u, k);
//				for(int i:itemList) {
//					denominator += (1-w0)* V.get(i,k)*V.get(i, k);
//					numerator += V.get(i, k)-(1-w0)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
//				}			
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				else 
//					
//				if (Double.isNaN(val)) {
//					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
//					val = 1;
//					acount +=1;
//				}
//				for (int i:itemList) {
//					rui[i] += (val-U.get(u, k))*V.get(i, k);
//				}		
//				U.set(u, k, val);	
//			}
//			if (acount > 3) System.exit(0);
////			for (int f = 0; f < factors; f ++) {
////				for (int k = 0; k <= f; k ++) {
////					double val = SUextra[f][k] - oldVector.get(f) * oldVector.get(k)
////							+ U.get(u, f) * U.get(u, k);
////					SUextra[f][k] = val;
////					SUextra[k][f] = val;
////				}
////			} // end for f
//			
//			
//		}break;
//		
//		case 1:{ //for U  Pus
//			//DenseVector oldVector = U.row(u);
//			for (int i:itemList)
//				//rui[i] = predict(u,i);
//				if (!targetPois.contains(i))
//					rui[i] = U.row(u, false).inner(V.row(i, false));
//			//for 0
//			for(int k =sharefactor;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = w0_c*SVextra[k][k]+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0_c*SVextra[k][f]*U.get(u, f);
//				}
//				numerator -= -w0_c*SVextra[k][k]*U.get(u, k);
//				//numerator += bigalpha * Ufinal.get(user_index[u],k);
//				for(int i:itemList) 
//					if (!targetPois.contains(i)){
//					denominator += (1-w0_c)* V.get(i,k)*V.get(i, k);
//					numerator += V.get(i, k)-(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
//				}			
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				else 
//					
//				if (Double.isNaN(val)) {
//					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
//					val = 1;
//				}
//				for (int i:itemList) {
//					if (!targetPois.contains(i))
//						rui[i] += (val-U.get(u, k))*V.get(i, k);
//				}		
//				U.set(u, k, val);	
//			}
//			
////			for (int f = 0; f < factors; f ++) {
////				for (int k = 0; k <= f; k ++) {
////					double val = SUtarget[f][k] - oldVector.get(f) * oldVector.get(k)
////							+ U.get(u, f) * U.get(u, k);
////					SUtarget[f][k] = val;
////					SUtarget[k][f] = val;
////				}
////			} // end for f
//		}break;	
//		
//		case 2 :{// U_final			
//			//DenseVector oldVector = Ufinal.row(user_index[u]);
//			for (int i:itemList)
//				//rui[i] = predict(u,i);
//				if (targetPois.contains(i))
//					rui[i] = Ufinal.row(user_index[u],false).inner(Vfinal.row(poi_index[i], false));
//			int acount = 0;
//			//for 0
//			for(int k =sharefactor;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = w0*SVfinal[k][k]+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0*SVfinal[k][f]*Ufinal.get(user_index[u],f);
//				}
//				numerator -= -w0*SVfinal[k][k]*Ufinal.get(user_index[u],k);
//				//numerator += bigalpha * U.get(u,k);
//				for(int i:itemList) 
//					if (targetPois.contains(i)){
//					denominator += (1-w0)* Vfinal.get(poi_index[i],k)*Vfinal.get(poi_index[i],k);
//					numerator += Vfinal.get(poi_index[i],k)-(1-w0)*(rui[i]-Ufinal.get(user_index[u],k)
//							*Vfinal.get(poi_index[i],k))*Vfinal.get(poi_index[i],k);
//				}			
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				else 
//					
//				if (Double.isNaN(val)) {
//					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
//					val = 1;
//					acount +=1;
//				}
//				for (int i:itemList) {
//					if (targetPois.contains(i))
//						rui[i] += (val-Ufinal.get(user_index[u],k))*Vfinal.get(poi_index[i],k);
//				}		
//				Ufinal.set(user_index[u], k, val);	
//			}
//			
//			if (acount > 3) System.exit(0);
////			for (int f = 0; f < factors; f ++) {
////				for (int k = 0; k <= f; k ++) {
////					double val = SUfinal[f][k] - oldVector.get(f) * oldVector.get(k)
////							+ Ufinal.get(user_index[u],k) * Ufinal.get(user_index[u],f);
////					SUfinal[f][k] = val;
////					SUfinal[k][f] = val;
////				}
////			} // end for f		
//		}break;
//		
//		
//		case 3:{ //for sharefacctor
//			for (int i:itemList)
//				if (!targetPois.contains(i))
//					rui[i] = predict_extra(u,i);
//				else
//					rui[i] = predict(u,i);
//			//for 0
//			for(int k =0;k<sharefactor;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = bigalpha*w0_c*SVextra[k][k]+w0*SVfinal[k][k]+(1+bigalpha)*reg;
//				for (int f = 0;f<factors;f++) {
//					if (f!=k)
//						numerator += -bigalpha*w0_c*SVextra[k][f]*U.get(u, f)
//							-w0*SVfinal[k][f]*Ufinal.get(user_index[u], f);
//				}
//				//numerator -= -bigalpha*w0_c*SVextra[k][k]*U.get(u, k)-w0*SVfinal[k][k]*U.get(u, k);
//				//numerator += bigalpha * Ufinal.get(user_index[u],k);
//				for(int i:itemList) 
//					if (!targetPois.contains(i)){
//						denominator += bigalpha*(1-w0_c)* V.get(i,k)*V.get(i, k);
//						numerator += bigalpha*V.get(i, k)-bigalpha*(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
//					}			
//					else {
//						denominator += (1-w0)* Vfinal.get(poi_index[i],k)*Vfinal.get(poi_index[i],k);
//						numerator += Vfinal.get(poi_index[i],k)-(1-w0)*(rui[i]-Ufinal.get(user_index[u],k)
//								*Vfinal.get(poi_index[i],k))*Vfinal.get(poi_index[i],k);
//					}
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				else 
//					
//				if (Double.isNaN(val)) {
//					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
//					val = 1;
//				}
//				for (int i:itemList) {
//					if (!targetPois.contains(i))
//						rui[i] += (val-U.get(u, k))*V.get(i, k);
//					else
//						rui[i] += (val-Ufinal.get(user_index[u],k))*Vfinal.get(poi_index[i],k);			
//				}		
//				U.set(u, k, val);	
//				Ufinal.set(user_index[u],k,val);
//			}
//		}break;	
//		
//		
//		case 4:{ //for share vector try 
//			DenseVector oldVector = U.row(u);
//			double []tryshow = new double [factors];
//			for (int i:itemList)
//				//rui[i] = predict(u,i);
//				if (targetPois.contains(i))
//					rui[i] = predict(u,i);
//				else
//					rui[i] = predict_extra(u,i);
//			
//			int acount = 0;
//			//for 0
//			for(int k =0;k<sharefactor;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = w0*SVfinal[k][k]+reg*(1+bigalpha)+w0_c*bigalpha*SVextra[k][k];
//				for (int f = 0;f<factors;f++) {
//					if (f == k) ;
//					else {
//						numerator += -w0*SVfinal[k][f]*U.get(u, f)-w0_c*bigalpha*SVextra[k][f]*U.get(u, f);
//					}
//				}
//				for(int i:itemList) {
//					if (targetPois.contains(i) ) {
//						denominator += (1-w0)* Vfinal.get(poi_index[i],k)*Vfinal.get(poi_index[i],k);
//						numerator += Vfinal.get(poi_index[i],k)-(1-w0)*(rui[i]-U.get(u, k)*Vfinal.get(poi_index[i],k))*Vfinal.get(poi_index[i],k);
//					}
//					if (!targetPois.contains(i) ) {
//					denominator += bigalpha*(1-w0_c)* V.get(i,k)*V.get(i, k);
//					numerator += bigalpha*V.get(i, k)-bigalpha*(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
//					}
//				}			
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				for (int i:itemList) {
//					if (targetPois.contains(i) )	
//						rui[i] += (val-U.get(u, k))*Vfinal.get(poi_index[i],k);
//					if ((!targetPois.contains(i)) )
//						rui[i] += (val-U.get(u, k))*V.get(i, k);
//				}		
//
//				tryshow[k] = val;
//			}
//			System.out.println(Arrays.toString(tryshow));
//			
//		}break;	
//		
//		
//		case 5:{ //for sharefacctor test
//			double []tryout = new double [factors];
//			for (int i:itemList)
//				if (!targetPois.contains(i))
//					rui[i] = predict_extra(u,i);
//				else
//					rui[i] = predict(u,i);
//			//for 0
//			for(int k =0;k<sharefactor;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = bigalpha*w0_c*SVextra[k][k]+w0*SVfinal[k][k]+(1+bigalpha)*reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -bigalpha*w0_c*SVextra[k][f]*U.get(u, f)-w0*SVfinal[k][f]*U.get(u, f);
//				}
//				numerator -= -bigalpha*w0_c*SVextra[k][k]*U.get(u, k)-w0*SVfinal[k][k]*U.get(u, k);
//				//numerator += bigalpha * Ufinal.get(user_index[u],k);
//				for(int i:itemList) 
//					if (!targetPois.contains(i)){
//						denominator += bigalpha*(1-w0_c)* V.get(i,k)*V.get(i, k);
//						numerator += bigalpha*V.get(i, k)-bigalpha*(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
//					}			
//					else {
//						denominator += (1-w0)* Vfinal.get(poi_index[i],k)*Vfinal.get(poi_index[i],k);
//						numerator += Vfinal.get(poi_index[i],k)-(1-w0)*(rui[i]-Ufinal.get(user_index[u],k)
//								*Vfinal.get(poi_index[i],k))*Vfinal.get(poi_index[i],k);
//					}
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				else 
//					
//				if (Double.isNaN(val)) {
//					System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
//					val = 1;
//				}
//				for (int i:itemList) {
//					if (!targetPois.contains(i))
//						rui[i] += (val-U.get(u, k))*V.get(i, k);
//					else
//						rui[i] += (val-Ufinal.get(user_index[u],k))*Vfinal.get(poi_index[i],k);			
//				}		
//				tryout[k] = val;
//			}System.out.println(Arrays.toString(tryout));
//		}break;	
//		
//		
//		}	
//	}
//	
//	private void update_item(int i,int mode ) {
//		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
//		switch(mode) {
//		case 0:{
//			DenseVector oldVector = V.row(i);
//			double []rui_u = new double [userCount];
//			for (int u :userList)
//				rui_u[u] = U.row(u, false).inner(V.row(i, false));
//			
//			for(int k =0;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = SUtarget[k][k]*w0_c+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0_c*SUtarget[f][k]*V.get(i, f);
//				}
//				numerator -= -w0_c*SUtarget[k][k]*V.get(i, k);					
//				for(int u :userList) {
//					denominator += (1-w0_c)*U.get(u, k)*U.get(u, k);
//					numerator += U.get(u, k)-(1-w0_c)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
//				}
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				if (Double.isNaN(val))
//					val = 1;
//				for (int u :userList) {
//					rui_u[u] += (val - V.get(i, k))*U.get(u,k);
//				}	
//				V.set(i, k, val);				
//			}
//			// Update the SV cache
//			for (int f = 0; f < factors; f ++) {
//				for (int k = 0; k <= f; k ++) {
//					double val = SVextra[k][f] - oldVector.get(f) * oldVector.get(k)
//							+ V.get(i, f) * V.get(i, k);
//					SVextra[k][f] = val;
//					SVextra[f][k] = val;
//				}
//			}
//		}break;
//		
//		case 1:{
//			DenseVector oldVector = V.row(i);
//			for (int u:userList)
//				if (!targetUsers.contains(u))
//					rui_u[u] = U.row(u, false).inner(V.row(i, false));
//			for(int k =0;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = SUextra[k][k]*w0+bigbeta+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0*SUextra[f][k]*V.get(i, f);
//				}
//				numerator -= -w0*SUextra[k][k]*V.get(i, k);	
//				numerator += bigbeta * Vfinal.get(poi_index[i],k);
//				for(int u :userList) 
//					if (!targetUsers.contains(u)){
//					denominator += (1-w0)*U.get(u, k)*U.get(u, k);
//					numerator += U.get(u, k)-(1-w0)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
//				}
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				if (Double.isNaN(val))
//					val = 1;
//				for (int u :userList) {
//					if (!targetUsers.contains(u))
//						rui_u[u] += (val - V.get(i, k))*U.get(u,k);
//				}	
//				V.set(i, k, val);				
//			}
//			for (int f = 0; f < factors; f ++) {
//				for (int k = 0; k <= f; k ++) {
//					double val = SVtarget[f][k] - oldVector.get(f) * oldVector.get(k)
//							+ V.get(i, f) * V.get(i, k);
//					SVtarget[f][k] = val;
//					SVtarget[k][f] = val;
//				}
//			} // end for f
//		}break;
//			
//		case 2:{
//			DenseVector oldVector = Vfinal.row(poi_index[i]);
//			for (int u:userList)
//				if (targetUsers.contains(u))
//					rui_u[u] = Ufinal.row(user_index[u], false).inner(Vfinal.row(poi_index[i]));
//			for(int k =0;k<factors;k++) {
//				double numerator = 0;
//				double denominator = 0;
//				denominator = w0*SUfinal[k][k]+bigbeta+reg;
//				for (int f = 0;f<factors;f++) {
//					numerator += -w0*SUfinal[f][k]*Vfinal.get(poi_index[i], f);
//				}
//				numerator -= -w0*SUfinal[k][k]*Vfinal.get(poi_index[i], k);	
//				numerator += bigbeta * V.get(i,k);
//				for(int u :userList) 
//					if (targetUsers.contains(u)){
//					denominator += (1-w0)*Ufinal.get(user_index[u], k)*Ufinal.get(user_index[u], k);
//					numerator += Ufinal.get(user_index[u], k)-(1-w0)*(rui_u[u]-Ufinal.get(user_index[u], k)
//							*Vfinal.get(poi_index[i], k))*Ufinal.get(user_index[u], k);
//				}
//				double val = 0;
//				if (denominator!= 0)
//					val = numerator/denominator;
//				if (Double.isNaN(val))
//					val = 1;
//				for (int u :userList) {
//					if (targetUsers.contains(u))
//						rui_u[u] += (val - Vfinal.get(poi_index[i], k))*Ufinal.get(user_index[u], k);
//				}	
//				Vfinal.set(poi_index[i], k, val);				
//			}
//			for (int f = 0; f < factors; f ++) {
//				for (int k = 0; k <= f; k ++) {
//					double val = SVfinal[f][k] - oldVector.get(f) * oldVector.get(k)
//							+ Vfinal.get(poi_index[i], f) * Vfinal.get(poi_index[i], k);
//					SVfinal[f][k] = val;
//					SVfinal[k][f] = val;
//				}
//			} // end for f
//		}break;
//		
//		}
//	}
//	
	public void output_vector(String p) {
		FileWriter out1 = null;
		try {
		out1 = new FileWriter(p);
		String str_head = "user:\n";
		out1.write(str_head);
		for (int u =0;u<userCount;u++)
			{
				if (targetUsers.contains(u)) {
				String str = String.format("%d:{%s},{%s}\n",u,U.row(u),Ufinal.row(user_index[u]));
				out1.write(str);
				}
			}	
		str_head = "poi:\n";
		out1.write(str_head);
		for (int i =0;i<itemCount;i++)
			{
				if (targetPois.contains(i)) {
				String str = String.format("%d:{%s},{%s}\n",i,V.row(i),Vfinal.row(poi_index[i]));
				out1.write(str);
				}
			}	
		out1.close();
		System.out.print("finish write file vector\n");
		System.out.println(p);
		}
		catch (Exception e) {   

            e.printStackTrace();   

        }
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
//		loss_cur = rawloss();
//		System.out.printf("[%s] rawloss:%f rawsamllss:%f, regpart:%f, target user part :%f\n", 
//				Printer.printTime(System.currentTimeMillis() - start1),loss_cur[0],loss_cur[1]
//				,loss_cur[2],loss_cur[3]);
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
	
	public double [] rawloss(){
		double regpart = 0;
		double targetu = 0;
		double L = 0;
		double crossL = 0;
		double tmp = 0;
		double []lcheck = new double[4];
		double ul = 0;
		double a = 0;
		double mtt = 0;
		double userreg = 0;
		double itemreg = 0;
		for (int u =0;u<userCount;u++) 
			if (targetUsers.contains(u)){	
				a = 0;
				for (int k =0;k<factors;k++) {
					a += bigalpha*reg * U.get(u, k)*U.get(u, k);
					a += reg*Ufinal.get(user_index[u], k)*Ufinal.get(user_index[u], k);
				}
				ul = L;
				L += a ;
				//regpart += a;
				mtt = L;
				for (int i = 0;i<itemCount;i++) {
					if (targetPois.contains(i)) 
						if (trainMatrix.getValue(u,i)!=0) {
							L += (1-predict(u,i))*(1-predict(u,i));
							crossL += (1-predict(u,i))*(1-predict(u,i));
							//a += (1-predict(u,i))*(1-predict(u,i)) - w0*predict(u,i)*predict(u,i);
						}
						else {
							L += w0*(predict(u,i))*(predict(u,i));
							crossL += w0*(predict(u,i))*(predict(u,i));
						}
					else
						if (trainMatrix.getValue(u,i)!=0) {
							L += bigalpha*(1-predict_extra(u,i))*(1-predict_extra(u,i));
							//a += (1-predict_extra(u,i))*(1-predict_extra(u,i)) - w0*predict_extra(u,i)*predict_extra(u,i);
						}
						else
							L += bigalpha*w0_c*(predict_extra(u,i))*(predict_extra(u,i));
				}
				ul = L - ul;
//				if (u==1 )
//					System.out.printf("target user:%d,total %f, train %f, minus %f\n", u,ul,a,ul-a);
				targetu += L - mtt;
			}
		else {
			for (int k =0;k<factors;k++) {
				L += reg * U.get(u, k)*U.get(u, k);
				regpart += reg * U.get(u, k)*U.get(u, k);
				lcheck[0] += reg * U.get(u, k)*U.get(u, k);
			}
			ul = L;
			for (int i:targetPois) {
				if (trainMatrix.getValue(u,i)!=0)
					L += (1-predict_extra(u,i))*(1-predict_extra(u,i));
				else
					L += w0*(predict_extra(u,i))*(predict_extra(u,i));
			}
//			ul = L - ul;
//			if (u<5)
//				System.out.printf("extra user:%d,%f\n", u,ul);
		}
		userreg = regpart;
		for (int i : targetPois) {
			for (int k = 0;k<factors;k++) {
				tmp = V.get(i, k)-Vfinal.get(poi_index[i], k);
				tmp = tmp * tmp;
				L += bigbeta*tmp;
				crossL += bigbeta*tmp;
				lcheck[1] += bigbeta*tmp;
			}
		}
		for (int i=0;i<itemCount;i++) 
			if (!targetPois.contains(i)){
				for (int k =0;k<factors;k++) {
					L += bigalpha*reg * V.get(i, k)*V.get(i, k);
					regpart += bigalpha*reg * V.get(i, k)*V.get(i, k);
					lcheck[1] += reg * V.get(i, k)*V.get(i, k);
				}
			}else {
				for (int k =0;k<factors;k++) {
					L += reg * Vfinal.get(poi_index[i], k)*Vfinal.get(poi_index[i], k);
					regpart += reg * Vfinal.get(poi_index[i], k)*Vfinal.get(poi_index[i], k);
					L += reg * V.get(i, k)*V.get(i, k);
					regpart += reg * V.get(i, k)*V.get(i, k);
				}
			}
		itemreg = regpart - userreg;
		System.out.printf("raw user reg : %f,raw item reg : %f\n",userreg,itemreg);
		return (new double [] {L,crossL,regpart,targetu});
	}
	
//	public double[] loss(int a) {
//		//System.out.printf("%d user, %d poi", targetUsers.size(),targetPois.size());
//		//double crossl = 0;
//		double L  = 0;	
//		double tmpvalue;
//		
//		for (int i :extraPois) 
//			for (int f = 0;f<factors;f++){
//			L += (bigalpha - 1) *reg* V.get(i, f) *  V.get(i, f);
//		}
//		for (int u:extraUsers)
//			for (int f = 0;f<factors;f++) {
//				L += (1-bigalpha) * reg * U.get(u, f) * U.get(u, f);
//			}
//		L += reg * (bigalpha*U.squaredSum() + V.squaredSum() 
//			+ Vfinal.squaredSum()+Ufinal.squaredSum());
//		for(int i = 0;i<itemCount;i++)
//			if (targetPois.contains(i))
//				for (int k = 0;k<factors;k++) {
//					tmpvalue = V.get(i, k)-Vfinal.get(poi_index[i], k);
//					tmpvalue = tmpvalue * tmpvalue;
//					L += bigbeta * tmpvalue;
//					//crossl += bigbeta * tmpvalue;
//				}
//		double tmpuse;
//		for (int u = 0; u < userCount; u ++) {
//			double l = 0;
//			double small = 0;
//			for(int i : trainMatrix.getRowRef(u).indexList()) {
//				if (targetPois.contains(i)&&targetUsers.contains(u)) {
//					tmpuse = predict(u, i);
//					l += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
//					//small += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
//					//l += Math.pow(1 - predict(u, i), 2);
//					//targetu += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
//				}
//				else if (extraPois.contains(i)&&targetUsers.contains(u)) {
//					tmpuse = predict_extra(u, i);
//					l +=bigalpha*( ((1-w0_c)*tmpuse *tmpuse + 1 - 2 * tmpuse));
//					//targetu += bigalpha*( ((1-w0_c)*tmpuse *tmpuse + 1 - 2 * tmpuse));
//				}			
//				else if (targetPois.contains(i)&&extraUsers.contains(u)) {
//					tmpuse = predict_extra(u, i);
//					l += ((1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse);
//				}
//				else {
//					System.out.println("error in algorithm:loss count!");
//					System.exit(1);
//				}
//			}
//			if (targetUsers.contains(u))
//				for (int k = 0;k<factors;k++)
//					for (int f = 0;f<factors;f++) {
//						l += bigalpha*U.get(u, k)*U.get(u,f)*SVextra[k][f]*w0_c;
//						l += Ufinal.get(user_index[u], f)*Ufinal.get(user_index[u], k)*SVfinal[k][f]*w0;
//						//small += Ufinal.get(user_index[u], f)*Ufinal.get(user_index[u], k)*SVfinal[k][f]*w0;	
//						//targetu += bigalpha*U.get(u, k)*U.get(u,f)*SVextra[k][f]*w0_c +
//						//		Ufinal.get(user_index[u], f)*Ufinal.get(user_index[u], k)*SVfinal[k][f]*w0;
//					}
//			else
//				for (int k = 0;k<factors;k++)
//					for (int f = 0;f<factors;f++)
//						l +=U.get(u, k)*U.get(u,f)*SVtarget[k][f]*w0;
//			L += l;
////			if (u==1)
////				System.out.printf("user %d has value for loss:%f,part %f, minus %f\n",u,l,b,l-b);
//			
//			//crossl += small;
//		}
//		//System.out.printf("reg part:%f, target user part:%f\n", regpart,targetu);
//		return new double[]{L,0};
//	}
//	

	public double predict(int u, int i) {
		if (targetUsers.contains(u) && targetPois.contains(i))
			return Ufinal.row(user_index[u], false).inner(Vfinal.row(poi_index[i],false));
		else {
			return U.row(u, false).inner(V.row(i, false));
			//System.out.println("illegal use for predict");
		}
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
