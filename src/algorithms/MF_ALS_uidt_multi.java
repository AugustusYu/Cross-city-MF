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


public class MF_ALS_uidt_multi extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double w0 = 0.01;	// weight for 0s
	double w0_c = 0;
	double reg = 0.01; 	// regularization parameters
	double init_mean = 0;  // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V
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
	
	double [][][] SU;
	double [][][] SV;
	double [][][] SUfinal;
	double [][][] SVfinal;
	
	public Integer [] [] buy_records;
	public int [] [] city_users;
	public int [] city_users_len;
	public int [] [] city_pois;
	public int [] city_pois_len;
	public int [] user_city;   //0
	public int [] poi_city;    //1
	
	int sharefactor = 0;
	int citynum = 2;

	double bigalpha = 0.5; // for user reg
	double bigbeta = 0.5;  // for poi reg 
	boolean loss_flag = false;
	boolean showProgress;
	boolean showLoss;
	
	public MF_ALS_uidt_multi(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
			int topK, int threadNum, int factors, int maxIter, double w0, double reg, 
			double init_mean, double init_stdev, boolean showProgress, boolean showLoss,int showbound,int showcount
			,int citynum) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0 ;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showLoss = showLoss;		
		this.citynum = citynum;
		this.showbound = showbound;
		this.showcount = showcount;
	}
	
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		//SU = U.transpose().mult(U);
		//SV = V.transpose().mult(V);
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
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		Ufinal = new DenseMatrix(userCount, factors);
		Vfinal = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		Ufinal.init(init_mean, init_stdev);
		Vfinal.init(init_mean, init_stdev);	
		
		SU = new double [citynum][factors][factors];
		SV = new double [citynum][factors][factors];
		SUfinal = new double [citynum][factors][factors];
		SVfinal = new double [citynum][factors][factors];
							
		for (int u = 0;u<userCount;u++) {
			if (true) {
				for (int f = 0;f<sharefactor;f++)
					Ufinal.set(u,f,U.get(u, f));	
				int this_ucity = user_city[u];
				for (int f=0; f<factors;f++)
					for (int k=f;k<factors;k++) {
						double val = U.get(u,f)*U.get(u,k);  
						SU[this_ucity][f][k] += val;
						if (f!=k) 
							SU[this_ucity][k][f] += val;
						val = Ufinal.get(u, k) *  Ufinal.get(u, f);
						SUfinal[this_ucity][f][k] += val;
						if (f!=k) 
							SUfinal[this_ucity][f][k] += val;						
					}
			}
		}
		for (int i = 0;i<itemCount;i++) {
			if (true) {
				int this_icity = poi_city[i];
				for (int f=0; f<factors;f++)
					for (int k=f;k<factors;k++) {
						double val = V.get(i,f)*V.get(i,k);  
						SV[this_icity][f][k] += val;
						if (f!=k) 
							SV[this_icity][k][f] += val;
						val = Vfinal.get(i, k) *  Vfinal.get(i, f);
						SVfinal[this_icity][f][k] += val;
						if (f!=k) 
							SVfinal[this_icity][k][f] += val;								
					}
			}				
		}
	}
	
	public void refresh_usercache() {
		for (int k = 0;k<factors;k++)
			for (int f = 0;f<factors;f++) 
				for (int city = 0;city<citynum;city++) {
					SU[city][k][f] = 0;
					SUfinal[city][k][f] = 0;
			}
		double val = 0;
		double tmp = 0;
		for(int u = 0;u<userCount;u++)
			for (int k = 0;k<factors;k++)
				for (int f = k;f<factors;f++) {
					if (k<sharefactor && f<sharefactor) {
						val = U.get(u, k) * U.get(u, f);
						tmp = Ufinal.get(u, k) *  Ufinal.get(u, f);
						if (val != tmp ) {
							System.out.printf("user:%d, index:%d,f=%d,k=%d\n",u,u,f,k);
							System.out.printf("user share vector error! %f != %f \n", val,tmp);
							System.exit(0);
						}
						else {
							SU[user_city[u]][k][f] += val;
							SU[user_city[u]][f][k]= SU[user_city[u]][k][f];
							SUfinal[user_city[u]][f][k]= SU[user_city[u]][k][f];
							SUfinal[user_city[u]][k][f]= SU[user_city[u]][k][f];
						}		
					}
					else {
						SU[user_city[u]][k][f] += U.get(u, k) * U.get(u, f);
						SU[user_city[u]][f][k]= SU[user_city[u]][k][f];
						SUfinal[user_city[u]][f][k] += Ufinal.get(u, k) *  Ufinal.get(u, f);
						SUfinal[user_city[u]][k][f]= SUfinal[user_city[u]][f][k];	
					}
		}		
	}
	
	
	public void buildModel() {
	}
	
	public void buildmulticityModel() {
		double loss_pre = Double.MAX_VALUE;
		refresh_usercache();
		for (int iter = 0; iter < maxIter; iter ++) {
			Long start = System.currentTimeMillis();
			
			for (int u = 0; u < userCount; u ++) {
				if (true) {	
					update_user_multi(u);
				}		
			}
			refresh_usercache();
			for (int i = 0; i < itemCount; i ++) {
				if (true) {
					if (bigalpha>0.001) {
						update_item_multi(i);
						//update_item_multi_home(i);
						//update_item_multi_tour(i);
					}
					else
						update_item_multi_alphaequalzero(i);
				}
			}	
			if (showProgress && (iter > showbound || iter % showcount == 0)) {
				long end_iter = System.currentTimeMillis();
				System.out.printf("iter = %d [%s]  ",iter,Printer.printTime(end_iter - start));			
				evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
			}
			if (showLoss) {			
				//cachecheck();
				loss_pre = showLoss(iter, start, loss_pre);
			}
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
	
	private void update_user_multi(int u) {
		ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
		int this_ucity = user_city[u];
		for (int i:itemList)
			rui[i] = predict(u,i);
		
		// udpate share part
		for(int k =0;k<sharefactor;k++) {
			double numerator = 0;
			double denominator = 0;
			//denominator = bigalpha*w0_c*SVextra[k][k]+w0*SVfinal[k][k]+(1+bigalpha)*reg;
			denominator = (1+bigalpha)*reg;
			for (int c = 0;c<citynum;c++)
				if (c!= this_ucity)
					denominator += w0*SVfinal[c][k][k];
				else
					denominator += bigalpha*w0*SV[c][k][k];
			
			for (int f = 0;f<factors;f++) {
				if (f!=k)
					//numerator += -bigalpha*w0_c*SVextra[k][f]*U.get(u, f)-w0*SVfinal[k][f]*Ufinal.get(u, f);
					for(int c = 0;c<citynum;c++)
						if (c!= this_ucity)
							numerator += -w0*SVfinal[c][k][f]*Ufinal.get(u, f);
						else
							numerator += -bigalpha*w0*SV[c][k][f]*U.get(u, f);			
			}
			for(int i:itemList) 
				if (poi_city[i] == user_city[u]){
					denominator += bigalpha*(1-w0_c)* V.get(i,k)*V.get(i, k);
					numerator += bigalpha*V.get(i, k)-bigalpha*(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
				}			
				else {
					denominator += (1-w0)* Vfinal.get(i,k)*Vfinal.get(i,k);
					numerator += Vfinal.get(i,k)-(1-w0)*(rui[i]-Ufinal.get(u,k)*Vfinal.get(i,k))*Vfinal.get(i,k);
				}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			else if (Double.isNaN(val)) {
				System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
				val = 1;
			}
				
			for (int i:itemList) {
				if (poi_city[i] == user_city[u])
					rui[i] += (val-U.get(u, k))*V.get(i, k);
				else
					rui[i] += (val-Ufinal.get(u,k))*Vfinal.get(i,k);			
			}		
			U.set(u, k, val);	
			Ufinal.set(u,k,val);
		}
		
		//update home part
		for(int k =sharefactor;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = w0_c*SV[this_ucity][k][k]+reg;
			for (int f = 0;f<factors;f++) {
				numerator += -w0_c*SV[this_ucity][k][f]*U.get(u, f);
			}
			numerator -= -w0_c*SV[this_ucity][k][k]*U.get(u, k);
			for(int i:itemList) 
				if (poi_city[i]==this_ucity){
				denominator += (1-w0_c)* V.get(i,k)*V.get(i, k);
				numerator += V.get(i, k)-(1-w0_c)*(rui[i]-U.get(u, k)*V.get(i, k))*V.get(i, k);
			}			
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			else if (Double.isNaN(val)) {
				System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
				val = 1;
			}
			for (int i:itemList) {
				if (poi_city[i]==this_ucity)
					rui[i] += (val-U.get(u, k))*V.get(i, k);
			}		
			U.set(u, k, val);	
		}	
		
		//update tour part 
		for(int k =sharefactor;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = reg;
			for (int c =0;c<citynum;c++)
				if (c!=this_ucity)
					denominator += w0*SVfinal[c][k][k];
			for (int f = 0;f<factors;f++) {
				for (int c =0;c<citynum;c++)
					if (c!=this_ucity)
						numerator += -w0*SVfinal[c][k][f]*Ufinal.get(u,f);
			}
			for (int c =0;c<citynum;c++)
				if (c!=this_ucity)
					numerator -= -w0*SVfinal[c][k][k]*Ufinal.get(u,k);
			for(int i:itemList) 
				if (poi_city[i]!=this_ucity){
				denominator += (1-w0)* Vfinal.get(i,k)*Vfinal.get(i,k);
				numerator += Vfinal.get(i,k)-(1-w0)*(rui[i]-Ufinal.get(u,k)*Vfinal.get(i,k))*Vfinal.get(i,k);
			}			
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			else if (Double.isNaN(val)) {
				System.out.printf("numerator = %f,denominator = %f",numerator,denominator);
				val = 1;
			}
			for (int i:itemList) {
				if (poi_city[i]!=this_ucity)
					rui[i] += (val-Ufinal.get(u,k))*Vfinal.get(i,k);
			}		
			Ufinal.set(u, k, val);	
		}
	}
	
	private void update_item_multi_alphaequalzero(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		int this_icity = poi_city[i];
		// update V -- home 
		DenseVector oldVector = V.row(i);
		for(int k =0;k<factors;k++) {
			V.set(i, k, Vfinal.get(i, k));				
		}
		for (int f = 0; f < factors; f ++){
			for (int k = 0; k <= f; k ++){
				double val = SV[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ V.get(i, f) * V.get(i, k);
				SV[this_icity][f][k] = val;
				SV[this_icity][k][f] = val;
			}
		} 
		//update Vfinal -- tour
		oldVector = Vfinal.row(i);
		for (int u:userList)
			if (user_city[u] != this_icity)
				rui_u[u] = predict(u,i);
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = bigbeta+reg;
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					denominator += w0*SUfinal[c][k][k];
			for (int f = 0;f<factors;f++) {
				for (int c = 0;c<citynum;c++)
					if (c!=this_icity)
						numerator += -w0*SUfinal[c][f][k]*Vfinal.get(i, f);
			}
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					numerator -= -w0*SUfinal[c][k][k]*Vfinal.get(i, k);	
			numerator += bigbeta * V.get(i,k);
			for(int u :userList) 
				if (user_city[u] != this_icity){
				denominator += (1-w0)*Ufinal.get(u, k)*Ufinal.get(u, k);
				numerator += Ufinal.get(u, k)-(1-w0)*(rui_u[u]-Ufinal.get(u, k)
						*Vfinal.get(i, k))*Ufinal.get(u, k);
			}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				if (user_city[u] != this_icity)
					rui_u[u] += (val - Vfinal.get(i, k))*Ufinal.get(u, k);
			}	
			Vfinal.set(i, k, val);				
		}
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SVfinal[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ Vfinal.get(i, f) * Vfinal.get(i, k);
				SVfinal[this_icity][f][k] = val;
				SVfinal[this_icity][k][f] = val;
			}
		} // end for f
	}

	private void cachecheck() {
		int cache_count = 0;
		double [][][] tmpuse = new double [citynum][factors][factors];
		for (int u=0;u<userCount;u++) 
			for (int k = 0;k<factors;k++)
				for(int f =k;f<factors;f++) {
			double a = U.get(u, k) *  U.get(u, f);
			tmpuse[user_city[u]][k][f] += a;
			if (k!=f)
				tmpuse[user_city[u]][f][k] += a;
				}
		boolean myflag = true;
		for (int k = 0;k<factors;k++)
			for(int f =k;f<factors;f++)
			for (int city = 0;city<citynum;city ++ ){
				if (tmpuse[city][k][f] - SU[city][k][f]<-0.1||tmpuse[city][k][f] - SU[city][k][f]>0.1)
					myflag = false;
			}
		if (myflag == true)
			cache_count++;
		else
			System.out.println("SU cache is wrongt!");
		
		tmpuse = new double [citynum][factors][factors];
		for (int u=0;u<userCount;u++) 
			for (int k = 0;k<factors;k++)
				for(int f =k;f<factors;f++) {
			double a = Ufinal.get(u, k) *  Ufinal.get(u, f);
			tmpuse[user_city[u]][k][f] += a;
			if (k!=f)
				tmpuse[user_city[u]][f][k] += a;
				}
		myflag = true;
		for (int k = 0;k<factors;k++)
			for(int f =k;f<factors;f++)
			for (int city = 0;city<citynum;city ++ ){
				if (tmpuse[city][k][f] - SUfinal[city][k][f]<-0.1||tmpuse[city][k][f] - SUfinal[city][k][f]>0.1)
					myflag = false;
			}
		if (myflag == true)
			cache_count++;
		else
			System.out.println("SUfinal cache is wrongt!");
		
		tmpuse = new double [citynum][factors][factors];
		for (int u=0;u<itemCount;u++) 
			for (int k = 0;k<factors;k++)
				for(int f =k;f<factors;f++) {
			double a = V.get(u, k) *  V.get(u, f);
			tmpuse[poi_city[u]][k][f] += a;
			if (k!=f)
				tmpuse[poi_city[u]][f][k] += a;
				}
		myflag = true;
		for (int k = 0;k<factors;k++)
			for(int f =k;f<factors;f++)
			for (int city = 0;city<citynum;city ++ ){
				if (tmpuse[city][k][f] - SV[city][k][f]<-0.1||tmpuse[city][k][f] - SV[city][k][f]>0.1)
					myflag = false;
			}
		if (myflag == true)
			cache_count++;
		else
			System.out.println("SV cache is wrongt!");
		
		tmpuse = new double [citynum][factors][factors];
		for (int u=0;u<itemCount;u++) 
			for (int k = 0;k<factors;k++)
				for(int f =k;f<factors;f++) {
			double a = Vfinal.get(u, k) *  Vfinal.get(u, f);
			tmpuse[poi_city[u]][k][f] += a;
			if (k!=f)
				tmpuse[poi_city[u]][f][k] += a;
				}
		myflag = true;
		for (int k = 0;k<factors;k++)
			for(int f =k;f<factors;f++)
			for (int city = 0;city<citynum;city ++ ){
				if (tmpuse[city][k][f] - SVfinal[city][k][f]<-0.1||tmpuse[city][k][f] - SVfinal[city][k][f]>0.1)
					myflag = false;
			}
		if (myflag == true)
			cache_count++;
		else
			System.out.println("SVfinal cache is wrongt!");
		
		System.out.println("finish cache check!");
	}
	
	private void update_item_multi(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		int this_icity = poi_city[i];
		// update V -- home 
		DenseVector oldVector = V.row(i);
		for (int u:userList)
			if (user_city[u] == this_icity)
				rui_u[u] = predict(u,i);
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator += SU[this_icity][k][k]*w0+bigbeta/bigalpha+reg;
			for (int f = 0;f<factors;f++) {
				numerator += -w0*SU[this_icity][f][k]*V.get(i, f);
			}
			numerator -= -w0*SU[this_icity][k][k]*V.get(i, k);	
			numerator += bigbeta/bigalpha * Vfinal.get(i,k);
			for(int u :userList) 
				if (user_city[u] == this_icity){
				denominator += (1-w0)*U.get(u, k)*U.get(u, k);
				numerator += U.get(u, k)-(1-w0)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
			}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				if (user_city[u] == this_icity)
					rui_u[u] += (val - V.get(i, k))*U.get(u,k);
			}	
			V.set(i, k, val);				
		}
		for (int f = 0; f < factors; f ++){
			for (int k = 0; k <= f; k ++){
				double val = SV[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ V.get(i, f) * V.get(i, k);
				SV[this_icity][f][k] = val;
				SV[this_icity][k][f] = val;
			}
		} 
		//update Vfinal -- tour
		oldVector = Vfinal.row(i);
		for (int u:userList)
			if (user_city[u] != this_icity)
				rui_u[u] = predict(u,i);
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = bigbeta+reg;
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					denominator += w0*SUfinal[c][k][k];
			for (int f = 0;f<factors;f++) {
				for (int c = 0;c<citynum;c++)
					if (c!=this_icity)
						numerator += -w0*SUfinal[c][f][k]*Vfinal.get(i, f);
			}
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					numerator -= -w0*SUfinal[c][k][k]*Vfinal.get(i, k);	
			numerator += bigbeta * V.get(i,k);
			for(int u :userList) 
				if (user_city[u] != this_icity){
				denominator += (1-w0)*Ufinal.get(u, k)*Ufinal.get(u, k);
				numerator += Ufinal.get(u, k)-(1-w0)*(rui_u[u]-Ufinal.get(u, k)
						*Vfinal.get(i, k))*Ufinal.get(u, k);
			}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				if (user_city[u] != this_icity)
					rui_u[u] += (val - Vfinal.get(i, k))*Ufinal.get(u, k);
			}	
			Vfinal.set(i, k, val);				
		}
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SVfinal[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ Vfinal.get(i, f) * Vfinal.get(i, k);
				SVfinal[this_icity][f][k] = val;
				SVfinal[this_icity][k][f] = val;
			}
		} // end for f
	}
	
	private void update_item_multi_home(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		int this_icity = poi_city[i];
		// update V -- home 
		DenseVector oldVector = V.row(i);
		for (int u:userList)
			if (user_city[u] == this_icity)
				rui_u[u] = predict(u,i);
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator += SU[this_icity][k][k]*w0+bigbeta/bigalpha+reg;
			for (int f = 0;f<factors;f++) {
				numerator += -w0*SU[this_icity][f][k]*V.get(i, f);
			}
			numerator -= -w0*SU[this_icity][k][k]*V.get(i, k);	
			numerator += bigbeta/bigalpha * Vfinal.get(i,k);
			for(int u :userList) 
				if (user_city[u] == this_icity){
				denominator += (1-w0)*U.get(u, k)*U.get(u, k);
				numerator += U.get(u, k)-(1-w0)*(rui_u[u]-U.get(u, k)*V.get(i, k))*U.get(u, k);
			}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				if (user_city[u] == this_icity)
					rui_u[u] += (val - V.get(i, k))*U.get(u,k);
			}	
			V.set(i, k, val);				
		}
		for (int f = 0; f < factors; f ++){
			for (int k = 0; k <= f; k ++){
				double val = SV[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ V.get(i, f) * V.get(i, k);
				SV[this_icity][f][k] = val;
				SV[this_icity][k][f] = val;
			}
		} 
	}
	
	private void update_item_multi_tour(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		int this_icity = poi_city[i];
		DenseVector oldVector = Vfinal.row(i);
		for (int u:userList)
				rui_u[u] = predict(u,i);
		for(int k =0;k<factors;k++) {
			double numerator = 0;
			double denominator = 0;
			denominator = bigbeta+reg;
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					denominator += w0*SUfinal[c][k][k];
			for (int f = 0;f<factors;f++) {
				for (int c = 0;c<citynum;c++)
					if (c!=this_icity)
						numerator += -w0*SUfinal[c][f][k]*Vfinal.get(i, f);
			}
			for (int c = 0;c<citynum;c++)
				if (c!=this_icity)
					numerator -= -w0*SUfinal[c][k][k]*Vfinal.get(i, k);	
			numerator += bigbeta * V.get(i,k);
			for(int u :userList) 
				if (user_city[u] != this_icity){
				denominator += (1-w0)*Ufinal.get(u, k)*Ufinal.get(u, k);
				numerator += Ufinal.get(u, k)-(1-w0)*(rui_u[u]-Ufinal.get(u, k)
						*Vfinal.get(i, k))*Ufinal.get(u, k);
			}
			double val = 0;
			if (denominator!= 0)
				val = numerator/denominator;
			if (Double.isNaN(val))
				val = 1;
			for (int u :userList) {
				if (user_city[u] != this_icity)
					rui_u[u] += (val - Vfinal.get(i, k))*Ufinal.get(u, k);
			}	
			Vfinal.set(i, k, val);				
		}
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SVfinal[this_icity][f][k] - oldVector.get(f) * oldVector.get(k)
						+ Vfinal.get(i, f) * Vfinal.get(i, k);
				SVfinal[this_icity][f][k] = val;
				SVfinal[this_icity][k][f] = val;
			}
		} // end for f
	}
	
	
	
	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		int a = 0;
		double [] loss_cur = new double [2];
		
		loss_cur = loss(a);	
		String symbol = loss_pre >= loss_cur[0] ? "-" : "+";
		if (symbol == "+")
			loss_flag = true;
//		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s], home: %.4f,tour: %.4f,beta:%.4f\n", iter, 
//				Printer.printTime(start1 - start), symbol, loss_cur[0], 
//				Printer.printTime(System.currentTimeMillis() - start1),loss_cur[1],loss_cur[2],loss_cur[3]);
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s], small loss: %.4f\n", iter, 
		Printer.printTime(start1 - start), symbol, loss_cur[0], 
		Printer.printTime(System.currentTimeMillis() - start1),loss_cur[1]);	
//		loss_cur = rawloss(a);
//		System.out.printf("Iter=%d [%s]\t [%s]raw loss: %.4f, home: %.4f,tour: %.4f,beta:%.4f\n", iter, 
//				Printer.printTime(start1 - start), symbol, loss_cur[0],loss_cur[1],loss_cur[2],loss_cur[3]);
//		
		return loss_cur[0];
	}
	
	public double loss() {
		double L = 0;
		return L;
	}
	
	public double []rawloss(int a ) {
		double Lhome = 0;
		double Ltour = 0;
		double Lbeta = 0;
		double Lhomeneg = 0;
		double Lhomepos = 0;
		//home 
		double lossreg = 0;
		double val = 0;
		for(int u = 0;u<userCount;u++)
			for (int f =0;f<factors;f++)
				lossreg += U.get(u, f)*U.get(u, f);
		for(int i = 0;i<itemCount;i++)
			for (int f =0;f<factors;f++)
				lossreg += V.get(i, f)*V.get(i, f);		
		Lhome += lossreg*reg;
		for (int u = 0;u<userCount;u++)
			for (int i = 0;i<itemCount;i++)
			if (user_city[u] == poi_city[i]){
				val = predict(u,i);
				if (trainMatrix.getValue(u,i)!=0) {
					Lhome += (1-val)*(1-val);
					Lhomepos += (1-val)*(1-val);
				}
					
				else {
					Lhome += w0 * val*val;
					Lhomeneg += w0 * val*val;
				}
			}		
		//tour
		lossreg = 0;
		for(int u = 0;u<userCount;u++)
			for (int f =0;f<factors;f++)
				lossreg += Ufinal.get(u, f)*Ufinal.get(u, f);
		for(int i = 0;i<itemCount;i++)
			for (int f =0;f<factors;f++)
				lossreg += Vfinal.get(i, f)*Vfinal.get(i, f);		
		Ltour += lossreg*reg;
		for (int u = 0;u<userCount;u++)
			for (int i = 0;i<itemCount;i++) 
				if (user_city[u] != poi_city[i]){
				val = predict(u,i);
				if (trainMatrix.getValue(u,i)!=0)
					Ltour += (1-val)*(1-val);
				else
					Ltour += w0 * val*val;
			}
		
		//beta
		
		for (int i = 0;i<itemCount;i++) 
			for(int f = 0;f<factors;f++){
				val = V.get(i, f) - Vfinal.get(i, f);
				Lbeta += val * val;
		}
		return new double [] {bigalpha*Lhome + Ltour + bigbeta*Lbeta,Lhome, Ltour, Lbeta, Lhomepos, Lhomeneg};
	}
	
	public double [] losstest(int a ) {
		double L = 0;
		double Lhome = 0;
		double Ltour = 0;
		double Lbeta = 0;
		double Lhomeneg = 0;
		double Lhomepos = 0;
		double tmpuse = 0;
		L = reg * (bigalpha*U.squaredSum() + bigalpha*V.squaredSum()+ Ufinal.squaredSum()+ Vfinal.squaredSum());
		Lhome +=  reg * (U.squaredSum() + V.squaredSum());
		Ltour += reg*(Ufinal.squaredSum()+ Vfinal.squaredSum());
		double tmpvalue = 0;
		for (int i = 0;i<itemCount;i++)
			for (int k = 0;k<factors;k++) {
				tmpvalue = V.get(i, k)-Vfinal.get(i, k);
				tmpvalue = tmpvalue * tmpvalue;
				L += bigbeta * tmpvalue;		
				Lbeta +=  tmpvalue;
		}
		for (int u = 0;u<userCount;u++) {
			double l = 0;
			double lhome=0;
			double ltour=0;
			for (int i : trainMatrix.getRowRef(u).indexList() )
				if (user_city[u] == poi_city[i]) {
					tmpuse = predict(u,i);
					l +=bigalpha*( ((1-w0_c)*tmpuse *tmpuse + 1 - 2 * tmpuse));
					lhome +=  ((1-w0_c)*tmpuse *tmpuse + 1 - 2 * tmpuse) ;
					Lhomepos += Math.pow(1 - predict(u, i), 2) ;
				}
				else {
					tmpuse = predict(u,i);
					l += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
					ltour += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
				}
			
			for (int c = 0;c<citynum;c++)
				if (c == user_city[u])
					for (int k = 0;k<factors;k++)
						for (int f = 0;f<factors;f++) {
							l += bigalpha*w0 * SV[c][k][f]*U.get(u, k)*U.get(u,f);
							lhome += w0 * SV[c][k][f]*U.get(u, k)*U.get(u,f);
							Lhomeneg +=  w0 * SV[c][k][f]*U.get(u, k)*U.get(u,f);
						}
				else
					for (int k = 0;k<factors;k++)
						for (int f = 0;f<factors;f++) {
							l += w0 * SVfinal[c][k][f]*Ufinal.get(u, k)*Ufinal.get(u,f);
							ltour += w0 * SVfinal[c][k][f]*Ufinal.get(u, k)*Ufinal.get(u,f);
						}			
			L += l;
			Lhome += lhome;
			Ltour += ltour;
		}	
		return new double [] {L,Lhome, Ltour, Lbeta};
	}

	public double [] loss(int a ) {
		double L = 0;
		double tmpuse = 0;
		L = reg * (bigalpha*U.squaredSum() + bigalpha*V.squaredSum()+ Ufinal.squaredSum()+ Vfinal.squaredSum());
		double tmpvalue = 0;
		for (int i = 0;i<itemCount;i++)
			for (int k = 0;k<factors;k++) {
				tmpvalue = V.get(i, k)-Vfinal.get(i, k);
				tmpvalue = tmpvalue * tmpvalue;
				L += bigbeta * tmpvalue;		
		}
		for (int u = 0;u<userCount;u++) {
			double l = 0;
			for (int i : trainMatrix.getRowRef(u).indexList() )
				if (user_city[u] == poi_city[i]) {
					tmpuse = predict(u,i);
					l +=bigalpha*( ((1-w0_c)*tmpuse *tmpuse + 1 - 2 * tmpuse));
				}
				else {
					tmpuse = predict(u,i);
					l += (1-w0)*tmpuse *tmpuse + 1 - 2 * tmpuse;
				}
			for (int c = 0;c<citynum;c++)
				if (c == user_city[u])
					for (int k = 0;k<factors;k++)
						for (int f = 0;f<factors;f++)
							l += bigalpha*w0 * SV[c][k][f]*U.get(u, k)*U.get(u,f);
				else
					for (int k = 0;k<factors;k++)
						for (int f = 0;f<factors;f++)
							l += w0 * SVfinal[c][k][f]*Ufinal.get(u, k)*Ufinal.get(u,f);			
			L += l;
		}	
		return new double [] {L,0};
	}
	
	public double predict(int u, int i) {
		if (user_city[u] == poi_city[i])
			return U.row(u, false).inner(V.row(i,false));
		else {
			return Ufinal.row(u, false).inner(Vfinal.row(i,false));
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
