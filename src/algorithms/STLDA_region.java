package algorithms;
import java.util.*;

//import org.apache.commons.math3.geometry.partitioning.Region.Location;
import org.apache.commons.math3.special.Beta;

import utility.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.math.*;

import Jama.*;
import data_structure.Rating;
import data_structure.SparseMatrix;
public class STLDA_region extends TopKRecommender{
   
	public  HashMap<Integer,ArrayList<CheckInRecord>> usercheckinset;
	//public HashMap<Integer,ArrayList<CheckInRecord>> testcheckinset;
	public  HashMap<Integer,Location> poi_location;
	//public  HashMap<Integer,Location> user_home_location;
	   public Location user_home_location[];
	  // public Location poi_location[];
	   public int poi_region_innitilization[];
	    public int V; //Number of POI
	    public int K; //Number of topics
	    public int R; // Number of Regions
	    public int U; //Number of Users
	    public int W; //Number of words

	    public int number_of_parameter_update; 
	    // Dirichlet parameter 
	    public double alpha;//
	  //  public double alpha_sum;
	    
	    public double gamma;//
	   // public double gamma_sum;
	    
	    
	    public double beta;
	   // public double beta_sum;
	    
	    public double eta;
	   // public double eta_sum;
	    
	    
	    public int[][] user_Region_Count;
	    public int[][][] user_Region_Topic_Count;
	    public int[][] user_Region_Number_of_Checkins;
	    public int[] topic_Count;
	    public double[] topic_Count_sum;
	    public int[] region_Count;
	    public double[] region_Count_sum;
	    
	    /*
	    public double[] topic_Popularity;
	    public double[] region_Popularity;
	    */
	    
	    public int[][] topic_Word_Count;
	    public int[][] Region_POI_Count;
	    
	    
	    public Location region_mu[];
	    public double region_covariance[][];
	    public Matrix region_covariance_M[];
	    
	    public Location region_mu_sum[];
	    public double region_covariance_sum[][];
	    public Matrix region_covariance_sum_M[];
	    
	    //public double topic_time_parameters[][];//dimension K*2
	    //public double topic_time_parameters_sum[][];//dimension K*2
	    
	    public double[][] user_Region_Distribution;
	    public double[][][] user_Region_Topic_Distribution;
	    
	    public double[][] topic_word_Distribution;
	    public double[][] region_POI_Distribution;
	    
	    public int ITERATIONS;
	    public int SAMPLE_LAG;
	    public int BURN_IN;
	    
         public String outputPath; 
         public int iter = 0;
	   
         
         public HashMap<Integer,ArrayList<Location>> perregion_assignments_list;
         //public HashMap<Integer,ArrayList<Double>> pertopic_assignments_list;
	    public HashMap<Integer,ArrayList<Integer>>friendnetwork;
	    
	    //new pare by ygh
	    public double [][] poi_topic_score;
	    public HashMap<Integer,POI> POIset=new HashMap<Integer,POI>();
	    
		public  HashSet<Integer> targetUsers  = new HashSet<Integer>();;
		public  HashSet<Integer> targetPois  = new HashSet<Integer>();;
		public  HashSet<Integer> extraUsers = new HashSet<Integer>();
		public  HashSet<Integer> extraPois = new HashSet<Integer>();
	    
		public void sethashset(HashSet<Integer> A,HashSet<Integer> B) {
			this.targetUsers = A;
			this.targetPois = B;	
			for (int u = 0;u<userCount;u++)
				if (!targetUsers.contains(u))
					extraUsers.add(u);
			for ( int i =0 ;i<itemCount;i++)
				if (!targetPois.contains(i))
					extraPois.add(i);
			System.out.println("hash set has been available!");
		}
		
//		public MF_ALS_mix_changeralpha(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
//				int topK, int threadNum, int factors, int maxIter, double w0, double reg, 
//				double init_mean, double init_stdev, boolean showProgress, boolean showLoss,int showbound,int showcount) {
//			super(trainMatrix, testRatings, topK, threadNum);
		
		
		public STLDA_region(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
				int topK, int threadNum,int K,int R,  double alpha, double beta, double eta,
				double gamma,int ITERATIONS,int SAMPLE_LAG,int BURN_IN ,String catefile) throws IOException{
			super(trainMatrix, testRatings, topK, threadNum);
		 	this.K=K;
	    	this.R=R;
	    	this.alpha=alpha;
	    	this.beta=beta;
	    	this.eta=eta;
	    	this.gamma=gamma;
	    	this.ITERATIONS=ITERATIONS;
	    	this.SAMPLE_LAG=SAMPLE_LAG;
	    	this.BURN_IN=BURN_IN;
	    	readcatefile(catefile);
	    	
	    	refreshparas();
	    	this.poi_topic_score = new double [V][K];
	    	
	    	user_Region_Count=new int[U][R];
	    	user_Region_Topic_Count=new int[U][R][K];
	    	user_Region_Number_of_Checkins=new int[U][R];
	    	topic_Count=new int[K];
	    	region_Count=new int[R];
	    	topic_Word_Count=new int[K][W];
	    	Region_POI_Count=new int[R][V];
	    	
	    	this.topic_Count_sum=new double[K];
	    	this.region_Count_sum=new double[R];
	    	
	    	region_mu=new Location[R];
	    	region_covariance=new double[R][2];
	    	region_mu_sum=new Location[R];
	    	for (int i =0;i<R;i++) {
	    		region_mu_sum[i] = new Location(0,0);	    	
	    	}
	    	
	    	region_covariance_sum=new double[R][2];
	    	region_covariance_M=new Matrix[R];
	    	region_covariance_sum_M=new Matrix[R];
	    	for (int i =0;i<R;i++) {
	    		double [][] A = new double [2][2];	    		
	    		region_covariance_sum_M[i] = new Matrix(A);
	    	}
	    	   	
	    	user_Region_Distribution=new double[U][R];
	    	user_Region_Topic_Distribution=new double[U][R][K];
	    	
	    	topic_word_Distribution=new double[K][W];
	    	region_POI_Distribution=new double[R][V];
	    	
	    	outputPath=null;
	    	this.ITERATIONS=ITERATIONS;
	    	this.SAMPLE_LAG=SAMPLE_LAG;
	    	this.BURN_IN=BURN_IN;
	    	
	    	usercheckinset=new  HashMap<Integer,ArrayList<CheckInRecord>>();
	    	
	    	
	    	user_home_location=new Location[U];
	    	poi_region_innitilization=new int[V];
	    	
	    	perregion_assignments_list=new HashMap<Integer,ArrayList<Location>>();
	    	for(int i=0;i<R;i++)
	    	{
	    		ArrayList<Location> li=new ArrayList<Location>();
	    		perregion_assignments_list.put(i,li);
	    	}
	    	
	    	System.out.println("STLDA model has been installed!");
		};
		
		public void refreshparas() {
			this.W = wordCount;
			U = userCount;
			V = itemCount;
		}
		
		
	    
	   
	    public void train(String KMeansFile) throws IOException
	    {
	    	initializeModel(KMeansFile);
	    	
	    	gibbsSampling(0);
	    	
	    	calDistribution();
	    	
	    	//output_model();
	    }
	    
	    public void buildcrosscitymodel(int totaliter,String kmeans) throws IOException{
	    	initializeModel(kmeans);
	    	System.out.println("finish initialize, begin iter!");
	    	for ( int iter = 0;iter<totaliter ;iter ++ ) {
	    		Long start = System.currentTimeMillis();
	    	  	gibbsSampling(iter);	
		    	calDistribution();
		    	refresh_poi_topic();
		    	System.out.printf("iter = %d:", iter*ITERATIONS +ITERATIONS );
		    	evaluatefor82crosscity_stlda_region(testRatings,start,targetUsers,targetPois);
	    	}
	    };
	    
	    public void readcatefile(String catefile) throws IOException{
	    	readcategary_location_yelp(catefile,POIset);
	    };
	    

	    
	    public void data_reshape() {
	    	//trainMatrix --> POIset usercheckinlist
	    	for (int u = 0;u<userCount;u ++) 
	    		for (int i = 0;i<itemCount;i++) {
	    			if (!usercheckinset.containsKey(u)) {
						ArrayList<CheckInRecord> list=new ArrayList<CheckInRecord>();
						usercheckinset.put(u,list);
	    			}
	    			if (trainMatrix.getValue(u,i)!=0) {
						CheckInRecord record=new CheckInRecord ();
						record.user_id=u;
						record.time=Math.random();
						record.poi=POIset.get(i);
						if (targetPois.contains(i))
							record.assigned_region = 0;
						else 
							record.assigned_region = 1;
						usercheckinset.get(u).add(record);
	    			}
	    			
	    			
	    			
	    			
	    		}
	    	
	    	
	    	this.W = wordCount;
	    	System.out.println("data reshape success");
	    }
	    
	    public void buildModel() {
	    	
	    }
	    
	    public void updateModel(int u,int i) {
	    	
	    }
	    
	    public void refresh_poi_topic() {
	    	for (int i = 0;i<V;i++) {
	    		ArrayList<Integer> words=POIset.get(i).wordset;
	    		int n = words.size();
	    		if (n == 0) {
	    			break;
	    		}
	    		else {
	    			double s = 1;
	    			for (int t = 0;t<K;t++) {
	    				s = 1;
	    				for (int j : words)
	    					s = s * topic_word_Distribution[t][j];
	    				s = Math.pow(s, 1.0/(double)n);
	    				poi_topic_score[i][t] = s;
	    			}
	    		}
	    		
	    	}
	    		    	    	
	    }
	    
	    public void random_initializa(HashMap<Integer,ArrayList<Location>> perregion_assignments_list,int[] poi_region_inni) {
//	    	int poi = V;
//	    	int maxregion = R;
//	    	poi_location=new  HashMap<Integer,Location>();
//      	  	for(int i=0; i<U; i++){
//		    	int length =usercheckinset.get(i).size();
//		    	for(int j=0; j<length; j++) {
//		    		CheckInRecord re=usercheckinset.get(i).get(j);
//		    		poi_location.put(re.poi.poi_id, re.poi.location);
//		    	}
//      	  	}
//		    		
//		    		
//	    	for (int i = 0;i<poi;i++) {
//	    		int r = (int)(Math.random()*maxregion);
//	    		perregion_assignments_list.get(r).add(poi_location.get(i));
//	    		poi_region_innitilization[i] = r;
//	    		
//	    	}
	    	
	    };
	    
	    public void crosscity_initialize(String kmeansfile) throws IOException{
	    	// region 0=beijing 1=other 
	    	int poi = V;
	    	int maxregion = R;
	    	poi_location=new  HashMap<Integer,Location>();
      	  	for(int i=0; i<U; i++){
		    	int length =usercheckinset.get(i).size();
		    	for(int j=0; j<length; j++) {
		    		CheckInRecord re=usercheckinset.get(i).get(j);
		    		poi_location.put(re.poi.poi_id, re.poi.location);
		    	}
      	  	}
      	  	
      	  	read_kmeans_file(kmeansfile);
	    }
	    
	    private void read_kmeans_file( String kmeansfile) throws IOException{
	    	
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(new FileInputStream(kmeansfile)));
			String line;
			int count = 0;
			while((line = reader.readLine()) != null) {
				String[] arr = line.split("\t");
				int poi = Integer.parseInt(arr[0]);
				int region = Integer.parseInt(arr[1])-1;
				poi_region_innitilization[poi] = region;
				if (region >= R) {
					System.out.printf("region error! exiting now\n");
					System.exit(0);
				}
				perregion_assignments_list.get(region).add(poi_location.get(poi));
				count ++;
			}
			reader.close();
			System.out.printf("kmeans file has been read for %d pois\n", count);
	    };
	    
          public void initializeModel(String KMeansFile) throws IOException
          {
        	  number_of_parameter_update=0;
        	// initialize region assignments r
	 	    	//utility.ReadInput.initializeModel(KMeansFile, perregion_assignments_list,poi_region_innitilization);
	 	    	
        	  //random_initializa(perregion_assignments_list,poi_region_innitilization);
        	  crosscity_initialize(KMeansFile);
        	  
        	  
        	  for(int i=0; i<U; i++){
		    		int length =usercheckinset.get(i).size();
		    		for(int j=0; j<length; j++)
		    		{
		    			CheckInRecord re=usercheckinset.get(i).get(j);
		    			int region=poi_region_innitilization[re.poi.poi_id];
		    			re.assigned_region=region;
		    			this.user_Region_Count[i][region]++;
		    			this.region_Count[region]++;
		    			this.Region_POI_Count[region][re.poi.poi_id]++;
		    			
		    		}
		    		
	 	    	}
	 	    	
	 	    	
	    	// initialize topic assignments z
	         
	    	 for(int i=0; i<U; i++){
	    		int length =usercheckinset.get(i).size();
	    		
	    		for(int j=0; j<length; j++){
	    			int ran = (int) (Math.random() * K);
	    			CheckInRecord re=usercheckinset.get(i).get(j);
	    			re.assigned_topic=ran;
	    			this.user_Region_Topic_Count[i][re.assigned_region][ran]++;
	    			this.topic_Count[ran]++;
	    			//pertopic_assignments_list.get(ran).add(re.time);
	    			 for(int word:re.poi.wordset)
	    			 {
	    				 this.topic_Word_Count[ran][word]++;
	    			 }
	    			 		
	    		}
	    	}
	    	
	    	
	    	//update Gaussian parameters simple version
	    	 for(int i=0;i<R;i++)
	    	 {
	    		    Location mu=new Location();
		    		double [] covariance=new double[2];
	    		 for(Location each:perregion_assignments_list.get(i))
	    		 {
	    			 
	    			 mu.latitude+=each.latitude;
	    			 mu.longitude+=each.longitude;
	    		 }
	    		 
	    		 mu.latitude=mu.latitude/perregion_assignments_list.get(i).size();
	    		 mu.longitude=mu.longitude/perregion_assignments_list.get(i).size();
	    		 
	    		 region_mu[i]=mu;
	    		 
	    	
	    		 
	    		 for(Location each:perregion_assignments_list.get(i))
	    		 {   
	    			 covariance[0]+=(each.latitude-mu.latitude)*(each.latitude-mu.latitude);
	    			 covariance[1]+=(each.longitude-mu.longitude)*(each.longitude-mu.longitude);
	    			 
	    		 }
	    		 
	    		 covariance[0]=covariance[0]/perregion_assignments_list.get(i).size();
	    		 covariance[1]=covariance[1]/perregion_assignments_list.get(i).size();
	    		 
	    		 this.region_covariance[i]=covariance;
	    	 }
	    	
	    	 
	    	//update Gaussian parameters non-simple version
	    	 for(int i=0;i<R;i++)
	    	 {
	    		    Location mu=new Location();
		    		//double [] covariance=new double[2];
	    		 for(Location each:perregion_assignments_list.get(i))
	    		 {
	    			 
	    			 mu.latitude+=each.latitude;
	    			 mu.longitude+=each.longitude;
	    		 }
	    		 
	    		 mu.latitude=mu.latitude/perregion_assignments_list.get(i).size();
	    		 mu.longitude=mu.longitude/perregion_assignments_list.get(i).size();
	    		 
	    		 region_mu[i]=mu;
	    		 
	    		 double[][] mu_temp=new double[2][1];
	    		 mu_temp[0][0]=mu.latitude;
	    		 mu_temp[1][0]=mu.longitude;
	    		 Matrix muM=new Matrix(mu_temp);
	    		 
	    		 
	    		 double[][] co_temp=new double[2][2];
	    	     Matrix covariance=new Matrix(co_temp);
	    		 
	    		 for(Location each:perregion_assignments_list.get(i))
	    		 {   
	    			 double[][] location=new double[2][1];
	    			 location[0][0]=each.latitude;
	    			 location[1][0]=each.longitude;
	    			 Matrix l=new Matrix(location);
	    			 
	    			 covariance= covariance.plus(l.minus(muM).times(l.minus(muM).transpose()));
	    			 
	    		 }
	    		 
	    		 covariance= covariance.times(1.0/perregion_assignments_list.get(i).size());
	    		 
	    		 this.region_covariance_M[i]=covariance;
	    	 }
	    	 
	    	
	    	
	    	 //update beta parameters
//	    	 for(int i=0;i<K;i++)
//	    	 {
//	    		 double mean_of_time=0;
//	    		 double variance=0;
//	    		 for(double time:pertopic_assignments_list.get(i))
//	    		 {
//	    			 mean_of_time+=time;
//	    		 }
//	    		 
//	    		 mean_of_time= mean_of_time/pertopic_assignments_list.get(i).size();
//	    		 for(double time:pertopic_assignments_list.get(i))
//	    		 {
//	    			 variance+=(time-mean_of_time)*(time-mean_of_time);
//	    		 }
//	    		 variance=variance/pertopic_assignments_list.get(i).size();
//	    		
//	    		
//	    		 this.topic_time_parameters[i][0]=mean_of_time*(mean_of_time*(1-mean_of_time)/variance -1);
//	    		 this.topic_time_parameters[i][1]=(1-mean_of_time)*(mean_of_time*(1-mean_of_time)/variance -1);
//	    		  
//	    	 }
	    	 
	    	} 
	    	
	    		
	    	
          
  	    public void gibbsSampling(int bigiter){
  	    	
  	    	for (int it = 1; it <= this.ITERATIONS; it++){
  	    		iter = it ;
  	    		int itout = it+ bigiter * ITERATIONS;
  	    		System.out.println("Iteration:"+itout);
  	        	for(int i=0; i<U; i++)
  	        	{
  	        		int length =usercheckinset.get(i).size();
  		    		
  		    		for(int j=0; j<length; j++)
  		    		{
  		    			int r=sample_region(i,j);
  		    			
  		    			CheckInRecord re=usercheckinset.get(i).get(j);
  		    			re.assigned_region=r;
  		    			int topic=sample_topic(i,j);
  		    			re.assigned_topic=topic;
  		    			
  		    		}
  		    		
  	        		}
  	        	
  	        	
  	        	for(int i=0;i<R;i++)
  	        	{
  	        		perregion_assignments_list.get(i).clear();
  	        	}
  	        	
  	        	
//  	        	for(int i=0;i<K;i++)
//  	        	{
//  	        		pertopic_assignments_list.get(i).clear();
//  	        	}
  	        	
  	        	
  	        	 // get statistics after burn-in    		
  	            if ((it >= BURN_IN) && (it % SAMPLE_LAG == 0))
  	            { 
  	                this.updateParameter();
  	               
  	                
  	            }
  	            
  	            System.out.println("iteration "+itout+" done");
  	           // testLambdaU();
  	        		
  	        	}
  	        
  	        	
  	    	}
  	    
  	  
  	  public void calDistribution(){	     
	        // userRegion
	        for(int i=0; i<U; i++){
	        	for(int j=0; j<R; j++){
	        		this.user_Region_Distribution[i][j] =
	        			this.user_Region_Distribution[i][j] / number_of_parameter_update;
	        	}
	        }
	       
	        // userRegionTopic
	        for(int i=0; i<U; i++){
	        	for(int j=0; j<R; j++)
	        		for(int k=0;k<K;k++)
	        	
	        	{
	        		this.user_Region_Topic_Distribution[i][j][k] =
	        				this.user_Region_Topic_Distribution[i][j][k] /number_of_parameter_update;

	        	}
	        }
	          
	        
	        //topic word
	        for(int i=0;i<K;i++)
	        	for(int j=0;j<W;j++)
	        	{
	        		this.topic_word_Distribution[i][j]=this.topic_word_Distribution[i][j]/number_of_parameter_update;
	        	}
	        
	        //region-POI
	     
	        for(int i=0;i<R;i++)
	        	for(int j=0;j<V;j++)
	        	{
	        		
	        		this.region_POI_Distribution[i][j]=this.region_POI_Distribution[i][j]/number_of_parameter_update;
	        	}
	        
	    
	      //Gaussian parameters
	        for(int i=0;i<R;i++)
	        {
	        	 this.region_mu[i].latitude=this.region_mu_sum[i].latitude/number_of_parameter_update;
	        	 this.region_mu[i].longitude=this.region_mu_sum[i].longitude/number_of_parameter_update;
	        	 
	        	 this.region_covariance_M[i]=this.region_covariance_sum_M[i].times(1.0/number_of_parameter_update);
	        	 
	        	 this.region_Count_sum[i]= this.region_Count_sum[i]/number_of_parameter_update;
	        	 
	        	for(int j=0;j<2;j++)
	        	{
	             this.region_covariance[i][j]=this.region_covariance_sum[i][j]/number_of_parameter_update;
	        	}
	       
	        }
	        
	        //Beta Parameters
//	        for(int i=0;i<K;i++)
//	        {
//	        	this.topic_Count_sum[i]=this.topic_Count_sum[i]/number_of_parameter_update;
//	        	for(int j=0;j<2;j++)
//	        	{
//	        		this.topic_time_parameters[i][j]=this.topic_time_parameters_sum[i][j]/number_of_parameter_update;
//	        	}
//	        	
//	        }
	       
	      }
	         
	    
	      
  	    
  	  public void updateParameter(){    
	        // userRegion
	        for(int i=0; i<U; i++){
	        	for(int j=0; j<R; j++){
	        		this.user_Region_Distribution[i][j]+=(this.user_Region_Count[i][j]+this.gamma)/(this.usercheckinset.get(i).size()+R*this.gamma);
	        		
	        	}
	        }
	       
	       
	        // userRegionTopic
	        for(int i=0; i<U; i++){
	        	for(int j=0; j<R; j++)
	        	  for(int k=0;k<K;k++)
	        	{
	        		this.user_Region_Topic_Distribution[i][j][k]+=(this.user_Region_Topic_Count[i][j][k]+this.alpha)/(this.user_Region_Number_of_Checkins[i][j]+K*this.alpha);
	        		
	        	}
	        }
	        
	       
	        // topicWord
	        for(int i=0; i<K; i++){
	        	for(int j=0; j<W; j++){
	        		this.topic_word_Distribution[i][j]+=(this.topic_Word_Count[i][j]+this.beta)/(this.topic_Count[i]+this.beta*W);
	        	}
	        }
	        
	        //regionPOI
	        for(int i=0; i<R; i++){
	        	 this.region_Count_sum[i]+=this.region_Count[i];
	        	for(int j=0; j<V; j++){
	        		this.region_POI_Distribution[i][j]+=(this.Region_POI_Count[i][j]+this.eta)/(this.region_Count[i]+this.eta*V);
	        	}
	        }
	        
	        //Gaussian parameters
	        for(int i=0;i<R;i++)
	        {
	        //System.out.printf("line 506: %f,",this.region_mu_sum[i].latitude );
	        //System.out.printf(" %f\n",this.region_mu[i].latitude );
	        	this.region_mu_sum[i].latitude+=this.region_mu[i].latitude;
	        	 this.region_mu_sum[i].longitude+=this.region_mu[i].longitude;
	        	 
	        	 this.region_covariance_sum_M[i]=this.region_covariance_sum_M[i].plus(this.region_covariance_M[i]);
	        	 
	        	for(int j=0;j<2;j++)
	        	{
	             this.region_covariance_sum[i][j]+=this.region_covariance[i][j];
	        	}
	       
	        }
	        
//	        //Beta Parameters
//	        for(int i=0;i<K;i++)
//	        {
//	        	this.topic_Count_sum[i]+=this.topic_Count[i];
//	        	for(int j=0;j<2;j++)
//	        	{
//	        		this.topic_time_parameters_sum[i][j]+=this.topic_time_parameters[i][j];
//	        	}
//	        	
//	        }
	        
	        number_of_parameter_update++;
	    }
  	    

	  public int sample_region(int user,int check_in)
	  {
		  int sampled_region=-1;
		  
		  CheckInRecord re=usercheckinset.get(user).get(check_in);
		  int region=re.assigned_region;
		  int topic=re.assigned_topic;
		  this.user_Region_Count[user][region]--;
		  this.region_Count[region]--;
		  this.Region_POI_Count[region][re.poi.poi_id]--;
		  
		  double probability[]=new double[R];
		  double temp_sum=0;
		  
		  for(int i=0;i<R;i++)
		  {
			 double first_term=this.user_Region_Count[user][i]+gamma;
			 double second_term=(this.user_Region_Topic_Count[user][i][topic]+alpha)/(this.user_Region_Count[user][i]+alpha*K);
			 double third_term=(this.Region_POI_Count[i][re.poi.poi_id]+eta)/(this.region_Count[i]+eta*V);
			 //System.out.printf("%d,%f,%f\n",i,re.poi.location.latitude,re.poi.location.longitude);
			 double fourth_term=GaussianProbability1(i,re.poi.location);
			 
			 probability[i]=first_term*second_term*third_term*fourth_term;
			 //System.out.printf("%f,%f,%f,%f\n", first_term,second_term,third_term,fourth_term);
			 
			 temp_sum  +=  probability[i];
		  }
		  
		  
	        double rand=Math.random()*temp_sum;
			
	        double tsum=0;
			for(int st=0;st<R;st++)
			{
				probability[st]=probability[st]+tsum;
				tsum=probability[st];
				
			}
	        
			
			for(int st=0;st<R;st++)
			{
				if(rand<=probability[st])
				{  
					sampled_region=st;
					
					break;
				}
				
			}
		  
			if(sampled_region<0)
			{
				System.out.println("sampling error in sample region");
				if (K>1) {
					System.out.printf("p0:%f, p1:%f, rand:%f, tmp_sum:%f\n",
							probability[0],probability[1],rand,temp_sum);
					System.out.printf("prob 0 : first:%f, second:%f, third:%f, fourth:%f\n",this.user_Region_Count[user][0]+gamma,
							(this.user_Region_Topic_Count[user][0][topic]+alpha)/(this.user_Region_Count[user][0]+alpha*K),
							(this.Region_POI_Count[0][re.poi.poi_id]+eta)/(this.region_Count[0]+eta*V),GaussianProbability1(0,re.poi.location));
					System.out.printf("prob 1 : first:%f, second:%f, third:%f, fourth:%f\n",this.user_Region_Count[user][1]+gamma,
							(this.user_Region_Topic_Count[user][1][topic]+alpha)/(this.user_Region_Count[user][1]+alpha*K),
							(this.Region_POI_Count[1][re.poi.poi_id]+eta)/(this.region_Count[1]+eta*V),GaussianProbability1(1,re.poi.location));
				
				}
				
				System.exit(0);
			}
			
			
			 this.user_Region_Count[user][sampled_region]++;
			  this.region_Count[sampled_region]++;
			  this.Region_POI_Count[sampled_region][re.poi.poi_id]++;
			
			  perregion_assignments_list.get(sampled_region).add(re.poi.location);
			return sampled_region;
		 
		  
	  }
	  
	  public int sample_topic(int user,int check_in)
	  {
		 
		  int sampled_topic=-1;
		  
		  CheckInRecord re=usercheckinset.get(user).get(check_in);
		  int region=re.assigned_region;
		  int topic=re.assigned_topic;
		  double time=re.time;
		  this.user_Region_Topic_Count[user][region][topic]--;
		  this.topic_Count[topic]--;
		  for(int each:re.poi.wordset)
		  {
			  
			  this.topic_Word_Count[topic][each]--;
		  }
		  
		 
		  double probability[]=new double[K];
		  double temp_sum=0;
		  
		  for(int i=0;i<K;i++)
		  {
			 double first_term=this.user_Region_Topic_Count[user][region][i]+alpha;
			 double second_term = 1;
			 //double second_term=Math.pow(1-time,this.topic_time_parameters[topic][0]-1)*Math.pow(time, this.topic_time_parameters[topic][1]-1)/BetaFunctionComputation(i,time);
			 //System.out.printf("%f,%f,%f,%f\n",time,this.topic_time_parameters[topic][0],this.topic_time_parameters[topic][1],BetaFunctionComputation(i,time));
			 double third_term=1;
			 for(int each:re.poi.wordset)
			 {
				 
				 third_term*=(this.topic_Word_Count[i][each]+beta)/(this.topic_Count[i]+W*beta);
			 }
			 
			 
			 probability[i]=first_term*second_term*third_term;
			 //System.out.printf("%f,%f,%f,%f\n", first_term,second_term,third_term,probability[i]);
			 temp_sum+=probability[i];
		  }
		  
		  
	        double rand=Math.random()*temp_sum;
			
	        double tsum=0;
			for(int st=0;st<K;st++)
			{
				probability[st]=probability[st]+tsum;
				tsum=probability[st];
				
			}
	        
			//System.out.printf("rand:%f,prob:", rand);
			for(int st=0;st<K;st++)
			{
				//System.out.printf("%f,",probability[st]);
				if(rand<=probability[st])
				{  
					sampled_topic=st;
					
					break;
				}
				
			}
			//System.out.printf("\n");
		  if(sampled_topic<0)
			{
				System.out.println("sampling error in sample topic");
				if (K>2)
					System.out.printf("p0:%f, p1:%f, p2:%f, rand:%f, tmp_sum:%f\n",
							probability[0],probability[1],probability[2],rand,temp_sum);
	
				System.exit(0);
			}
		  
		  this.user_Region_Topic_Count[user][region][sampled_topic]++;
		  this.topic_Count[sampled_topic]++;
		  for(int each:re.poi.wordset)
		  {
			  this.topic_Word_Count[sampled_topic][each]++;
		  }
		  
		
		  //pertopic_assignments_list.get(sampled_topic).add(time);
		  
		  
		return sampled_topic;  
	  }
	  
	  
	  
	  public double BetaFunctionComputation(int topic,double time)
	  {
		  
		 System.out.printf("beta function should not be called!\n");
		  
		 return ( 0);
		//return Math.exp(Beta.logBeta(this.topic_time_parameters[topic][0],this.topic_time_parameters[topic][1])); 
		//  (this.topic_time_parameters[topic][0],this.topic_time_parameters[topic][1]);
		 
		  
	  }
	  
	  //simple
	  public double GaussianProbability1(int region,Location l)
	  {
		  
		  double probability;
			
			if(this.region_covariance[region][0]==0&&this.region_covariance[region][1]==0)
			{
				return 1;
			}
			else if(this.region_covariance[region][0]==0||this.region_covariance[region][1]==0)
			{
				this.region_covariance[region][0]=(this.region_covariance[region][0]+this.region_covariance[region][1])/2;
				this.region_covariance[region][1]=this.region_covariance[region][0];
			}
			
			double normalization=1.0/(2*Math.sqrt(this.region_covariance[region][0])*Math.sqrt(this.region_covariance[region][1]));
			double X=(l.latitude-this.region_mu[region].latitude);
			double Y=(l.longitude-this.region_mu[region].longitude);
			
		   double body=	(X*X)/(-2*this.region_covariance[region][0])+(Y*Y)/(-2*this.region_covariance[region][1]);
		
			double main=Math.exp(body);
			
			probability=normalization*main;
			//System.out.printf("body:%f, norm:%f,covar1:%f, covar2:%f\n", body,normalization,this.region_covariance[region][0],this.region_covariance[region][1]);
			
			return probability;
	  }
	  
	  //accurate
	  public double GaussianProbability2(int region,Location l)
	  {
		  double[][] loc=new double[2][1];
		  loc[0][0]=l.latitude;
		  loc[1][0]=l.longitude;
		  Matrix l1=new Matrix(loc);
		  
		  double[][] mu=new double[2][1];
		  mu[0][0]=this.region_mu[region].latitude;
		  mu[0][1]=this.region_mu[region].longitude;
		  
		  Matrix mu1=new Matrix(mu);
		  
		  Matrix tem= l1.minus(mu1).transpose().times(this.region_covariance_M[region].inverse()).times(l1.minus(mu1));
		  double exp=Math.exp(tem.det()*(-0.5)); 
		  
		  double result=exp/(Math.sqrt(Math.abs(this.region_covariance_M[region].det())));
		   
		  return result;
	  }
	  
	  public void output_model(){
	    	System.out.println("output model ...");
	    	
	    	// parameter
	    	String parameter_file = outputPath + "hyper_parameter.txt";
	    	OutputStreamWriter oswpf = data_storage.file_handle(parameter_file);
	    	output_hyperparameter(oswpf);
	    	
	    	// matrix
	    	output_learntparameters(outputPath+"matrix/");
	    	
	    	System.out.println("output model ... done");
	    }
	  
	  public void output_learntparameters(String base_path){
	    	
		    try{
		        // userTopic
		    	String userRegion_file = base_path + "userRegion.txt";
		    	OutputStreamWriter oswpf = data_storage.file_handle(userRegion_file);
		    	oswpf.write(U+","+R+"\n");
		        for(int i=0; i<U; i++){
		        	for(int j=0; j<R; j++){
		        		if(j==R-1)
		        		{
		        		oswpf.write(this.user_Region_Distribution[i][j]+"");
		        		}
		        		else
		        		{	
		        		oswpf.write(this.user_Region_Distribution[i][j]+",");
		        		}
		        	}
		        	oswpf.write("\n");
		        }
				oswpf.flush();
				oswpf.close();
		       
		        // userRegionTopic
		    	String userRegionTopic_file = base_path + "userRegionTopic.txt";
		    	oswpf = data_storage.file_handle(userRegionTopic_file);
		    	oswpf.write(U+","+R+","+K+"\n");
		        for(int i=0; i<U; i++)
		        	for(int j=0; j<R; j++)
		        	{
		        		for(int k=0;k<K;k++)
		        	{
		        			if(k==K-1)
			        		{
			        		oswpf.write(this.user_Region_Topic_Distribution[i][j][k]+"");
			        		}
			        		else
			        		{	
			        		oswpf.write(this.user_Region_Topic_Distribution[i][j][k]+",");
			        		}
		        	}
		        	oswpf.write("\n");
		        	}
		        
				oswpf.flush();
				oswpf.close();
		       
		        // topicWord
		    	String topicWord_file = base_path + "topicWord.txt";
		    	oswpf = data_storage.file_handle(topicWord_file);
		    	oswpf.write(K+","+W+"\n");
		        for(int i=0; i<K; i++){
		        	for(int j=0; j<W; j++){
		        		if(j==W-1)
		        		{
		        		oswpf.write(this.topic_word_Distribution[i][j]+"");
		        		}
		        		else
		        		{	
		        		oswpf.write(this.topic_word_Distribution[i][j]+",");
		        		}
		        	}
		        	oswpf.write("\n");
		        }
				oswpf.flush();
				oswpf.close();
				
		        // regionPOI
		    	String regionPOI_file = base_path + "regionPOI.txt";
		    	oswpf = data_storage.file_handle(regionPOI_file);
		    	oswpf.write(R+","+V+"\n");
		        for(int i=0; i<R; i++){
		        	for(int j=0; j<V; j++){
		        		if(j==V-1)
		        		{
		        		oswpf.write(this.region_POI_Distribution[i][j]+"");
		        		}
		        		else
		        		{	
		        		oswpf.write(this.region_POI_Distribution[i][j]+",");
		        		}
		        	}
		        	oswpf.write("\n");
		        }
				oswpf.flush();
				oswpf.close();
				
				
				// mu
				String mu_file=base_path+"mu.txt";
				oswpf=data_storage.file_handle(mu_file);
				oswpf.write(R+"\n");
				for(int i=0;i<R;i++)
				{
					oswpf.write(this.region_mu[i].latitude+","+this.region_mu[i].longitude);
					oswpf.write("\n");
				}
				
				oswpf.flush();
				oswpf.close();
			
				//covariance
				
				String co_file=base_path+"Simplecovariance.txt";
				oswpf=data_storage.file_handle(co_file);
				oswpf.write(R+"\n");
				for(int i=0;i<R;i++)
				{
					oswpf.write(this.region_covariance[i][0]+","+this.region_covariance[i][1]);
					oswpf.write("\n");
				}
				
				oswpf.flush();
				oswpf.close();
				
				co_file=base_path+"covarianceMatrix.txt";
				oswpf=data_storage.file_handle(co_file);
				oswpf.write(R+"\n");
				for(int i=0;i<R;i++)
				{
					for(int j=0;j<2;j++)
						for(int k=0;k<2;k++)
					if(j==1&&k==1)
					{
						oswpf.write(this.region_covariance_M[i].get(j,k)+"");
					}
					else
					{
					oswpf.write(this.region_covariance_M[i].get(j,k)+",");
					}
					
					oswpf.write("\n");
				}
				
				oswpf.flush();
				oswpf.close();
			
				
				// beta	
				
				String beta_file=base_path+"beta.txt";
				oswpf=data_storage.file_handle(beta_file);
				oswpf.write(K+"\n");
//				for(int i=0;i<K;i++)
//				{
//						oswpf.write(this.topic_time_parameters[i][0]+","+this.topic_time_parameters[i][1]);
//						oswpf.write("\n");	
//				}
				
				oswpf.flush();
				oswpf.close();
				
				// region Popularity
				String Region_file = base_path + "regionPopularity.txt";
		    	oswpf = data_storage.file_handle(Region_file);
		    	oswpf.write(R+"\n");
		        	for(int j=0; j<R; j++){
		        		if(j==R-1)
		        		{
		        			oswpf.write(this.region_Count_sum[j]+"");
		        		}
		        		else
		        		{
		        		oswpf.write(this.region_Count_sum[j]+",");
		        		}
		        	
		        }
				oswpf.flush();
				oswpf.close();
				
				
				//topic Popularity
				String Topic_file = base_path + "topicPopularity.txt";
		    	oswpf = data_storage.file_handle(Topic_file);
		    	oswpf.write(K+"\n");
		        	for(int j=0; j<K; j++){
		        		if(j==K-1)
		        		{
		        			oswpf.write(this.topic_Count_sum[j]+"");
		        		}
		        		else
		        		{
		        		oswpf.write(this.topic_Count_sum[j]+",");
		        		}
		        	
		        }
				oswpf.flush();
				oswpf.close();
				
						
		    }
			catch(Exception e){
				e.printStackTrace();
			}
		    }
	  
	  
	  public void output_hyperparameter(OutputStreamWriter oswpf)
	  {
	    	try{
	    		oswpf.write("U: " + U + "\n");
	    		oswpf.write("V: " + V + "\n");
	    		oswpf.write("R: " + R + "\n");
	    		oswpf.write("K: " + K + "\n");
	    		oswpf.write("W: " + W + "\n");
	    		oswpf.write("alpha: " + this.alpha + "\n");
	    		oswpf.write("beta: " + this.beta + "\n");
	    		oswpf.write("gamma: " + this.gamma + "\n");
	    		oswpf.write("eta: " + this.eta + "\n");
	    		oswpf.write("ITERATIONS: " + ITERATIONS + "\n");
	    		oswpf.write("SAMPLE_LAG: " + SAMPLE_LAG + "\n");
	    		oswpf.write("BURN_IN: " + BURN_IN + "\n");
	    		oswpf.write("outputPath: " + outputPath + "\n");
				oswpf.flush();
				oswpf.close();
	    	}
			catch(Exception e){
				e.printStackTrace();
			}
	    }
	    	    
	  
	  public double predict(int u ,int i ) {
		  double a = 0;
		  
		  // region should be tour city
		  int region = poi_region_innitilization[i]; 
		  
		  for (int t = 0;t<K;t++) {
			  a += user_Region_Topic_Distribution[u][region][t] * poi_topic_score[i][t];
		  }
		  return a ;
	  }
	  
	  public double predict_region(int u, int i, int r ) {
		  double a = 0;		  
		  // region should be tour city
		  int region = poi_region_innitilization[i]; 
		  if (region == r)
			  for (int t = 0;t<K;t++) {
				  a += poi_topic_score[i][t];
			  }
		  else
			  a = 0;
		  return a ;
	  }
	  
	  public int get_region(int i ) {
		  int a = poi_region_innitilization[i];
		  return a;
	  }
	  
	  public double predict_currentlocal(int u ,int i ,Location l) {
		  double a = 0;
		  
		  // region should be tour city
		  int region = poi_region_innitilization[i]; 
		  
		  for (int t = 0;t<K;t++) {
			  a += user_Region_Topic_Distribution[u][region][t] * poi_topic_score[i][t];
		  }
		  a = a *user_Region_Distribution[u][region] *GaussianProbability1(region,l);
		  return a ;
	  }
	  
	  public void testformodel() {
		  refresh_poi_topic();
		  
		  
		  
	  }
	  
	
	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		// TODO Auto-generated method stub
		System.out.println("try find another main");

	}

}
