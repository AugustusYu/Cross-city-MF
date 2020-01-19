package main_social;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import algorithms.STLDA_region_social;
import data_structure.Rating;
import data_structure.SparseMatrix;

public class stlda_yelp_social extends main.main_crosscity{
	public static void main(String argv[]) throws IOException {
		String check_in_file = "data/yelp/yelpdata";
		String datafile = check_in_file;
		String catefile = "data/yelp/yelp_poi_info_code";
		int nativecity = 300;
		int tourcity = 400;
		
		int K = 3;  // topics
		int R = 10;  // regions
		double alpha = 50/(double)K;
		double beta = 0.01;
		double eta = 0.01;
		double gamma = 50/(double)R;
		
		int U = 61413;// users
		int V = 42304; // pois
		int W = 22576; // words
		
		String path_model_output = null;
		int ITERATIONS = 5;
		int SAMPLE_LAG = 1;
		int BURN_IN = 0;
		int maxoutput = 1;
		String KMeansFile = "data/yelp/yelpregionkmeans10";
		String socialFile = "data/yelp/social_info";
		if (argv.length>0) {
			datafile = argv[0];
			check_in_file = datafile;
			catefile = argv[1];
			K = Integer.parseInt(argv[2]);
			nativecity = Integer.parseInt(argv[3]);
			tourcity = Integer.parseInt(argv[4]);
			maxoutput = Integer.parseInt(argv[5]);
			ITERATIONS = Integer.parseInt(argv[6]);
			SAMPLE_LAG = Integer.parseInt(argv[7]);
			BURN_IN = Integer.parseInt(argv[8]);		
			R = Integer.parseInt(argv[9]);
			KMeansFile = argv[10];		
			socialFile = argv[11];	
			alpha = 50/(double)K;
			gamma = 50/(double)R;
		}
		
		System.out.printf("STLDA: K:%d, iter:%d, maxiter:%d, sample:%d, burnin:%d, R:%d\n", K,
				ITERATIONS,ITERATIONS*maxoutput,SAMPLE_LAG,BURN_IN,R);
		System.out.printf("alpha:%f, beta:%f, eta:%f, gamma:%f\n",alpha,beta,eta,gamma);
		ReadRatings_GlobalSplit_notimestamp(datafile, 0.2,nativecity,tourcity);
		Long start = System.currentTimeMillis();	
		System.out.println("this is stlda model!");

		STLDA_region_social mylda = new STLDA_region_social(trainMatrix,  testRatings, 
				topK, threadNum,K, R,   alpha, beta, eta,
				 gamma,ITERATIONS, SAMPLE_LAG, BURN_IN,catefile,socialFile);
		
		mylda.refreshparas();
		mylda.sethashset(nativeUsers, tourPois); 
		mylda.data_reshape();
		
		mylda.buildcrosscitymodel(maxoutput,KMeansFile);		
		
		String datafile_write = datafile.replace('/', '_');
		String outpath = "Kout/social_stlda_"+datafile_write+"_K_"+R+"_city"+nativecity;
		File outfile = new File("Kout/");
		if (!outfile.exists()) {
			outfile.mkdir();
		}	
		mylda.evaluatefororder_output_stlda(testRatings,start,nativeUsers,tourPois,outpath);
		System.out.printf("finish write in file: %s \n",outpath);
		
	}
	
}
