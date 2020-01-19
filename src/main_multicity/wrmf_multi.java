package main_multicity;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;

import algorithms.ItemPopularity;
import algorithms.MF_ALS_quickiter_multi;

public class wrmf_multi extends main.main_multicity {
	public static void main(String argv[]) throws IOException {
		//String dataset_name = "yelp";
		String method = "als";
		double w0 = 0.01;
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 100;
		double reg = 0.1;
		double alpha = 0;
		double lr = 0.01; 
		boolean adaptive = false;
		String datafile = "data/yelp_smalltest";
		int showbound = 0;
		int showcount = 1;
		int citynum = 3;
		
		if (argv.length > 0) {
			//dataset_name = argv[0];
			//method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			datafile = argv[9];
			showbound = Integer.parseInt(argv[10]);
			showcount = Integer.parseInt(argv[11]);
			citynum = Integer.parseInt(argv[12]);	
		}
	
		ReadRatings_GlobalSplit_multicity(datafile, 0.2,citynum);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha);
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		popularity.buildModel();
		System.out.printf("popularity model: ");
		popularity.evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
		
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (method.equalsIgnoreCase("als")) {
			MF_ALS_quickiter_multi als = new MF_ALS_quickiter_multi(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount
					,citynum);
			als.setintarray(city_users, city_pois, user_city, poi_city ) ;
			
			als.buildmulticityModel();
			als.evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
			
			String datafile_write = datafile.replace('/', '_');
			String outpath = "Kout/multi_als_"+datafile_write+"_w0_"+w0+"_reg_"+reg+
					"_K_"+factors;
			File outfile = new File("Kout/");
			if (!outfile.exists()) {
				outfile.mkdir();
			}	
			als.evaluatefororder_output_multi(testRatings,start, city_pois, user_city,outpath, poi_city);
			System.out.printf("finish write in file: %s \n",outpath);
			
		}
	} // end main
}
