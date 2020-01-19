package main;

import java.io.IOException;

import algorithms.ItemPopularity;
import algorithms.MF_ALS_quickiter;


public class ALS_baseline extends main {
	public static void main(String argv[]) throws IOException {
		//String dataset_name = "yelp";
		String method = "als";
		double w0 = 0.5;
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 500;
		double reg = 0.1;
		double alpha = 0;
		double lr = 0.01; 
		boolean adaptive = false;
		String datafile = "data/testdata";
		int showbound = 20;
		int showcount = 10;
		boolean nativemode = false;

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
		}
	
		ReadRatings_GlobalSplit_notimestamp(datafile, 0.2,nativemode);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha);
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		popularity.evaluatefor82(testRatings,start);
		
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (method.equalsIgnoreCase("als")) {
			MF_ALS_quickiter als = new MF_ALS_quickiter(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount);
			als.buildModel();
			als.evaluatefor82(testRatings,start);
			
		}
		

	
	} // end main
}
