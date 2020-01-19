package main_multicity;

import java.io.File;
import java.io.IOException;

import algorithms.ItemPopularity;
import algorithms.MF_bpr_multi;


public class bpr_multi extends main.main_multicity {
	public static void main(String argv[]) throws IOException {
		String method = "none";
		double lr = 0.01;
		
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 500;
		double reg = 0.1;
		double alpha = 0;
		String datafile = "data/yelp_smalltest";
		int showbound = 100;
		int showcount = 2;
		double bigalpha=0.0;
		double bigbeta=0;
		int sharefactor = 0;
		int mode = 3;
		int citynum = 2;
		if (argv.length > 0) {
			lr = Double.parseDouble(argv[2]);
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
		sharefactor = factors;
				
		ReadRatings_GlobalSplit_multicity(datafile, 0.2,citynum);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, lr=%.6f, alpha=%.4f, mode=%d\n",
				method, showProgress, factors, maxIter, reg, lr, alpha,mode);
		System.out.println("this is bpr baseline singlebpr_1010");
		System.out.printf("new para:  bigalpha=%f bigbeta=%f sharefactor = %d\n",bigalpha,bigbeta,sharefactor);
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		popularity.buildModel();
		System.out.printf("popularity model: ");
		popularity.evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (true) {
			MF_bpr_multi als = new MF_bpr_multi(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, lr, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount,citynum);
			als.setintarray(city_users, city_pois, user_city, poi_city ) ;
			als.setintpara(new int [] {sharefactor});
			als.initialize();
			als.buildmulticityModel(mode);
			als.evaluatefor82multicity(testRatings,start, city_pois, user_city, poi_city);

			String datafile_write = datafile.replace('/', '_');
			String outpath = "Kout/multi_bpr_"+datafile_write+"_lr_"+lr+"_reg_"+reg+
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
