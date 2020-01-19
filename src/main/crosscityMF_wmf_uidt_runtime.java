package main;

import java.io.IOException;
import java.util.HashSet;

import algorithms.ItemPopularity;
import algorithms.MF_ALS_uidt;


public class crosscityMF_wmf_uidt_runtime extends main_crosscity {
	public static void main(String argv[]) throws IOException {
		String method = "none";
		double w0 = 0.1;
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 500;
		double reg = 0.01;
		double alpha = 0;
		double lr = 0.001; 
		boolean adaptive = false;
		String datafile = "data/cross_part";
		int showbound = 100;
		int showcount = 10;
		int nativecity = 1100;
		int tourcity = 5401;
		double bigalpha=0.8;
		double bigbeta=100;
		int sharefactor = 5;
		double data_ratio = 0.1;
			
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
			nativecity = Integer.parseInt(argv[12]);
			tourcity = Integer.parseInt(argv[13]);
			sharefactor = Integer.parseInt(argv[14]);
			bigalpha = Double.parseDouble(argv[15]);
			bigbeta = Double.parseDouble(argv[16]);
			data_ratio = Double.parseDouble(argv[17]);
		}
	
		ReadRatings_GlobalSplit_notimestamp(datafile, 0.2,nativecity,tourcity);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha);
		System.out.println("this algorithm test runtime for wmf-uidt");
		System.out.printf("K:%d,ratio:%f\n",factors,data_ratio);
		
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		double init_mean = 0;
		double init_stdev = 0.01;
			
		if (true) {
			MF_ALS_uidt als = new MF_ALS_uidt(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount);
			als.sethashset(nativeUsers,tourPois);
			als.setbigw(new double [] {bigalpha,bigbeta});
			als.setintpara(new int [] {sharefactor});
			als.initialize();
			als.buildcrosscityModel_runtime_test(data_ratio);
		}
	} // end main
}
