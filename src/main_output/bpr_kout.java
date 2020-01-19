package main_output;

import java.io.File;
import java.io.IOException;

import algorithms.ItemPopularity;
import algorithms.MF_bpr_single;


public class bpr_kout extends main.main_crosscity {
	public static void main(String argv[]) throws IOException {
		String method = "none";
		double lr = 0.01;
		
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 10;
		double reg = 0.1;
		double alpha = 0;
		String datafile = "data/cross_part";
		int showbound = 100;
		int showcount = 2;
		int nativecity = 1100;
		int tourcity = 5401;
		double bigalpha=0.0;
		double bigbeta=0;
		int sharefactor = 0;
		int mode = 3;		
		if (argv.length > 0) {
			//dataset_name = argv[0];
			//method = argv[1];
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
			nativecity = Integer.parseInt(argv[12]);
			tourcity = Integer.parseInt(argv[13]);
			mode = Integer.parseInt(argv[14]);	
		}
		sharefactor = factors;
		ReadRatings_GlobalSplit_notimestamp(datafile, 0.2,nativecity,tourcity);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, lr=%.6f, alpha=%.4f, mode=%d\n",
				method, showProgress, factors, maxIter, reg, lr, alpha,mode);
		System.out.println("this is bpr baseline singlebpr_1010");
		System.out.printf("new para:  bigalpha=%f bigbeta=%f sharefactor = %d\n",bigalpha,bigbeta,sharefactor);
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		popularity.buildModel();
		popularity.evaluatefor82crosscity(testRatings,start,nativeUsers,tourPois);
		popularity.evaluatefor82crosscity_showFactnum(testRatings,nativeUsers,tourPois);
		double init_mean = 0;
		double init_stdev = 0.01;

		if (true) {
			MF_bpr_single als = new MF_bpr_single(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, lr, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount);
			als.sethashset(nativeUsers,tourPois);
			als.setbigw(new double [] {bigalpha,bigbeta});
			als.setintpara(new int [] {sharefactor});
			als.initialize();
			als.buildcrosscityModel(mode);
			als.evaluatefor82crosscity(testRatings,start,nativeUsers,tourPois);

			String datafile_write = datafile.replace('/', '_');
			String outpath = "Kout/bpr_"+datafile_write+"_w0_"+lr+"_reg_"+reg+
					"_K_"+factors+"_city"+nativecity;
			File outfile = new File("Kout/");
			if (!outfile.exists()) {
				outfile.mkdir();
			}	
			als.evaluatefororder_output(testRatings,start,nativeUsers,tourPois,outpath);
			System.out.printf("finish write in file: %s \n",outpath);
		}
	} // end main
}
