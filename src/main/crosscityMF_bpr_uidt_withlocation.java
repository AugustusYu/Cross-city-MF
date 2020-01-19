package main;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;

import algorithms.ItemPopularity;
import algorithms.MF_bpr_uidt_withlocation;


public class crosscityMF_bpr_uidt_withlocation extends main_crosscity {
	public static void main(String argv[]) throws IOException {
		String method = "none";
		double w0 = 0.001;
		boolean showProgress = true;
		boolean showLoss = true;
		int factors = 10;
		int maxIter = 10;
		double reg = 0.1;
		double alpha = 0;
		double lr = 0.001; 
		boolean adaptive = false;
		String datafile = "data/yelp/yelpdata";
		int showbound = 100;
		int showcount = 10;
		int nativecity = 300;
		int tourcity = 400;
		double bigalpha=0.8;
		double bigbeta=100;
		int sharefactor = 5;
		double location_threhold = 1;
		String catefile = "data/yelp/yelp_poi_info_code";
		String kmeansfile = "data/yelp/yelpregion2";
		int R = 1;	
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
			catefile = argv[17];
			kmeansfile = argv[18];
			R = Integer.parseInt(argv[19]);
		}
	
		ReadRatings_GlobalSplit_notimestamp(datafile, 0.2,nativecity,tourcity);
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%.6f, w0=%.6f, alpha=%.4f,region=%d\n",
				method, showProgress, factors, maxIter, reg, w0, alpha,R);
		System.out.println("extra location was used in this algo");
		System.out.printf("new para:  bigalpha=%f bigbeta=%f sharefactor = %d\n",bigalpha,bigbeta,sharefactor);
		System.out.println("====================================================");
		Long start = System.currentTimeMillis();
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		popularity.evaluatefor82crosscity(testRatings,start,nativeUsers,tourPois);
		popularity.evaluatefor82crosscity_showFactnum(testRatings,nativeUsers,tourPois);
		double init_mean = 0;
		double init_stdev = 0.01;

		if (true) {
			algorithms.MF_bpr_uidt_withlocation als = new algorithms.MF_bpr_uidt_withlocation(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss,showbound,showcount,catefile);
			als.sethashset(nativeUsers,tourPois);
			als.setbigw(new double [] {bigalpha,bigbeta,location_threhold});
			als.setintpara(new int [] {sharefactor});
			als.initialize();
			als.buildcrosscityModel();
			als.evaluatefor82crosscity(testRatings,start,nativeUsers,tourPois);
			String datafile_write = datafile.replace('/', '_');
			String outpath = "orderoutputfile/bprmix_stlda_"+datafile_write+"_native_"+nativecity+"_w0_"+w0+"_reg_"+reg+
					"_K_"+factors+"_sh_"+sharefactor+"_a_"+bigalpha+"_b_"+bigbeta+"_region_"+R;
			File outfile = new File("orderoutputfile/");
			if (!outfile.exists()) {
				outfile.mkdir();
			}	
			als.refresh_stlda_apras(kmeansfile,R);
			als.evaluatefor82crosscity_stlda_region_output(testRatings,start,nativeUsers,tourPois,outpath);
			System.out.printf("finish write in file: %s \n",outpath);
		}
	} // end main
}
