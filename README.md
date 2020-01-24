# Cross-city-MF

Codes for paper:  Ding, Jingtao, et al. "Learning from Hometown and Current City: Cross-city POI Recommendation via Interest Drift and Transfer Learning." Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 3.4 (2019): 1-28.



## Running example

$src/main/$ folder contains all cross-city models used in out paper. To run a certain model. first export corresponding main .java file into executable .jar file. Some examples are shown in below.

BPR baseline:

```
# crosscityMF_bpr_baseline.java
java bpr.jar 'none' 'none' 0.01 true false 32 1000 0.01 0.4 data/yelp/yelpdata 1 1 300 400 2 0.2
```

 ALS baseline:

```
# crosscityMF_als_baseline.java
java als.jar 'none' 'none' 0.01 true true 32 100 0.01 0.4 data/yelp/yelpdata 1 1 300 400
```

BPR_UIDT model:

```
# crosscityMF_bpr_uidt.java
java bpr_uidt.jar 'none' 'none' 0.01 true false 32 1000 0.01 0.4 data/yelp/yelpdata 1 1 300 400 16 0.1 10 
```

WMF_UIDT model:

```
# crosscityMF_wmf_uidt.java
java wmf_uidt.jar 'none' 'none' 0.01 true true 32 100 0.01 0.4 data/yelp/yelpdata 1 1 300 400 16 0.1 10 
```

STLDA model:

```
# main_social/stlda_yelp_social.java
java stlda.jar data/yelp/yelpdata data/yelp/yelp_poi_info_code 32 300 400 1 100 1 0.1 100 data/yelp/yelpregion100 data/yelp/social_info
```

BPR_UIDT with location model:

```
# crosscityMF_bpr_uidt_withlocation.java
java bpr_uidt_withlocation.jar 'none' 'none' 0.01 true false 32 1000 0.01 0.4 data/yelp/yelpdata 1 1 300 400 16 0.1 10 data/yelp/yelp_poi_info_code data/yelp/yelpregion100 100 
```

