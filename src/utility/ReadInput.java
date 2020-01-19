package utility;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import algorithms.CheckInRecord;


public class ReadInput 
{
	
	public static HashMap<Integer,POI> POIset=new HashMap<Integer,POI>();
	
	public static void initializeModel(String Kmeansfile, HashMap<Integer,ArrayList<Location>> perregion_assignments_list,int[] poi_region_inni)
	{
		
	}
	
	
	public static void load_checkin_data(String checkinfile,HashMap<Integer,ArrayList<CheckInRecord>> usercheckinset)
	{
		
     //format: User_id, POI_id, latitude, longitude, time, contents
	   
		BufferedReader reader;
		try{
			int count = 0;
			reader=new BufferedReader(new InputStreamReader(new FileInputStream(checkinfile),"UTF-8"));
			String line=reader.readLine();
			while(line!=null)
			{
			 String[] tokens=line.split("\t");
			 int user_id=Integer.parseInt(tokens[0]);
			 int poi_id=Integer.parseInt(tokens[1]);
			 String oritime = tokens[4]; 
			 String newString = oritime.replaceAll("[^0-9]", "");
			 //double time=Double.parseDouble(tokens[4]);
			 double time = Double.parseDouble(newString);
			 time = (int)( time % 1000000);
			 int h = (int ) time/10000;
			 int m =(int ) time/100%100;
			 int s = (int) time%100;

			 time = (double)(h*3600 + m*60 + s)/86400;
//			 if (count == 0) {
//				 System.out.printf("%d,%d,%d,%f\n", h,m,s,time);
//				 count = 1;
//			 }
			 if(POIset.containsKey(poi_id))
			 {
				if( usercheckinset.containsKey(user_id))
				{
					CheckInRecord record=new CheckInRecord ();
					record.user_id=user_id;
					record.time=time;
					record.poi=POIset.get(poi_id);
					usercheckinset.get(user_id).add(record);
					
				}
				else
				{
					ArrayList<CheckInRecord> list=new ArrayList<CheckInRecord>();
					CheckInRecord record=new CheckInRecord ();
					record.user_id=user_id;
					record.time=time;
					record.poi=POIset.get(poi_id);
					list.add(record);
					usercheckinset.put(user_id, list);
					
				}
				 
			 }
			 else
			 {
			  double latitude=Double.parseDouble(tokens[2]);
			  double longtitude=Double.parseDouble(tokens[3]);
			 String[] words=tokens[5].split("#");
			 Location l=new Location();
			 l.latitude=latitude;
			 l.longitude=longtitude;
			 ArrayList<Integer> list=new ArrayList<Integer>();
			 for(String word:words)
			 {
				 list.add(Integer.parseInt(word));
				 
			 }
			 POI p=new POI();
			 p.poi_id=poi_id;
			 p.location=l;
			 p.wordset=list;
			 POIset.put(poi_id,p); 
			 
			 
			 if( usercheckinset.containsKey(user_id))
				{
					CheckInRecord record=new CheckInRecord ();
					record.user_id=user_id;
					record.time=time;
					record.poi=p;
					usercheckinset.get(user_id).add(record);
					
				}
				else
				{
					ArrayList<CheckInRecord> lists=new ArrayList<CheckInRecord>();
					CheckInRecord record=new CheckInRecord ();
					record.user_id=user_id;
					record.time=time;
					record.poi=p;
					lists.add(record);
					usercheckinset.put(user_id, lists);
					
				}
			 
			 
			 }
			 
			 line=reader.readLine();
				
			}
			
			reader.close();
			}
		catch(IOException e)
		{
			e.printStackTrace();
		}
		
		
	
		
		
	}
	
	
	
	public static void loadfriend(String networkfile,HashMap<Integer,HashMap<Integer,Double>> network)
	{   
		
		BufferedReader reader;
		try{
			
			reader=new BufferedReader(new InputStreamReader(new FileInputStream(networkfile),"UTF-8"));
			String line=reader.readLine();
			while(line!=null)
			{
			 String[] tokens=line.split("\t");
			 int start_user=Integer.parseInt(tokens[0]);
			 int end_user=Integer.parseInt(tokens[1]);
			 double weight=Double.parseDouble(tokens[2]);
			 
			 if(network.containsKey(start_user))
			 {
			   network.get(start_user).put(end_user, weight);	 
			 }
			 else
			 {
				 
				 HashMap<Integer,Double> friendlist=new HashMap<Integer,Double>();
				 friendlist.put(end_user,weight);
				 network.put(start_user, friendlist);
				 
			 }
			 
			 line=reader.readLine();
			}
			
			reader.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}	
	}
	
	private  void ComputeHomeLocation()
	{
		
		
		System.out.println("Home Location Inference Finished.");
	}
	
	
		
	

}
