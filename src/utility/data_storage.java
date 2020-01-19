package utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Hashtable;

import exception.length_check_exception;

public class data_storage {

	//create a streamwriter

	public static OutputStreamWriter file_handle(String name)

	{

		try{

			OutputStreamWriter osw=new OutputStreamWriter(new FileOutputStream(new File(name)));

			return osw;

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}

	//create a streamwriter for adding content

	public static OutputStreamWriter file_handle_add(String name)

	{

		try{

			OutputStreamWriter osw=new OutputStreamWriter(new FileOutputStream(new File(name),true));

			return osw;

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}

	//creat a stream writer for reading

	public static BufferedReader file_handle_read(String name)

	{

		try{

			BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(name)));

			return br;

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}

	public static int[][] load_matrix_int(BufferedReader br)

	{

		try{

			String line=br.readLine();

			String part[]=line.split(" ");

			int len1=Integer.parseInt(part[0]);

			int len2=Integer.parseInt(part[1]);

			int matrix[][]=new int[len1][len2];

			for(int i=0;i<len1;i++){

				line=br.readLine();

				part=line.split(" ");

				if(part.length!=len2){

					System.out.println(part.length);

					throw new length_check_exception();

				}

				for(int j=0;j<len2;j++){

					matrix[i][j]=Integer.parseInt(part[j]);

				}

			}

			System.out.println("Load matrix complete!");

			return matrix;

		}

		catch(length_check_exception l){

			l.printStackTrace();

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}

	public static double[][] load_matrix(BufferedReader br)

	{

		try{

			String line=br.readLine();

			String part[]=line.split(" ");

			int len1=Integer.parseInt(part[0]);

			int len2=Integer.parseInt(part[1]);

			double matrix[][]=new double[len1][len2];

			for(int i=0;i<len1;i++){

				line=br.readLine();

				part=line.split(" ");

				if(part.length!=len2){

					System.out.println(part.length);

					throw new length_check_exception();

				}

				for(int j=0;j<len2;j++){

					matrix[i][j]=Double.parseDouble(part[j]);

				}

			}

			System.out.println("Load matrix complete!");

			return matrix;

		}

		catch(length_check_exception l){

			l.printStackTrace();

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}

	

	//topics_given_documents_timeslices

	public static void store_matrix(OutputStreamWriter osw,double matrix[][][])

	{

		try{

			int len1=matrix.length;

			int len2=matrix[0].length;

			int len3=matrix[0][0].length;

			osw.write(len1+" "+len2+"\n");

			for(int j=0;j<len2;j++)

				for(int k=0;k<len3;k++)

			     {

				for(int i=0;i<len1;i++){

					osw.write(matrix[i][j][k]+" ");

				}

				osw.write("\n");

				osw.flush();

			}

			osw.close();

			System.out.println("Store matrix complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	

	

	

	

	public static void store_matrix2(OutputStreamWriter osw,double matrix[][])

	{

		try{

			int len1=matrix.length;

			int len2=matrix[0].length;

			osw.write(len1+" "+len2+"\n");

			for(int j=0;j<len2;j++){

				for(int i=0;i<len1;i++){

					osw.write(matrix[i][j]+" ");

				}

				osw.write("\n");

				osw.flush();

			}

			osw.close();

			System.out.println("Store matrix complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	

	public static void store_matrix(OutputStreamWriter osw,double matrix[][])

	{

		try{

			int len1=matrix.length;

			int len2=matrix[0].length;

			osw.write(len1+" "+len2+"\n");

			for(int i=0;i<len1;i++){

				for(int j=0;j<len2;j++){

					osw.write(matrix[i][j]+" ");

				}

				osw.write("\n");

				osw.flush();

			}

			osw.close();

			System.out.println("Store matrix complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	public static void store_matrix(OutputStreamWriter osw,int matrix[][])

	{

		try{

			int len1=matrix.length;

			int len2=matrix[0].length;

			osw.write(len1+" "+len2+"\n");

			for(int i=0;i<len1;i++){

				for(int j=0;j<len2;j++){

					osw.write(matrix[i][j]+" ");

				}

				osw.write("\n");

				osw.flush();

			}

			osw.close();

			System.out.println("Store matrix complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	public static void store_array(OutputStreamWriter osw,double array[])

	{

		try{

			int len1=array.length;

			osw.write(len1+"\n");

			for(int i=0;i<len1;i++)

			{

				osw.write(array[i]+" ");

			}

			osw.write("\n");

			osw.flush();

			System.out.println("Store array complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	public static void store_array2(OutputStreamWriter osw,double array[])

	{

		try{

			int len1=array.length;

			osw.write(len1+"\n");

			for(int i=0;i<len1;i++)

			{

				osw.write(array[i]+"\n");

			}

			osw.write("\n");

			osw.flush();

			System.out.println("Store array complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	

	public static void store_array(OutputStreamWriter osw,double para)

	{

		try{

		

			osw.write(para+"\n");

			osw.write("\n");

			osw.flush();

			System.out.println("Store array complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	public static void store_array2(OutputStreamWriter osw,double para)

	{

		try{

		

			osw.write(para+"\n");

			osw.write("\n");

			osw.flush();

			System.out.println("Store array complete!");

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	

	public static double load_single_parameter(BufferedReader br)

	{ 

	   	double para=0;

		try{

			String line=br.readLine();

			para=Double.parseDouble(line);

			

		}

		catch(Exception e)

		{

			e.printStackTrace();

		}

		return para;

	}

	

	public static double[] load_array(BufferedReader br)

	{

		try{

			String line=br.readLine();

			int len1=Integer.parseInt(line);

			double array[] = new double[len1];

			line=br.readLine();

			String part[]=line.split(" ");

			if(part.length!=len1){

				System.out.println(part.length);

				throw new length_check_exception();

			}

			for(int i=0;i<len1;i++){

				array[i]=Double.parseDouble(part[i]);

			}

			System.out.println("Load array complete!");

			return array;

		}

		catch(Exception e)

		{

			e.printStackTrace();

		}

		return null;

	}

	

	public static void store_map(OutputStreamWriter osw, int reverse_map[], int size)

	{

		try{

			int len=reverse_map.length;

			assert(size<=len);

			osw.write(size+"\n");

			for(int i=0;i<size;i++){

				osw.write(reverse_map[i]+" "+i+"\n");//the real_id and map_id in matrix

			}

			osw.flush();

			osw.close();

		}

		catch(Exception e){

			e.printStackTrace();

		}

	}

	public static Hashtable<Integer,Integer> load_map(BufferedReader br)

	{

		try{

			Hashtable <Integer,Integer>ht =new Hashtable<Integer,Integer>();

			String line=br.readLine();

			int number=Integer.parseInt(line);

			for(int i=0;i<number;i++){

				line=br.readLine();

				String part[]=line.split(" ");

				int keyword_id=Integer.parseInt(part[0]);

				int map_id=Integer.parseInt(part[1]);

				ht.put(keyword_id, map_id);

			}

			System.out.println("Load Map complete!");

			return ht;

		}

		catch(Exception e){

			e.printStackTrace();

		}

		return null;

	}
	
	public static void main(String args[])
	{
		System.out.println("Test");
	}

}

