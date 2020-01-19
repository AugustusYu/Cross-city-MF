package data_structure;

public class Crossrating {
	public int uid;
	public int pid;
	public int count;
	public int ucity;
	public int pcity;
	public boolean nativemode;
	
	public Crossrating(String line) {
		String[] arr = line.split("\t");
		uid = Integer.parseInt(arr[0]);
		pid = Integer.parseInt(arr[1]);
		count = Integer.parseInt(arr[2]);
		ucity = Integer.parseInt(arr[3]);
		pcity = Integer.parseInt(arr[4]);
		if (ucity == pcity)
			nativemode = true;
		else
			nativemode = false;
	}
	
	public Crossrating() {
		uid = -1;
		pid = -1;
		count = -1;
		ucity = -1;
		pcity = -1;
		nativemode = false;
	}
	
}
