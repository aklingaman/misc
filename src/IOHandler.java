//This class handles file IO.
import java.util.*;
import java.io.*;
public class IOHandler {
    
    public static void writeToFile(NeuralNet model, String path) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));    
            out.writeObject(model);
            out.flush();
            return;
        }catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return;
    }
	//Takes in a configuration file that outlines the NN structure, and returns an NN using that structure. 	
	//Format: name: abc, inputsize: 784, hlquantity: 2, hlsize: 20, outputsize: 10
	//Currently we have 2 hidden layers hard coded, so the hlquantity field gets ignored.
	//TODO: handle the learnrate from file, and add a createfromTerminal function to allow realtime NN creation.  
	public static NeuralNet createFromConfigFile(String path, String inputName) {
		ArrayList<String> configurations = new ArrayList<String>();
		try {	
            BufferedReader br = new BufferedReader(new FileReader(path));
			String line, name;
			int inputsize = -1; //Because im lazy, im just making sure with my sanity checks in NN that this fails if they arent updated by the switch;
			int hlquantity = -1;
			int hlsize = -1;
			int outputsize = -1;
			while((line = br.readLine()) !=null) {
				String[] tokens = line.split(", ");
				if(tokens.length!=5) {
					continue;
				}
				if(!tokens[0].split(": ")[1].equals(inputName)) {
					continue; 
				}
				for(int i = 1; i<tokens.length; i++) {
					String[] chunks = tokens[i].split(": ");
					if(chunks.length!=2) {
						System.out.println("error in config reading, wrong num: "+ chunks.length);
						continue;
					}
					int val = Integer.parseInt(chunks[1]);
					switch(i) {
						case 1: inputsize  = val; break;
						case 2: hlquantity = val; break;
						case 3: hlsize     = val; break;
						case 4: outputsize = val; break;
						
					}	
				}
				//Currently we only support 2 hidden layers, that is much more complicated to fix, so yeah...
				NeuralNet ret = new NeuralNet(inputsize, hlsize, hlquantity, outputsize, 0.05);
//				System.out.println("Found the configuration");
				return ret;
			}
//			System.out.println("Unable to find this configuration");
			System.exit(1);
			return null;//darn you java, making me return after an exit call.	
		}catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}
	//reads in an already existing NN using serializable.
    public static NeuralNet readFromFile(String path) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
            NeuralNet model = (NeuralNet) in.readObject();
            in.close();
            return model;
        }catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
       return null; 
    }   
	public static ArrayList<Image> collectImages(String path) {
        ArrayList<Image> dataSet = new ArrayList<Image>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;
            while((line = br.readLine())!=null) {
                String[] vals = line.split(",");
                double[] pixels = new double[784];
                for(int i = 0; i<784; i++ ) {
                    pixels[i] = 1.0*Integer.parseInt(vals[i+1])/255;
                } 
                dataSet.add(new Image(Integer.parseInt(vals[0]),pixels)); 
            }
            return dataSet;
        }catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return null; //Needed for compilation only
    }
    
}
