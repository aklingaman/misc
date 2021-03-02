import java.util.*;
import java.io.*;
public class Driver {
	static String path = "/home/aklingam/misc";
	public static void main(String[] args) {
		System.out.println("Basic Neural Network made by Adam Klingaman");
		System.out.println("Designed for use on linux systems, may require some monkeying around with file extensions/paths to work on windows");     
		System.out.println("You have to edit the path string in the source code to match where the file is located.");
		help();
		System.out.println("for now, improper usage will dump excess tokens, or just crash");
		System.out.println("Sample configurations and their names are located in config directory");	
		System.out.println("Experiment is just a dummy function that can be edited to do whatever you want with it.");
		System.out.println("Also note that the way Serializable works, any changes to NN's source will invalidate any models you have stored.");	
		Scanner sc = new Scanner(System.in);
		String ans;
		while((ans=sc.nextLine())!=null) {
			String[] tokens = ans.split(" ");
			tokens[0] = tokens[0].toLowerCase();
			switch(tokens[0]) {
				case "create":     create(tokens);   break;
				case "train":      train(tokens);    break;
				case "test" :      test(tokens);     break;
				case "experiment": experiment();     break;
				case "help":       help();           break;
				case "metadata":   metadata(tokens); break;
				default:  System.out.println("Not an option");
			} 
		}		
	}

	//attempt to create a NN given the following name and configuration, returns success or fail. 
	public static void create(String[] tokens) {
		String name = tokens[1];
		String configuration = tokens[2];	
		NeuralNet model = IOHandler.createFromConfigFile(path+"/config/config.txt", configuration);
		IOHandler.writeToFile(model,path+"/models/"+name);
		System.out.println("Successfully created a model and serialized it under models");
	}	
	//Does training. TODO: pass things like bucket size as params.
	public static void train(String[] tokens){
		String nnPath = tokens[1];
		List<Image> trainingSet = IOHandler.collectImages(path+"/data/mnist_train.csv");
		if(trainingSet==null||trainingSet.size()==0) {
			System.out.println("Unable to get training data");
			return;
		}
		System.out.println("Succesfully managed to obtain training data with: "+trainingSet.size()+" data points");
		NeuralNet model = IOHandler.readFromFile(path+"/models/"+nnPath);
		if(model!=null) {
			System.out.println("Succesfully managed to obtain the model from file");
		} else {
			System.out.println("Unable to find file");
			System.exit(1);
		}
		System.out.println("Word of warning: file does not get saved until it is completely finished");

		//Both of these commented out chunks work as different forms of training. The first one will take a random subset of the training data of size {inner loop condition} and repeat {outer loop condition}  times. The second one iterates through the training data creating a sublist of size 50 and keeps passing that into the training function.
		int printFrequency = 500;  //Every PrintFrequency buckets, we will print it out, and the avg cost. 
		int epochCount = 5000;
		long startTime = System.currentTimeMillis();
		int trainingSetSize = trainingSet.size();	
		int bucketSize = 100;
		model.setLearnRate(0.01);
		for(int i = 0; i<epochCount; i++) { 
			ArrayList<Image> bucket = new ArrayList<Image>();
			int size = 0;
			while(size++<bucketSize) { 
				int random = (int)(Math.random()*trainingSetSize); 
				bucket.add(trainingSet.get(random));
			}
			double cost = model.train(bucket);
			if(i%printFrequency==0) {
				System.out.println("Training bucket num: " + i+" avg cost: " + cost);
			}
		}	 
		System.out.println("Training complete, total time: " + 1.0*(System.currentTimeMillis()-startTime)/1000 + " seconds.");
		/* 
		   int j = 0;
		   for(int i = 0; i+50<60000; i+=50) {
		   List<Image> bucket = trainingSet.subList(i,i+50);
		   model.train(bucket);
		   System.out.print("Training bucket num: " + (j++)); 
		   }
		   */
		IOHandler.writeToFile(model,path + "/models/" + nnPath);
	}
	public static void test(String[] tokens) {
		String nnPath = tokens[1];
		ArrayList<Image> testSet = IOHandler.collectImages(path+"/data/mnist_test.csv");
		if(testSet==null||testSet.size()==0) { 
			System.out.println("Error getting test data");
			return;
		}
		System.out.println("Succesfully obtained testing data");
		NeuralNet model = IOHandler.readFromFile(path+"/models/"+nnPath);
		if(model!=null) {
			System.out.println("Succesfully obtained model");
		} else {
			System.out.println("Unable to find file");
			System.exit(1);
		}
		//Begin testing procedure
		int count = 0; //10k test images.
		int correct = 0;
		int[] guesses = new int[10];
		int[] actual = new int[10];
		for(Image i : testSet) {
			//System.out.println(model.firstLayerWeights[0][0]);
			int prediction = LinAlg.vote(model.forwardProp(i.data));
			count++;
			guesses[prediction]++;
			actual[i.label]++;
			if(prediction==i.label) {
				correct++;
			} 
		}
		System.out.println("test results: ");
		System.out.println("Distribution of model guesses: ");
		for(int i = 0; i<10; i++) {
			System.out.print(i+": " + guesses[i] + ", ");
		}
		System.out.println();
		System.out.println("Distribution of actual images: ");
		for(int i = 0; i<10; i++) {
			System.out.print(i+": " + actual[i] + ", ");
		}
		System.out.println();
		System.out.println("Model classified " + correct + " correctly out of " + count +  " testing records");
		System.out.println(100.0*correct/count+"% accuracy.");
	}
	//Dummy function used for messing around with stuff. Used the word experiment to distinguish from test. 	
	public static void experiment() {
		System.out.println("Current experiment: making a bunch of identical NN's except with different learning rates to see which ones can converge to a solution");
		double[] learnRates = {0.0001,0.0005, 0.001, 0.005, 0.01,0.05,0.1,0.5};
		List<Image> trainingSet = IOHandler.collectImages(path+"/data/mnist_train.csv");
		for(int i = 0; i<learnRates.length; i++) {
			NeuralNet model = IOHandler.createFromConfigFile(path+"/config/config.txt", "mnist2by20");
			for(int j = 0; j<1; j++) {
				for(int k = 0; k<10000; k++) { 
					ArrayList<Image> bucket = new ArrayList<Image>();
					int size = 0;
					while(size<100) { //Change the 100 to change how big the buckets are.
						int random = (int)(Math.random()*60000); //60000 is the number of elements in the training set. 
						bucket.add(trainingSet.get(random));
						size++;
					}
					model.setLearnRate(learnRates[i]);	
					double cost = model.train(bucket);
				}
				ArrayList<Image> testSet = IOHandler.collectImages(path+"/data/mnist_test.csv");
				int count = testSet.size();
				int correct = 0;
				for(Image l : testSet) {
					int prediction = LinAlg.vote(model.forwardProp(l.data));
					if(prediction == l.label) {
						correct++;
					}
				}
				double accuracy = 1.0*correct/count;
				System.out.println("learnRate: " + learnRates[i] + " run " + j + "/5" + ", Accuracy: " + accuracy);			    
			}
		}
	}
	public static void help() {
		System.out.printf("Usage:\n\tcreate <filename> <configuration>\n\ttrain <filename>\n\ttest <filename>\n\texit\n\texperiment\n\tmetadata <filename>\n");
	}
	//prints out an overview of a NN's characteristic meta params. Does this by actually creating the NN, so its expensive, so dont spam it. 
	public static void metadata (String[] tokens) {
		System.out.println(IOHandler.readFromFile(path+"/models/"+tokens[1]).metadata());
	}


}
