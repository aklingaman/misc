import java.util.*;
import java.io.*;
public class Driver {
    public static void main(String[] args) {
        String path = "/home/aklingam/misc/cs484proj";
        System.out.println("Basic Neural Network made by Adam Klingaman");
        System.out.println("Designed for use on linux systems, may require some monkeying around with file extensions/paths to work on windows");
        System.out.println("UI is not dummy proofed much, if you enter a bad file you get a java error");
        
        System.out.println("For now you also have to edit the path string in the source code to match where the file is located. ");
        System.out.println("Usage: type create or train or test followed by filename with a .dat extension");
        System.out.println("Multiple options in the same run not allowed. Rerun required.");
        Scanner sc = new Scanner(System.in);
        String ans = sc.nextLine();
        String[] tokens = ans.split(" ");
        tokens[0] = tokens[0].toLowerCase();
        if(tokens[0].equals("create")) {
            //CREATE
			NeuralNet model = IOHandler.createFromConfigFile(path+"/config/config.txt", tokens[2]);
            //NeuralNet model = new NeuralNet(784,20,10,true);
            IOHandler.writeToFile(model,path+"/models/"+tokens[1]);
            System.out.println("Successfully created a model and serialized it under models");
            System.exit(0);
            //CREATE
        } else if(tokens[0].equals("train")) {
            //TRAIN
            List<Image> trainingSet = IOHandler.collectImages(path+"/data/mnist_train.csv");
            System.out.println("Succesfully managed to obtain training data with: "+trainingSet.size()+" data points");
            NeuralNet model = IOHandler.readFromFile(path+"/models/"+tokens[1]);
            if(model!=null) {
                System.out.println("Succesfully managed to obtain the model from file");
            } else {
                System.out.println("Unable to find file");
                System.exit(1);
            }
            System.out.println("Word of warning: file does not get saved until it is completely finished");
            //Begin training procedure  
            
            //Both of these commented out chunks work as different forms of training. The first one will take a random subset of the training data of size {inner loop condition} and repeat {outer loop condition}  times. The second one iterates through the training data creating a sublist of size 50 and keeps passing that into the training function.
            int printFrequency = 100;  //Every PrintFrequency buckets, we will print it out, and the avg cost. 
	    	long startTime = System.currentTimeMillis();
			for(int i = 0; i<25000; i++) { //Edit the for loop termination to change how many epochs we will have.
                ArrayList<Image> bucket = new ArrayList<Image>();
                while(bucket.size()<100) { //Change the 100 to change how big the buckets are.
                    int random = (int)(Math.random()*60000); //60000 is the number of elements in the training set. 
                    bucket.add(trainingSet.get(random));
                }
				double cost = model.train(bucket,0.5);
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


       
            IOHandler.writeToFile(model,path + "/models/" + tokens[1]);
            System.exit(0);
            //TRAIN
        }else if(tokens[0].equals("test")) {
            //TEST
            ArrayList<Image> testSet = IOHandler.collectImages(path+"/data/mnist_test.csv");
            System.out.println("Succesfully managed to obtain testing data");
            NeuralNet model = IOHandler.readFromFile(path+"/models/"+tokens[1]);
            if(model!=null) {
                System.out.println("Succesfully managed to obtain the model from file");
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
                int prediction = model.vote(model.predict(i.data));
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
            System.exit(0);
            //TEST
        } else {
            System.out.println("Refer to usage");
            System.exit(1);
        }
    

        

    

    }   
}
