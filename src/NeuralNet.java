//We can imagine the neural net as simply being a process that takes us from layer to layer. All each layer "is" is a list of decimal values corresponding to weights, and another vector for the biases. 
//Its entirely possible that this is too much contiguous memory for fast operation. It may become pertinent to switch to an object for the weights. I have no idea how java lays out its objects, and whether 2 arrays will be consecutive in memory. Typically i would think they are not contiguous, but with implementing serializable, this gets called into question.   

/*
    this nn is flexible on: Hidden layer size.
    ""      ""   strict on: Number of hidden layers ( its always 2 ) 


*/
import java.util.*;
import java.io.*;

public class NeuralNet implements Serializable{
    public double[] firstLayerBias;
    public double[] secondLayerBias;
    public double[] outputLayerBias;
    public double[][] firstLayerWeights;
    public double[][] secondLayerWeights;
    public double[][] outputLayerWeights;

    //The first matrix needs to map to HLS output values, so it needs HLS rows. Each row needs to have a weight for each of the 28^2 (784) inputs, so each row needs to have 784 elements, so there is 784 columns. 
   
    //From the firstLayerBias&Weights, we now are at the first hidden layer. We now need  HLS biases for the second layer, and a HLS by HLS matrix for the second layer.
 
    /*
    //Creates another delta neuralnet using the same size parameters.
    public NeuralNet(NeuralNet orig) {
        NeuralNet delta = new NeuralNet(); 
        delta.firstLayerBias    = new double[orig.firstLayerBias.length];
        delta.secondLayerBias   = new double[orig.secondLayerBias.length];
        delta.outputLayerBias   = new double[orig.outputLayerBias.length];
        delta.firstLayerWeights  = new double[orig.firstLayerWeights.length][orig.firstLayerWeights[0].length];
        delta.secondLayerWeights  = new double[orig.secondLayerWeights.length][orig.secondLayerWeights[0].length];
        delta.outputLayerWeights  = new double[orig.outputLayerWeights.length][orig.outputLayerWeights[0].length];
    }
    */
    //Creates an actual neural net, with randomized parameters iff the boolean is true. ( false for when we are using this constructor in training to collect changes to the NN. 
    public NeuralNet(int inputSize, int HLS, int outputSize, boolean randomParameters) {
		if(inputSize<1 || HLS < 1 || outputSize < 2 ) {
			System.out.println("NN initializer failed sanity check on input parameters.");
			System.exit(1); 
		}        
        firstLayerBias    = new double[HLS];
        secondLayerBias   = new double[HLS];
        outputLayerBias   = new double[outputSize];

        firstLayerWeights  = new double[HLS][inputSize];
        secondLayerWeights = new double[HLS][HLS];
        outputLayerWeights = new double[outputSize][HLS];

        if(!randomParameters) { return; }
        System.out.println("Memory Allocated, randomizing parameters");
        Random rd = new Random();
        for(int i = 0; i<HLS; i++) {
            firstLayerBias[i]  = rd.nextInt(10)-5;
            secondLayerBias[i] = rd.nextInt(10)-5;   
            for(int j = 0; j<inputSize; j++) {
                firstLayerWeights[i][j]  = rd.nextInt(10)-5;
            }
            for(int j = 0; j<HLS; j++) {
                secondLayerWeights[i][j] = rd.nextInt(10)-5;
            }        
        }
        for(int i = 0; i<outputSize; i++) {
             outputLayerBias[i] = rd.nextInt(10)-5;
             for(int j = 0; j<HLS; j++) {
                outputLayerWeights[i][j] = rd.nextInt(10)-5;
            }
        }
    }
   


	//Performs training on the neuralnet, returns the average cost of all the inputs before the model trains. This is used for printing by main.
	//TODO: pass a learning rate parameter 
    public double train(List<Image> bucket, double learnRate) {
        
        //Im going to do this iteratively because I am using a matrix and so do not have a direct reference to what comes before a neuron.
        //Bang for your buck philosophy: When a neuron has high activation, changing the weight of that neuron will proportionally change the activation of the output more than if we changed the weight of a neuron with low activation. Essentially, we are trying to take advantage of this so that when we change the weights, the increased affinity we have with one digit will decrease our cost function more than the decreased affinity we have with other digits will increase our cost function. By doing this stochastically in batches: we can cancel out the "noise" and just get improvements for every single weight and bias. Basically, no one digit can properly tell you the negative gradient, but a whole batch of elements can get close.
        
    
        //Attempt 2 at writing out the procedure in a somewhat pseudocode form:       
        //Create a giant vector called the nudge vector, where each element corresponds in a prticular order to the weights and biases to the model (population below ).
        //Use all zeroes and a 1 for the correct value as the initial desired activation vector
        //For each layer, starting with the final layer:
            //Create a vector of what we want the input layer to look like. (Population  below) 
            //For each node in the output layer: 
                //Nudge the biases for the output layer to manipulate the activation to be closer to its desired value according to the vector we have.
                //populate the nudge vector with weight changes proportional to the  difference between the activation achieved and the activation desired.
                //shift the desired input vector with changes that will make the activation of the output layer be closer to its desired value. This is a voting system amongst all of the output neurons for control over what the input neuron's activation should be
                    //This means: if the activation is too high, we want a higher activation for nodes that have a negative weight to this neuron, and a lower activation for nodes that have a positive weight to this neuron, and vice versa if the activation is too low. 
             
            //Utilize the new activation vector we have made as the desired change vector. Note that we cannot change the initial input, so we just change the weights to those values. 
            
             
        
        
        //So i will do this in 2 methods. One is just to compute, the other is to pass in the bucket one element at a time and keep track of everything, and apply the changes at the end. 
        //System.out.println("Enterred train method. bucket size: " + bucket.size());
        NeuralNet grandDelta = computeDelta(bucket.get(0));
        double cost = 0.0;   
		int size = bucket.size(); 
		for(int i = 1; i<size; i++) { //Here we are holding onto the first series of changes, then just modifying it as we go.
            grandDelta.combineNeuralNets(computeDelta(bucket.get(i)),1.0); 
            cost+=cost(predict(bucket.get(i).data),bucket.get(i).label);
        }   
        cost/=bucket.size();
        combineNeuralNets(grandDelta,(learnRate*-1)/bucket.size()); //Apply the changes to the gradient to our model
    	return cost;
	}

    //Computes the desired changes to the neuralnet, and stores them in a delta NN.
    public NeuralNet computeDelta(Image image) {
        NeuralNet delta = new NeuralNet(784,20,10, false);
        if(delta.firstLayerBias==null) {
            System.out.println("Error constructing new model");
        }           

        //So the first thing we need to do is we need to pass this input through the model, holding onto all relevant information along the way.  
        //The main important thing that we hold onto, is the weighted activation values of the neurons.

        double[] firstLayerVals = matrixVectorMult(firstLayerWeights, image.data);
        vectorAddition(firstLayerVals,firstLayerBias); 
        double[] firstLayerWeightedActivations = Arrays.copyOf(firstLayerVals,firstLayerVals.length);
        firstLayerWeightedActivations = sigmoidPrimeVector(firstLayerWeightedActivations);
        sigmoidVector(firstLayerVals);
        double[] firstLayerActivations = Arrays.copyOf(firstLayerVals,firstLayerVals.length);
        

        double[] secondLayerVals = matrixVectorMult(secondLayerWeights, firstLayerVals);
        vectorAddition(secondLayerVals,secondLayerBias);
        double[] secondLayerWeightedActivations = Arrays.copyOf(secondLayerVals,secondLayerVals.length);
        secondLayerWeightedActivations = sigmoidPrimeVector(secondLayerWeightedActivations); 
        sigmoidVector(secondLayerVals);
        double[] secondLayerActivations = Arrays.copyOf(secondLayerVals,secondLayerVals.length);
        
        double[] resultLayerVals = matrixVectorMult(outputLayerWeights, secondLayerVals);
        vectorAddition(resultLayerVals,outputLayerBias);         
        double[] resultLayerWeightedActivations = Arrays.copyOf(resultLayerVals,resultLayerVals.length);
        resultLayerWeightedActivations = sigmoidPrimeVector(resultLayerWeightedActivations);
        sigmoidVector(resultLayerVals);
        double[] resultLayerActivations = Arrays.copyOf(resultLayerVals,resultLayerVals.length);

        
        //Keeping track of everything, we now have: the sigmoid ' ( z^L ) as the first second and result layer weighted activations, and the actual activation of every neuron.  
        
        
        //Now we find the error in the output layer.
        double[] desiredOutputActivations = new double[10];
        desiredOutputActivations[image.label] = 1;
        
        
        double[] outputActivationError = Arrays.copyOf(resultLayerActivations,resultLayerActivations.length);
        //The error is just the difference between what we have and what we want ( because its the derivative of the cost function which we specifically designed to have this derivative ) so we just copy the values. 
        outputActivationError[image.label]-=1;  //Of course we want the actual value to be a one, so we subtract one instead of 'subtracting zero' which was gotten from the copy of the output layers activations. 
        outputActivationError = hardamadProduct(outputActivationError,resultLayerWeightedActivations); //Taking the hardamad product gets us to our delta ^ L 
        //We now need to find the delta^L-1. Since we have only 2 layers I will do this without a layer to layer for loop.
        
        
        //double[][] outputWeightTranspose = computeTranspose(outputLayerWeights);   
        double[] secondLayerActivationError = matrixVectorMult(computeTranspose(outputLayerWeights),outputActivationError);
        secondLayerActivationError = hardamadProduct(secondLayerActivationError, secondLayerWeightedActivations);
        //We now have the errors in the activations of the second layer.
        double[] firstLayerActivationError = matrixVectorMult(computeTranspose(secondLayerWeights),secondLayerActivationError);
        firstLayerActivationError = hardamadProduct(firstLayerActivationError, firstLayerWeightedActivations);
        //We now have the errors in the activations of the first layer. 

        //According to our formula we can apply the bias error exactly as it is.    
        for(int i = 0; i<delta.outputLayerBias.length; i++) {
            delta.outputLayerBias[i] = outputActivationError[i];
        }
        for(int i = 0; i<delta.secondLayerBias.length; i++) {
            delta.secondLayerBias[i] = secondLayerActivationError[i];
            delta.firstLayerBias[i] =  firstLayerActivationError[i];
        }
        //The rate of change of the weights is only slightly more complicated.
        //The rate of change of the cost function with respect to the weight that goes from the j'th neuron in the l-1'th layer to the k'th neuron in the l'th layer is equal to the error term we already have of the activation of the k'th neuron, times the activation value of the input neuron.
        
        for(int i = 0; i<firstLayerWeights.length; i++) {
            for(int j = 0; j<firstLayerWeights[0].length; j++) {
                delta.firstLayerWeights[i][j] = image.data[j]*firstLayerActivationError[i];
            }
        }
        for(int i = 0; i<secondLayerWeights.length; i++) {
            for(int j = 0; j<secondLayerWeights[0].length; j++) {
                delta.secondLayerWeights[i][j] = firstLayerActivations[j]*secondLayerActivationError[i];
            }
        }
        for(int i = 0; i<outputLayerWeights.length; i++) {
            for(int j = 0; j<outputLayerWeights[0].length; j++) {
                delta.outputLayerWeights[i][j] = secondLayerActivations[j]*outputActivationError[i];
            }
        }
        return delta;                 
    }

    //Computes the transpose of a matrix and returns it as a full copy. As such, it allocates a large amount of memory. garbage collection does a good job, but use this sparingly.
	//Issue with not using this: we need it in order to do the matrix vector multiplication operation. We would have to make a combined 
    public static double[][] computeTranspose(double[][] matrix) {
        double[][] transposeMatrix = new double[matrix[0].length][matrix.length];
        for(int i = 0; i<transposeMatrix.length; i++) {
            for(int j = 0; j<transposeMatrix[0].length; j++) {
                transposeMatrix[i][j] = matrix[j][i];
            }
        }
        return transposeMatrix; 
    }
	//Similar to computeTranspose, this creates a new copy in memory for the output. This time its the hardamad product ( take 2 vectors and multiply the vals at each index, product in a new vector of same length)
    public static double[] hardamadProduct(double[] a, double[] b) {
        if(a.length!=b.length) {
            System.out.println("Hardamad product error: length mismatch");
            System.exit(1);
        }
        double[] product = new double[a.length];
        for(int i = 0; i<a.length; i++) {
            product[i] = a[i]*b[i];
        }
        return product;
    }


    //This function is used because we are holding neural net objects as gradient vectors as well as for actual neural nets, so we need to combine them.
    //sign boolean is true when we are adding the values, and false when subtracting.
    //Factor is used to divide the values of the delta out before adding them to our main network. This is used because we are looking for the average change that each training sample wants, so we need to divide out by bucketsize, 100.
    public void combineNeuralNets(NeuralNet delta, double factor) {
        for(int i = 0; i<this.firstLayerBias.length; i++) {
            this.firstLayerBias[i]+=factor*delta.firstLayerBias[i];
            this.secondLayerBias[i]+=factor*delta.secondLayerBias[i];
            for(int j = 0; j<784; j++) {
                this.firstLayerWeights[i][j] += factor*delta.firstLayerWeights[i][j];
            }
            for(int j = 0; j<this.firstLayerBias.length; j++) {
                this.secondLayerWeights[i][j]+=factor*delta.secondLayerWeights[i][j];
            }    
        }
        for(int i = 0; i<10; i++) {
            this.outputLayerBias[i]+=factor*delta.outputLayerBias[i];
            for(int j = 0; j<this.outputLayerWeights[0].length; j++) {
                this.outputLayerWeights[i][j] += factor*delta.outputLayerWeights[i][j];
            }
        }
    } 

    //Returns a vector prediction of the value that gets passed to the NN. Optimized for testing speed and does not hold onto the partial values in the net needed for backpropagation. 
    public double[] predict(double[] image) {
        if(image.length!=784) {
            System.out.println("Error (NN): bad image size passed to prediction");
        }
            

        double[] firstLayerVals = matrixVectorMult(firstLayerWeights, image);
        vectorAddition(firstLayerVals,firstLayerBias); 
        sigmoidVector(firstLayerVals);
        //Since our final output vector will be length 10, we cannot carry it forward all the way, this increases our memory load somewhat. 
    
        double[] secondLayerVals = matrixVectorMult(secondLayerWeights, firstLayerVals);
        vectorAddition(secondLayerVals,secondLayerBias);
        sigmoidVector(secondLayerVals);
        
        double[] resultVector = matrixVectorMult(outputLayerWeights, secondLayerVals);
        vectorAddition(resultVector,outputLayerBias);
        sigmoidVector(resultVector);
        return resultVector; 
    }

    //Returns the cost of a particular guess. 
    public static double cost(double[] prediction, int realAns){
        double ret = 0;
        for(int i = 0; i<10; i++) {
            if(i==realAns) {
                ret+=Math.pow(prediction[i]-1,2);
            } else {
                ret+=Math.pow(prediction[i],2);
            }
            
        }
        return ret/2;                
 
    }


    //Performs matrix vector multiplication. The vector corresponds to each input neuron, and the matrix spot i,j corresponds to output neuron number i's weight at j. 
    //The answer to this will get added into the bias vector we are storing per layer, then get input into the sigmoid. 
    public static double[] matrixVectorMult(double[][] matrix, double[] vector) {
        //System.out.println("Enterred matrixVectorMult with matrix with " + matrix.length + " rows and " + matrix[0].length + " cols and a vector with length " + vector.length);

        if(matrix[0].length!=vector.length) {
            System.out.println("Error: matrix vector size mismatch");
            System.exit(1);
        }
        double[] ans = new double[matrix.length]; //answer has number of rows of matrix
        for(int i = 0; i<matrix.length; i++) { //For each B
            //System.out.println("i = " + i);
            double val = 0;
            for(int j = 0; j<matrix[i].length; j++) { //Go across the row multiplying B's of j times matrix[i]'s of j
                val += matrix[i][j]*vector[j];
            }
            ans[i] = val;
        }
        return ans;
    }
    
    //Returns vector after each element has add[i] added to it. Lets me one liner adding in the biases. 
    public static void vectorAddition(double[] vector, double[] add) {
        if(vector.length!=add.length) {
            System.out.println("Error: vectorAddition length mismatch: vector: " + vector.length + " add: " + add.length);
        }
        for(int i = 0; i<vector.length; i++) {
            vector[i]+=add[i];
        }
    }        




    //Sigmoid squishification function.  
    public static double sigmoid(double input) {
        return 1.0/(1+(Math.pow(Math.E,input*-1)));
    }
    
    //Performs sigmoid over a vector
    public static void sigmoidVector(double[] vector) {
        for(int i = 0; i<vector.length; i++) {
            vector[i] = sigmoid(vector[i]);
        }
    }
    
    
    //Computes the derivative of the sigmoid function at the input.
    public static double sigmoidPrime(double input) {
        return sigmoid(input)*(1-sigmoid(input));
    }

    public static double[] sigmoidPrimeVector(double[] vector) {
        double[] result = new double[vector.length];
        for(int i = 0; i<vector.length; i++) {
            result[i] = sigmoidPrime(vector[i]);
        }
        return result;
    }

    //returns max value from the array.
    public static int vote(double[] prediction) {
        int max = 0;
        for(int i = 1; i<10; i++) {
            if(prediction[i]>prediction[max]) {
                max = i;
            }
        }
        return max;
	}   
}
