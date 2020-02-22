//Version 2.0 of the main NN class. The main difference is we are allowing for a generic number of hidden layers rather than hard coding 2. This was done by simply holding an array of layers, where each layer has its own weight matrix and bias vector. This was chosen over lists because java wrappers are very slow compared to primitives.  Even in lists best case scenario, where they would be autoboxed and unboxed, that still is more overhead over simply using primitive arrays. This also couples the layers, as opposed to if i used a list, there would be a risk of me removing an element and offsetting the weights and biases, which would be very bad. 

import java.util.*;
import java.io.*;

public class NeuralNet implements Serializable{
	private int HLS, HLQ, insize, outsize;    
	public NNLayer[] layers;
    //Creates an actual neural net, with randomized parameters iff the boolean is true. ( false for when we are using this constructor in training to collect changes to the NN. )
    public NeuralNet(int inputSize, int HLS, int HLQ, int outputSize, boolean randomParameters) {
		if(inputSize<1 || HLS < 1 || HLQ < 1 || outputSize < 2 ) {
			System.out.println("NN initializer failed sanity check on input parameters.");
			System.exit(1); 
		}        
		this.insize = inputSize;
		this.outsize = outputSize;
		this.HLS = HLS;
		this.HLQ = HLQ;
		layers = new NNLayer[HLQ+1]; //+1 is for the output layer.
		for(int i = 0; i<=HLQ; i++) {
			if(i == 0) {
				layers[i] = new NNLayer(HLS,HLS,inputSize);
				continue;
			}
			if(i==HLQ) {
				layers[i] = new NNLayer(outputSize,outputSize,HLS);
				continue;
			}
			layers[i] = new NNLayer(HLS,HLS,HLS);
		}


		
        if(!randomParameters) { return; }
        System.out.println("Memory Allocated, randomizing parameters");
        Random rd = new Random();
		for(int i = 0; i<layers.length; i++) {
			double[][] weight = layers[i].weightMatrix;
			double[] bias = layers[i].biasVector;
			for(int j = 0; j<weight.length; j++) {
				for(int k = 0; k<weight[j].length; k++) {
					weight[j][k] = rd.nextDouble()*10-5;
				}
			}
			for(int j = 0; j<bias.length; j++) {
				bias[j] = rd.nextDouble()*10-5;
			}	
		}			
    }
   

	
    //Organizes the training of the NN. Takes in the training data. 
	public double train(List<Image> bucket, double learnRate) {
        
        //Im going to do this iteratively because I am using a matrix and so do not have a direct reference to what comes before a neuron.
        //Bang for your buck philosophy: When a neuron has high activation, changing the weight of that neuron will proportionally change the activation of the output more than if we changed the weight of a neuron with low activation. Essentially, we are trying to take advantage of this so that when we change the weights, the increased affinity we have with one digit will decrease our cost function more than the decreased affinity we have with other digits will increase our cost function. By doing this stochastically in batches: we can cancel out the "noise" and just get improvements for every single weight and bias. Basically, no one digit can properly tell you the negative gradient, but a whole batch of elements can get close.
        
            
             
        
        
        //So i will do this in 2 methods. One is just to compute, the other is to pass in the bucket one element at a time and keep track of everything, and apply the changes at the end. 
        //System.out.println("Enterred train method. bucket size: " + bucket.size());
        NeuralNet grandDelta = computeDelta(bucket.get(0));
        //double cost = 0.0; //error metric, used for printing exclusively at the moment.
		int size = bucket.size(); 
		for(int i = 1; i<size; i++) { //Here we are holding onto the first series of changes, then just modifying it as we go.
			Image image = bucket.get(i);
            grandDelta.combineNeuralNets(computeDelta(image)); 
            //cost+=cost(predict(image.data),image.label);
        }   
        //cost/=bucket.size();
    	grandDelta.multiplyNNByScalar((learnRate*-1)/bucket.size());
		combineNeuralNets(grandDelta);
		//return cost;
		return 0.0;
	}

    //Computes the desired changes to the neuralnet, and stores them in a delta NN.
    public NeuralNet computeDelta(Image image) {
        NeuralNet delta = new NeuralNet(insize, HLS, HLQ, outsize, false);

        //So the first thing we need to do is we need to pass this input through the model, holding onto all relevant information along the way.  
        //The main important thing that we hold onto, is the weighted activation values of the neurons.
				
		double[][] weightedActivations = new double[HLQ+1][];	//Holds our delta l's 
		double[][] activations = new double[HLQ+1][]; //Yay for jagged arrays, we can just leave the 2nd param blank and instantiate the inner arrays whenever we want later. 
		for(int i = 0; i<=HLQ; i++) {
			//System.out.println("Backprop, forward stage: " + i);
			double[] previousActivation;
			if(i == 0) {
				previousActivation = image.data;
			} else {
				previousActivation = activations[i-1]; 
			}	
			activations[i] = matrixVectorMult(layers[i].weightMatrix, previousActivation);              //Allocate for the next layers activations by matrix vector multing the last layers activations by this layers weight matrix.
			vectorAddition(activations[i],layers[i].biasVector); 												  //Chuck in the biases.
			weightedActivations[i] = sigmoidPrimeVector(activations[i]);	
			sigmoidVector(activations[i]);
		}
        //The error is just the difference between what we have and what we want ( because its the derivative of the cost function which we specifically designed to have this derivative ) so we just copy the values. 
		//The tricky part here is we want to list the errors going forward, but we have to compute them backwards. 
		double[][] activationErrors = new double[HLQ+1][];
		activationErrors[HLQ] = Arrays.copyOf(activations[HLQ],activations[HLQ].length);
		activationErrors[HLQ][image.label]-=1;	
		activationErrors[HLQ] = hardamadProduct(activationErrors[HLQ], weightedActivations[HLQ]);
        for(int i = HLQ-1; i>=0; i--) {		
			activationErrors[i] = matrixVectorMult(computeTranspose(layers[i+1].weightMatrix),activationErrors[i+1]);
			activationErrors[i] = hardamadProduct(activationErrors[i],weightedActivations[i]);
		}	
			
        //According to our formula, the partial derivative is just the delta we already calculated, so we just apply it directly. 
		for(int i = 0; i<HLQ; i++) {
			for(int j = 0; j<delta.layers[i].biasVector.length; j++) {
				delta.layers[i].biasVector[j] = activationErrors[i][j];	
			}
		}
    	//Formula says we multiply our errors by the activations that fed into it.     
		for(int i = 0; i<HLQ; i++) {
			double[] activation = (i==0)? image.data : activations[i];
			for(int j = 0; j<delta.layers[i].weightMatrix.length; j++) {
				for(int k = 0; k<delta.layers[i].weightMatrix[j].length; k++) {
					delta.layers[i].weightMatrix[j][k] = activation[k]*activationErrors[i][j];
				}
			}
		}
        return delta;                 
    }

    //Computes the transpose of a matrix and returns it as a full copy. As such, it allocates a large amount of memory. garbage collection does a good job, but use this sparingly.
 
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
        //System.out.println(a.length); //Debug print for ambiguous NPE on a line with multiple object calls.
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

	//takes in a NN and combines with NN caller. I originally had a factor multiplier, but i removed it in favor of having a separate function to handle it. In the interest of speed, i may bring it back to see what that does to the runtime. 
    public void combineNeuralNets(NeuralNet delta) {
		for(int i = 0; i<layers.length; i++) {
			double[][] weightA = layers[i].weightMatrix;
			double[][] weightB = delta.layers[i].weightMatrix;
			for(int j = 0; j<weightA.length; j++) {
				for(int k = 0; k<weightA[j].length; k++) {
					weightA[j][k] += weightB[j][k];
				}
			}
			double[] biasA = layers[i].biasVector;
			double[] biasB = delta.layers[i].biasVector;
			for(int j = 0; j<biasA.length; j++) {
				biasA[j]+=biasB[j];
			}
		}
    }

	//Multiplies a NN's weights and biases by a factor. This is used by the training function because we need to apply a learning rate that changes how fast changes get made. 
	public void multiplyNNByScalar(double factor) {
		for(int i = 0; i<layers.length; i++) {
			double[][] weight = layers[i].weightMatrix;
			double[] bias = layers[i].biasVector;
			for(int j = 0; j<bias.length; j++) {
				bias[j]*=factor;
			}
			for(int j = 0; j<weight.length; j++) {
				for(int k = 0; k<weight[j].length; k++) {
					weight[j][k]*=factor;
				}
			}
		}			
	}

    //Returns a vector prediction of the value that gets passed to the NN. Optimized for testing speed and does not hold onto the partial values in the net needed for backpropagation. 
    public double[] predict(double[] image) {
        if(image.length!=insize) {
            System.out.println("Error (NN): bad image size passed to prediction");
        }

		
		double[] activation = Arrays.copyOf(image, image.length);
		for(int i = 0; i<HLQ+1; i++) {
			activation = matrixVectorMult(layers[i].weightMatrix,activation);
			vectorAddition(activation, layers[i].biasVector);
			sigmoidVector(activation);			
		}
		return activation;
    }

    //Returns the cost of a particular guess. 
    public static double cost(double[] prediction, int realAns){
		double ret = 0;
        for(int i = 0; i<prediction.length; i++) {
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
	//Currently this is done in the basic way, could be modified later to be made faster, but AFAIK ATM only sparse matrices have significant speedups.
    public static double[] matrixVectorMult(double[][] matrix, double[] vector) {
        //System.out.println("Enterred matrixVectorMult with matrix with " + matrix.length + " rows and " + matrix[0].length + " cols and a vector with length " + vector.length);

        if(matrix[0].length!=vector.length) {
            System.out.println("Error: matrix vector size mismatch");
            System.out.println("Matrix has: " + matrix.length + " rows, and " + matrix[0].length+" cols.");
			System.out.println("Vector has: " + vector.length + " elements");
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
            System.out.println("Error: vectorAddition length mismatch: vector: " + vector.length + ", and " + add.length);
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

	//Computes sigmoid prime over a vector. The thing is, since this function gets called on memory we want to keep, we need to make a copy SOMEWHERE, so im making here because it looks cleaner than make a copy, then modify all the values in a second loop. 
    public static double[] sigmoidPrimeVector(double[] vector) {
        double[] copy = new double[vector.length];
		for(int i = 0; i<vector.length; i++) {
            copy[i] = sigmoidPrime(vector[i]);
        }
    	return copy;
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
