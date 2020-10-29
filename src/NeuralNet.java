//Version 2.0 of the main NN class. The main difference is we are allowing for a generic number of hidden layers rather than hard coding 2. This was done by simply holding an array of layers, where each layer has its own weight matrix and bias vector. This was chosen over lists because java wrappers are very slow compared to primitives.  Even in lists best case scenario, where they would be autoboxed and unboxed, that still is more overhead over simply using primitive arrays. This also couples the layers, as opposed to if i used a list, there would be a risk of me removing an element and offsetting the weights and biases, which would be very bad. 

import java.util.*;
import java.io.*;

public class NeuralNet implements Serializable{
	private int HLS, HLQ, insize, outsize;    
	private double learnRate;
	public NNLayer[] layers;
    //Creates an actual neural net, with randomized parameters iff the boolean is true. ( false for when we are using this constructor in training to collect changes to the NN. )
    public NeuralNet(int inputSize, int HLS, int HLQ, int outputSize, boolean randomParameters, double learnRate) {
		if(inputSize<1 || HLS < 1 || HLQ < 1 || outputSize < 2 ) {
			System.out.println("NN initializer failed sanity check on input parameters.");
			System.exit(1); 
		}        
		this.insize = inputSize;
		this.outsize = outputSize;
		this.HLS = HLS;
		this.HLQ = HLQ;
		this.learnRate = learnRate;
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
        //System.out.println("Memory Allocated, randomizing parameters");
        Random rd = new Random();
		rd.setSeed(3);
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
   
	public double getLearnRate(){ return this.learnRate; }
	public void   setLearnRate(double newLearnRate) { this.learnRate = newLearnRate; }

	
    //Organizes the training of the NN. Takes in the training data. 
	public double train(List<Image> bucket) {
        
        //Im going to do this iteratively because I am using a matrix and so do not have a direct reference to what comes before a neuron.
        //Bang for your buck philosophy: When a neuron has high activation, changing the weight of that neuron will proportionally change the activation of the output more than if we changed the weight of a neuron with low activation. Essentially, we are trying to take advantage of this so that when we change the weights, the increased affinity we have with one digit will decrease our cost function more than the decreased affinity we have with other digits will increase our cost function. By doing this stochastically in batches: we can cancel out the "noise" and just get improvements for every single weight and bias. Basically, no one digit can properly tell you the negative gradient, but a whole batch of elements can get close.
        
        
        //So i will do this in 2 methods. One is just to compute, the other is to pass in the bucket one element at a time and keep track of everything, and apply the changes at the end. 
        //System.out.println("Enterred train method. bucket size: " + bucket.size());
        
		//NeuralNet grandDelta = computeDelta(bucket.get(0));
		NeuralNet grandDelta = new NeuralNet(insize, HLS, HLQ, outsize, false, 0.0);
		NeuralNet transferNet = new NeuralNet(insize, HLS, HLQ, outsize, false, 0.0); //used to transfer into the granddelta
        double cost = 0.0; //error metric, used for printing exclusively at the moment.
		int size = bucket.size(); 
		for(int i = 0; i<size; i++) { //Here we are holding onto the first series of changes, then just modifying it as we go.
			Image image = bucket.get(i);
			computeDelta(image, transferNet);
            grandDelta.combineNeuralNets(transferNet); //adds in the changes to the delta NN.
			double[] output = forwardProp(image.data); //printing this out, so holding it, when not printing replace accordingly.
            cost+=cost(forwardProp(image.data),image.label);
            cost+=cost(output,image.label);
			//System.out.println(Arrays.toString(output));

        }   
        cost/=bucket.size();
    	//grandDelta.multiplyNNByScalar((learnRate*-1)/bucket.size());
		grandDelta.multiplyNNByScalar(-1.0*learnRate/bucket.size()); //Scales the delta NN by the learn rate factor. 
		combineNeuralNets(grandDelta);
		return cost;
	}

    //Computes the desired changes to the neuralnet, and stores them into a delta NN.
    public void computeDelta(Image image, NeuralNet delta) {	
		double[][] weightedActivations = new double[HLQ+1][];	//Holds our delta l's 
		double[][] activations = new double[HLQ+1][]; //Yay for jagged arrays. 
		//Forward pass
		for(int i = 0; i<=HLQ; i++) {
			//System.out.println("Backprop, forward stage: " + i);
			double[] previousActivation = (i==0)? image.data : activations[i-1];
			activations[i] = LinAlg.matrixVectorMult(layers[i].weightMatrix, previousActivation);    
			LinAlg.vectorAdditionShallow(activations[i],layers[i].biasVector); 
			weightedActivations[i] = LinAlg.sigmoidPrimeVectorDeep(activations[i]);	
			LinAlg.sigmoidVectorShallow(activations[i]);
		}

		/*	
		//Test to verify that the forward pass was accurate in train compared to forward prop. 
		System.out.println("train's forward pass: "+Arrays.toString(activations[HLQ]));
		System.out.println("Verified forward pass: "+Arrays.toString(forwardProp(image.data)));
		*/

		double[][] activationErrors = activations; //Renaming for simplicity
		activationErrors[HLQ][image.label]-=1; //A^L-y = gradient vector	
		LinAlg.hardamadShallow(activationErrors[HLQ], weightedActivations[HLQ]); //output error
		//Backprop pass
        for(int i = HLQ-1; i>=0; i--) {		
			activationErrors[i] = LinAlg.matrixVectorMultTranspose(layers[i+1].weightMatrix,activationErrors[i+1]);
			LinAlg.hardamadShallow(activationErrors[i],weightedActivations[i]);
		}	
		//output
		for(int i = 0; i<HLQ; i++) { 
			for(int j = 0; j<delta.layers[i].biasVector.length; j++) { 
				delta.layers[i].biasVector[j] = activationErrors[i][j];	
			}
			double[] activation = (i==0)? image.data : activations[i-1];     
			for(int j = 0; j<delta.layers[i].weightMatrix.length; j++) { 
				for(int k = 0; k<delta.layers[i].weightMatrix[j].length; k++) {
					delta.layers[i].weightMatrix[j][k] = activation[k]*activationErrors[i][j];
				}
			}
		}
		
		//Now we test that the delta holds what we wanted it to hold. 

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

	//Similar to combine NeuralNets, but instead of averaging the Nets, it takes in the parameter and replaces the Callers NN. This is used to keep the memory allocation down.  
	//Currently unused because its unneeded, changed compute delta instead. 
    public void replaceNeuralNets(NeuralNet delta) {
		for(int i = 0; i<layers.length; i++) {
			double[][] weightA = layers[i].weightMatrix;
			double[][] weightB = delta.layers[i].weightMatrix;
			for(int j = 0; j<weightA.length; j++) {
				for(int k = 0; k<weightA[j].length; k++) {
					weightA[j][k] = weightB[j][k];
				}
			}
			double[] biasA = layers[i].biasVector;
			double[] biasB = delta.layers[i].biasVector;
			for(int j = 0; j<biasA.length; j++) {
				biasA[j] = biasB[j];
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

    //Returns a vector prediction of the value that gets passed to the NN. Optimized for testing speed and does not hold onto the partial values in the net needed for backprop.
    public double[] forwardProp(double[] image) {
        if(image.length!=insize) {
            System.out.println("Error (NN): bad image size passed to prediction");
        }

		
		double[] activation = Arrays.copyOf(image, image.length); //TODO: make this unneeded.
		for(int i = 0; i<HLQ+1; i++) {
			activation = LinAlg.matrixVectorMult(layers[i].weightMatrix,activation);
			LinAlg.vectorAdditionShallow(activation, layers[i].biasVector);
			LinAlg.sigmoidVectorShallow(activation);			
		}
		return activation;
    }

    //Returns the cost of a particular guess. Only needed for diagnostics, this is slow.
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
	//Returns a String containing a lot of metadata, used by drivers metadata function to print to user in command line
	public String metadata() {
		StringBuilder sb = new StringBuilder();
		sb.append("Input Layer Size: "      + this.insize    +"\n");
		sb.append("Hidden Layer Size: "     + this.HLS       +"\n");
		sb.append("Hidden Layer Quantity: " + this.HLQ       +"\n");
		sb.append("Output Layer Size: "     + this.outsize   +"\n");
		sb.append("Learn rate: "            + this.learnRate +"\n");
		return sb.toString();
		
	
	}

}
