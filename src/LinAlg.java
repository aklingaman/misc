//Contains a bunch of static linear algebra functions, primarily split on a shallow copy and a deep copy. 
//None of these functions have any verifications on them because some of them get used many many times, and comparing against null every time gets boring. I have some commented out sanity checks that can be enabled incase it matters. 
//Shallow functions will perform onto the first vector passed. 
public abstract class LinAlg {
	
	public static void hardamadShallow(double[] a, double[] b) {
		for(int i = 0; i<a.length; i++) {
			a[i]*=b[i];
		}	
	}

	public static double[] hardamadDeep(double[] a, double[] b) {
		double[] c = new double[a.length];
		for(int i = 0; i<a.length; i++) {
			c[i] = a[i]*b[i];
		}	
		return c;		
	}
	
	public static void vectorAdditionShallow(double[] a, double[] b) {
		for(int i = 0; i<a.length; i++) {
			a[i]+=b[i];
		}	
	}

	public static double[] vectorAdditionDeep(double[] a, double[] b) {
		double[] c = new double[a.length];
		for(int i = 0; i<a.length; i++) {
			c[i] = a[i]+b[i];
		}	
		return c;		
	}
	//returns index of maximal element passed. Ties broken by lowest index.
	public static int vote(double[] a) {
		int max = 0;
		for(int i = 1; i<a.length; i++) {
			if(a[i]>a[max]) { max = i; }
		}		
		return max;
	}	
		
	//Simple matrix vector mult. Shallow vs deep doesnt make sense because we need the entire vector until the end. 
    public static double[] matrixVectorMult(double[][] matrix, double[] vector) {
        //System.out.println("Enterred matrixVectorMult with matrix with " + matrix.length + " rows and " + matrix[0].length + " cols and a vector with length " + vector.length);
		/*
        if(matrix[0].length!=vector.length) {
            System.out.println("Error: matrix vector size mismatch");
            System.out.println("Matrix has: " + matrix.length + " rows, and " + matrix[0].length+" cols.");
			System.out.println("Vector has: " + vector.length + " elements");
			System.exit(1);
        }
		*/
        double[] ans = new double[matrix.length]; //answer has number of rows of matrix
        for(int i = 0; i<matrix.length; i++) { //For each B
            //System.out.println("i = " + i);
            ans[i] = 0;
            for(int j = 0; j<matrix[i].length; j++) { //Go across the row multiplying B's of j times matrix[i]'s of j
                ans[i] += matrix[i][j]*vector[j];
            }
        }
        return ans;
    }
	//Removes the need for compute transpose by allowing us to multiply the matrix input's transpose by the vector, by just bounds switching rather than computing transpose then using matrixvector mult. 
	public static double[] matrixVectorMultTranspose(double[][] matrix, double[] vector) {
        double[] ans = new double[matrix[0].length]; //answer has number of rows of matrix
		for(int i = 0; i<matrix[0].length; i++) {
			ans[i] = 0; 
			for(int j = 0; j<matrix.length; j++) {
				double a = matrix[j][i];
				double b = vector[j]; 
				ans[i]+=a*b;
			}
		}

		

		return ans;	
	}


	//Computes the transpose of a matrix. Theres no good reason to use this for speed, if all is well this shouldnt actually be getting called, simply used to verify correctness.  
	

//This is because the main way the transpose gets used is by multiplying things times the transpose, which can effectively be done by just using the original multiply function, but switching our i's and j's properly. 
    public static double[][] computeTranspose(double[][] matrix) {
        double[][] transposeMatrix = new double[matrix[0].length][matrix.length];
        for(int i = 0; i<transposeMatrix.length; i++) {
            for(int j = 0; j<transposeMatrix[0].length; j++) {
                transposeMatrix[i][j] = matrix[j][i];
            }
        }
        return transposeMatrix; 
    }
	
	//Now comes the sigmoidVector and sigmoidprimevector functions. These are optimized for speed and so they look terrible.
	public static void sigmoidVectorShallow(double[] a) {
		for(int i = 0; i<a.length; i++) {
			a[i] = 1.0/(1.0+(Math.pow(Math.E,a[i]*-1.0)));
			//a[i] = a[i]/(1+Math.abs(a[i]));
		}
	}
	public static double[] sigmoidVectorDeep(double[] a) {
		double[] b = new double[a.length];
		for(int i = 0; i<a.length; i++) {
			b[i] = 1.0/(1.0+(Math.pow(Math.E,a[i]*-1.0)));	
			//b[i] = a[i]/(1+Math.abs(a[i]));
		}
		return b;
	}
	public static void sigmoidPrimeVectorShallow(double[] a) {
		for(int i = 0; i<a.length; i++) {
			double sigmoid = 1.0/(1.0+(Math.pow(Math.E,a[i]*-1.0)));
			//double sigmoid = a[i]/(1+Math.abs(a[i]));
			a[i] = sigmoid*(1-sigmoid);
		}
	}
	public static double[] sigmoidPrimeVectorDeep(double[] a) {
		double[] b = new double[a.length];
		for(int i = 0; i<a.length; i++) {
			double sigmoid = 1.0/(1.0+(Math.pow(Math.E,a[i]*-1.0)));
			//double sigmoid = a[i]/(1+Math.abs(a[i]));
			b[i] = sigmoid*(1-sigmoid);
		}
		return b;
	}	

}

