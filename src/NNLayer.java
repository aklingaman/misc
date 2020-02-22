import java.io.*;
public class NNLayer implements Serializable {
	public double[] biasVector;
	public double[][] weightMatrix;
	public NNLayer(int biasSize, int rowSize, int colSize) {
		biasVector = new double[biasSize];
		weightMatrix = new double[rowSize][colSize];	
	}
}
