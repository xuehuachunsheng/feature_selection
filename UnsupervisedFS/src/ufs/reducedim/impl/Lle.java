package ufs.reducedim.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.reducedim.DimensionalReduction;
import ufs.utils.Utils;

/**
 * The implementation of Locality Linear Embedding. See more details in 'Machine
 * learning, Zhihua Zhou'.
 * 
 * @author Yanxue
 *
 */
public class Lle extends DimensionalReduction {

	/**
	 * R^n×k, where n is the number of samples and k is the number of neighbors.
	 * The element located in i-th row and j-th column stores the index of the
	 * j-th neighbor of the i-th sample.
	 * 
	 */
	Matrix neighborIndicesMatrix;

	/**
	 * R^n×n. If the j-th sample is the i-th sample's neighbor, the element
	 * located in i-th row and j-th column stores the coefficient. Else 0. Of
	 * course, the elements in the diagonal line is zero.
	 */
	Matrix coefficientMatrix;

	int numDimension = 2;
	
	public int getNumDimension() {
		return numDimension;
	}

	public void setNumDimension(int numDimension) {
		this.numDimension = numDimension;
	}

	public Lle(Matrix pData) {
		super(pData);
		// TODO Auto-generated constructor stub
	}

	public Lle(Matrix pData, Matrix pNeighborIndicesMatrix) {
		super(pData);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
	}
	
	public Lle(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumDimension) {
		super(pData);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
		numDimension = pNumDimension;
	}

	@Override
	public void middleProcess() {
		computeCoefficientMatrix();
		computeLowDimData();
	}

	public void computeLowDimData() {
		Matrix I = Matrix.Factory.eye(coefficientMatrix.getRowCount(), coefficientMatrix.getRowCount());
		Matrix I_W = I.minus(coefficientMatrix);
		Matrix M = I_W.transpose().mtimes(I_W);
		Matrix[] eig = M.eig();
		long[] dimensions = new long[numDimension]; 
		for (int j = 0; j < dimensions.length; j++) {
			dimensions[j] = j;
		}
		reducedData = eig[0].selectColumns(Ret.NEW, dimensions);
	}

	public Matrix computeCoefficientMatrix() {
		int k = (int) neighborIndicesMatrix.getColumnCount();
		int n = (int) data.getRowCount();
		Matrix W = Matrix.Factory.zeros(n, n);
		for (int i = 0; i < n; i++) {
			Matrix C = constructC(i);
			double sumC = C.sum(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);
			for (int j = 0; j < k; j++) {
				W.setAsDouble(C.getAsDouble(j, 0) / sumC, i,
						neighborIndicesMatrix.getAsInt(i, j));
			}
//			System.out.println(i + "-th for:");
		}
		coefficientMatrix = W;
		return W;
	}

	private Matrix constructC(int xi) {
		int k = (int) neighborIndicesMatrix.getColumnCount();
		Matrix tC = Matrix.Factory.zeros(k, k);

		for (int i = 0; i < k; i++) {
			for (int j = i; j < k; j++) {
				Matrix xiSample = data
						.selectRows(Ret.NEW, xi);
				double innerMult = 
						xiSample.minus(data.selectRows(Ret.NEW,
								neighborIndicesMatrix.getAsInt(xi, i)))
						.mtimes(xiSample
								.minus(data.selectRows(Ret.NEW,
										neighborIndicesMatrix.getAsInt(xi, j)))
								.transpose()).getAsDouble(0, 0);

				tC.setAsDouble(1 / innerMult, i, j);
				tC.setAsDouble(1 / innerMult, j, i);
			}
		}
		return tC.sum(Ret.NEW, Matrix.COLUMN, false);
	}

	public Matrix getNeighborIndicesMatrix() {
		return neighborIndicesMatrix;
	}

	public void setNeighborIndicesMatrix(Matrix neighborIndicesMatrix) {
		this.neighborIndicesMatrix = neighborIndicesMatrix;
	}

	public Matrix getCoefficientMatrix() {
		return coefficientMatrix;
	}
	public static void main(String[] args) {
		double[][] testArray = new double[][]{
				{100, 5, 200},
				{1, 2, 4},
				{-200, -400, 0},
				{200, 400, -100}
		};
		Matrix m = Matrix.Factory.importFromArray(testArray);
		m = Matrix.Factory.randn(10, 10);
		System.out.println("Data: \r\n" + m);
		Matrix neighbors = Utils.kNeighborsIndicesMatrix(m, 3);
		System.out.println("Neighbor Index Matrix: " + neighbors);
		Lle l = new Lle(m, neighbors, 3);
		Matrix reducedData = l.reduceDimension();
		System.out.println(reducedData);
		double[] array = {5, 4, 2, 10, Double.NaN, 9, Double.NaN};
		Arrays.sort(array);
		System.out.println(Arrays.toString(array));
	}
}
