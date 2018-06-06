package ufs.featureselection.impl;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.reducedim.impl.Lle;
import ufs.utils.ConstValues;
import ufs.utils.Utils;


/**
 * Nonnegative Neighborhood preserving effective distance-based Feature selection.
 * @author Yanxue
 *
 */
public class NNEdls extends Edls {

	/**
	 * R^n×k, where n is the number of samples and k is the number of neighbors.
	 * The element located in i-th row and j-th column stores the index of the
	 * j-th neighbor of the i-th sample.
	 * 
	 */
	protected Matrix neighborIndicesMatrix;
	
	public NNEdls(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNumFeatures);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
	}
	
	@Override
	public Matrix computeCoefficientMatrix() {
		Npfs npfs = new Npfs(data, neighborIndicesMatrix, 0);
		coefficientMatrix = npfs.optimalW();
		// Diagonal to 1
		for (int i = 0; i < coefficientMatrix.getRowCount(); i++) {
			coefficientMatrix.setAsDouble(1,  i, i);
		}
		return coefficientMatrix;
	}
	
	/**
	 * Compute the ES matrix
	 * 
	 * @return
	 */
	@Override
	public Matrix computeWeightMatrix() {
		long n = data.getRowCount();
		long d = data.getColumnCount();

		//
		Matrix ES = Matrix.Factory.zeros(n, n);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double pij = coefficientMatrix.getAsDouble(i, j);
				if (pij > 1e-6) {
					double temp = 1 - Math.log(pij);
					ES.setAsDouble(Math.pow(Math.E, -temp * temp / t), i, j);
				}
			}
			// Diagonal to 1
			ES.setAsDouble(1, i, i);
		}
		
		// To avoid the log 0, we modify the pi to pi+0.01
		try {
			RandomAccessFile raf = new RandomAccessFile(
					"src/data/test/test.txt", "rw");
			raf.writeBytes(ES.toString());
			raf.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		weightMatrix = ES;
		return ES;
	}
	
	@Override
	public void modifyBandwidthParam() {
		long n = data.getRowCount();

		int numNonzero = 0;
		double sum = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double pij = coefficientMatrix.getAsDouble(i, j);
				if (pij > 1e-6) {
					numNonzero++;
					sum += 1 - Math.log(pij);
				}
			}
		}
		t = sum / numNonzero;
	}
	
	public static void main(String[] args) throws Exception {

		Matrix m = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH
						+ ConstValues.STD_WARPAR_MATRIX_130$2400$10), "X");
		Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/WarpAR10P_std_Neighbors129.data"), " ", Integer.class);
		int numNeighbors = 9;
		long[] selectNeighbors = new long[numNeighbors];
		for (int i = 0; i < selectNeighbors.length; i++) {
			selectNeighbors[i] = i;
		}
		NNEdls ls = new NNEdls(m, neighbors.selectColumns(Ret.NEW, selectNeighbors), 20);
		ls.middleProcess();
		System.out.println(Arrays.toString(ls.getEdls()));
		System.out.println(Arrays.toString(ls.getFeatureSubset()));

	}
}
