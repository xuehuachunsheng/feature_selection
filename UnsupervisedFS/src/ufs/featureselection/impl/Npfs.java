package ufs.featureselection.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.SparseDoubleMatrix;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

/**
 * The basic implementation of Neighborhood Preserving Unsupervised Feature
 * Selection (NPFS).
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class Npfs extends UnsupervisedFeatureSelection {

	/**
	 * Iterative Parameter: The Coefficient of exterior penalty function. Rho.
	 */
	double rho;

	/**
	 * Iterative Parameter: Convergent value-Epsilon
	 */
	double convergeValue;

	/**
	 * Iterative Parameter: max number of iterative. If the number of iterative
	 * greater than this number, it stops.
	 */
	int maxIterativeNum;

	/**
	 * The neighbor indices matrix. i-th row of this matrix is the i-th samples'
	 * neighbor indices in the original sample matrix.
	 * 
	 */
	Matrix neighborIndicesMatrix;

	/**
	 * The coefficient matrix.
	 * 
	 * <pre>
	 * W = (W_1, W_2, ..., W_n)_{n \times n}, where W_i is the coefficient of X_i.
	 * </pre>
	 * 
	 * W_i is reconstructed by X_i's neighbors coefficient.
	 */
	Matrix W;

	public Npfs(Matrix data, Matrix pNeighborIndicesMatrix, int numFeatures) {
		super(data, numFeatures);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
		defaultInitialize();
	}

	private void defaultInitialize() {
		rho = ConstValues.RHO;
		convergeValue = ConstValues.CONVERG_VALUE;
		maxIterativeNum = ConstValues.MAX_ITE_NUM;
	}

	/**
	 * Obtain the optimal coefficient matrix W.
	 */
	public Matrix optimalW() {
		long[][] neighborsIndices = neighborIndicesMatrix.toLongArray();
		int numNeighbors = neighborsIndices[0].length;
		long n = data.getRowCount();
		Matrix rho_E_k = Matrix.Factory.eye(numNeighbors, numNeighbors).times(
				rho);
		W = SparseDoubleMatrix.Factory.zeros(n, n);
		for (int i = 0; i < n; i++) {
			// numNeighbors * d
			Matrix X_ = data.selectRows(Ret.NEW, neighborsIndices[i]);
			// numNeighbors * numNeighbors
			Matrix X_X = X_.mtimes(X_.transpose());
			// numNeighbors * d d * 1
			Matrix X_Xi = X_.mtimes(data.selectRows(Ret.NEW, i).transpose());
			// numNeighbors * 1
			Matrix Z = Matrix.Factory.rand(numNeighbors, 1);
			// System.out.println(Z);
			// numNeighbors * 1
			Matrix Y = Matrix.Factory.rand(numNeighbors, 1);
			// numNeighbors * 1
			Matrix Vi = Matrix.Factory.rand(numNeighbors, 1);
			Matrix V_i_1 = Matrix.Factory.zeros(numNeighbors, 1);
			int counter = 0;
			while (Vi.minus(V_i_1).normF() > convergeValue
					&& counter < maxIterativeNum) {
				V_i_1 = Vi;
				
				Vi = Z.minus(Y.times(1 / rho)).plus(1.0 / numNeighbors)
						.plus(Y.times(1 / rho).minus(Z).getMeanValue());

				// a+n
				for (int j2 = 0; j2 < numNeighbors; j2++) {
					if (Vi.getAsDouble(j2, 0) < 0) {
						Vi.setAsDouble(0, j2, 0);
					}
				}
				
				Z = X_X.plus(rho_E_k).inv()
						.mtimes(X_Xi.plus(Vi.times(rho)).plus(Y));
				for (int j2 = 0; j2 < numNeighbors; j2++) {
					if (Z.getAsDouble(j2, 0) < 0) {
						Z.setAsDouble(0, j2, 0);
					}
				}
				
				// System.out.println("Z: \r\n" + Z);
				Y = Y.plus(Vi.minus(Z).times(rho));
				// System.out.println("Y: \r\n" + Y);
				counter++;
			}
			for (int j = 0; j < neighborsIndices[i].length; j++) {
				W.setAsDouble(Vi.getAsDouble(j, 0), neighborsIndices[i][j], i);
			}
		}
		return W;
	}

	public int[] computeFeatureRanking() {
		long n = data.getRowCount();
		long d = data.getColumnCount();
		// n*n
		Matrix E_n_n_subW_T = SparseDoubleMatrix.Factory.eye(n, n).minus(W)
				.transpose();
		double[] normValue = new double[(int) d];
		for (int i = 0; i < d; i++) {
			normValue[i] = E_n_n_subW_T.mtimes(data.selectColumns(Ret.LINK, i))
					.normF();
		}
		featureSubset = Utils.argSort(normValue, Utils.Order.ASC, (int) d);
		return featureSubset;
	}

	@Override
	public int[] getFeatureSubset() {
		return Arrays.copyOf(featureSubset, numFeatures);
	}

	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays
				.copyOf(featureSubset, numFeatures)));
	}

	public double getRho() {
		return rho;
	}

	public void setRho(double rho) {
		this.rho = rho;
	}

	public double getConvergeValue() {
		return convergeValue;
	}

	public void setConvergeValue(double convergeValue) {
		this.convergeValue = convergeValue;
	}

	public int getMaxIterativeNum() {
		return maxIterativeNum;
	}

	public void setMaxIterativeNum(int maxIterativeNum) {
		this.maxIterativeNum = maxIterativeNum;
	}

	public Matrix getNeighborIndicesMatrix() {
		return neighborIndicesMatrix;
	}

	public void setNeighborIndicesMatrix(Matrix neighborIndicesMatrix) {
		this.neighborIndicesMatrix = neighborIndicesMatrix;
	}

	public Matrix getW() {
		return W;
	}

	@Override
	public void middleProcess() {
		optimalW();
		computeFeatureRanking();
	}
}
