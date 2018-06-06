package ufs.featureselection.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.SparseDoubleMatrix;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

/**
 * The basic implementation of Centrosymmetric mapping Neighborhood Preserving
 * Unsupervised Feature Selection (CAP).
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 */
public class Cap extends UnsupervisedFeatureSelection {

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
	 * The data after centralized.
	 */
	Matrix centralizedData;

	/**
	 * The centrosymmetric mapping data matrix.
	 */
	Matrix centrosymCentralizedData;

	/**
	 * The neighbor indices matrix. i-th row of this matrix is the i-th sample
	 * centrosymmetric's neighbor indices in the original sample matrix.
	 * 
	 */
	Matrix centroNeighborIndicesMatrix;

	/**
	 * The coefficient matrix.
	 * 
	 * <pre>
	 * W = (W_1, W_2, ..., W_n)_{n \times n}, where W_i is the coefficient of X_i.
	 * </pre>
	 * 
	 * W_i is reconstructed by X_i's centromapping's neighbors coefficient.
	 */
	Matrix W;

	public Matrix getW() {
		return W;
	}

	/**
	 * The average fitting error.
	 */
	double avgFittingError;

	public double getAvgFittingError() {
		return avgFittingError;
	}

	public Cap(Matrix data, Matrix pCentroNeighborIndicesMatrix,
			int pNumFeatures) {
		super(data, pNumFeatures);
		centroNeighborIndicesMatrix = pCentroNeighborIndicesMatrix;
		defaultInitialize();
	}

	private void defaultInitialize() {
		rho = ConstValues.RHO;
		convergeValue = ConstValues.CONVERG_VALUE;
		maxIterativeNum = ConstValues.MAX_ITE_NUM;
	}

	public Matrix centralize() {
		centralizedData = Utils.centralize(data, 0);
		return centralizedData;
	}

	public Matrix centrosymmetricMapping() {
		int n = (int) centralizedData.getRowCount();
		int m = (int) centralizedData.getColumnCount();
		Matrix negX = Matrix.Factory.zeros(n, m);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				negX.setAsDouble(-centralizedData.getAsDouble(i, j), i, j);
			}
		}

		centrosymCentralizedData = negX;
		return negX;
	}

	public Matrix optimalW() {
		long[][] neighborsIndices = centroNeighborIndicesMatrix.toLongArray();
		int numNeighbors = neighborsIndices[0].length;
		long n = data.getRowCount();
		Matrix rho_E_k = Matrix.Factory.eye(numNeighbors, numNeighbors).times(
				rho);
		// [1/k, 1/k, 1/k, ..., 1/k]
		Matrix unitColumnVector = Matrix.Factory.fill(1.0 / numNeighbors,
				numNeighbors, 1);
		W = SparseDoubleMatrix.Factory.zeros(n, n);
		double tSum = 0;
		for (int i = 0; i < n; i++) {
			// numNeighbors * d
			Matrix X_ = centralizedData
					.selectRows(Ret.NEW, neighborsIndices[i]);
			// numNeighbors * numNeighbors
			Matrix X_X = X_.mtimes(X_.transpose());
			// numNeighbors * d d * 1
			Matrix X_Xi = X_.mtimes(centrosymCentralizedData.selectRows(
					Ret.LINK, i).transpose(Ret.NEW));
			// numNeighbors * 1
			Matrix Z = Matrix.Factory.randn(numNeighbors, 1);
			// numNeighbors * 1
			Matrix Y = Matrix.Factory.randn(numNeighbors, 1);
			// numNeighbors * 1
			Matrix Vi = Matrix.Factory.zeros(numNeighbors, 1);
			Matrix V_i_1 = Matrix.Factory.randn(numNeighbors, 1);
			int counter = 0;
			while (Vi.minus(V_i_1).normF() > convergeValue
					&& counter < maxIterativeNum) {
				// Update Vi (numFeatures, 1)
				V_i_1 = Vi;
				Vi = Z.minus(Y.times(1 / rho)).plus(1.0 / numNeighbors)
						.plus(Y.times(1 / rho).minus(Z).getMeanValue());
				// System.out.println("V_"+counter+": " + Vi);
				// a+
				for (int j2 = 0; j2 < numNeighbors; j2++) {
					if (Vi.getAsDouble(j2, 0) < 0) {
						Vi.setAsDouble(0, j2, 0);
					}
				}
				// System.out.println("V_"+counter+"+: " + Vi + "");
				Z = X_X.plus(rho_E_k).inv()
						.mtimes(X_Xi.plus(Vi.times(rho)).plus(Y));
				for (int j2 = 0; j2 < numNeighbors; j2++) {
					if (Z.getAsDouble(j2, 0) < 0) {
						Z.setAsDouble(0, j2, 0);
					}
				}
				// System.out.println("Z_"+counter+": " + Z + "");
				Y = Y.plus(Vi.minus(Z).times(rho));
				// System.out.println("Y_"+counter+"+: " + Y + "");
				counter++;
			}
			;
			tSum += centrosymCentralizedData.selectRows(Ret.NEW, i).transpose()
					.minus(X_.transpose().mtimes(Vi)).normF();

			for (int j = 0; j < neighborsIndices[i].length; j++) {
				W.setAsDouble(Vi.getAsDouble(j, 0), neighborsIndices[i][j], i);
			}
		}
		avgFittingError = tSum / n;
		return W;
	}

	public int[] computeFeatureRanking() {
		int n = (int) centralizedData.getRowCount();
		int d = (int) centralizedData.getColumnCount();
		Matrix E_n_n_plusW_T = SparseDoubleMatrix.Factory.eye(n, n).plus(W)
				.transpose();
		double[] normValue = new double[(int) d];
		for (int i = 0; i < d; i++) {
			normValue[i] = E_n_n_plusW_T.mtimes(
					centralizedData.selectColumns(Ret.LINK, i)).normF();
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

	@Override
	public void middleProcess() {
		centralize();
		centrosymmetricMapping();
		optimalW();
		computeFeatureRanking();
	}
}
