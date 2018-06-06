package ufs.featureselection.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.general.algorithm.LARs;
import ufs.utils.ConstValues;
import ufs.utils.Utils;
import ufs.utils.Utils.Order;

/**
 * The MCFS method's implementation. See detail in
 * "Multi-Cluster Feature Selection.".
 * 
 * @author Administrator
 *
 */
public class Mcfs extends UnsupervisedFeatureSelection {
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
	 */
	Matrix neighborIndicesMatrix;

	Matrix weightMatrix;

	/**
	 * Response Matrix. Each column is the response vector.
	 */
	Matrix responseMatrix;

	/**
	 * The coefficient matrix.
	 * 
	 * <pre>
	 * W = (W_1, W_2, ..., W_n)_{d \times n}, where W_i is the coefficient of Y_i response vector.
	 * </pre>
	 * 
	 */
	Matrix W;

	/**
	 * The weight matrix constant. weightMatrix[i][j] = exp(-||x_i - x_j||^2 /
	 * t)
	 */
	double t;

	public double getT() {
		return t;
	}

	public void setT(double t) {
		this.t = t;
	}

	public Mcfs(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNumFeatures);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
		defaultInitialize();
	}

	private void defaultInitialize() {
		t = ConstValues.LS_CONSTANT;
		rho = ConstValues.RHO;
		convergeValue = ConstValues.CONVERG_VALUE;
		maxIterativeNum = ConstValues.MAX_ITE_NUM;
	}

	public Matrix computeWeightMatrix() {
		int n = (int) data.getRowCount();
		Matrix tWeightMatrix = Matrix.Factory.zeros(n, n);
//		double sum = 0;
//		for (int i = 0; i < n; i++) {
//			for (int j = 0; j < neighborIndicesMatrix.getColumnCount(); j++) {
//				double tValue = data
//						.selectRows(Ret.LINK, i)
//						.minus(data.selectRows(Ret.LINK,
//								neighborIndicesMatrix.getAsInt(i, j))).normF();
////				tWeightMatrix.setAsDouble(Math.exp(-tValue * tValue / t), i,
////						neighborIndicesMatrix.getAsInt(i, j));
//				sum += tValue;
//			}
////			tWeightMatrix.setAsDouble(1, i, i);
//		}
//		t = sum / (n * neighborIndicesMatrix.getColumnCount());
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < neighborIndicesMatrix.getColumnCount(); j++) {
				double tValue = data
						.selectRows(Ret.LINK, i)
						.minus(data.selectRows(Ret.LINK,
								neighborIndicesMatrix.getAsInt(i, j))).normF();
				tWeightMatrix.setAsDouble(Math.exp(-tValue * tValue / t), i,
						neighborIndicesMatrix.getAsInt(i, j));
			}
			tWeightMatrix.setAsDouble(1, i, i);
		}
		weightMatrix = tWeightMatrix;
		
		return weightMatrix;
	}

	/**
	 * Use Laplacian Eigenmap to obtain the response matrix.
	 * 
	 * @return
	 */
	public Matrix computeY() {
		Matrix D = weightMatrix.mtimes(
				Matrix.Factory.ones(data.getRowCount(), 1)).diag(Ret.NEW);
//		System.out.println("OK1");
		
		responseMatrix = D.inv().mtimes(D.minus(weightMatrix)).eig()[0];
//		System.out.println("OK2");
		responseMatrix = Utils.centralize(responseMatrix, 0);
		return responseMatrix;
	}

	/**
	 * By using LARs algorithm to solve the Lasso problem.
	 * 
	 * @param X
	 *            X鈭圧^(n脳d), n is the number of samples, d is the number of
	 *            features. The original data.
	 * @param k
	 *            k is the number of selected features.
	 * @param rho
	 *            rho 鈭� R+.
	 * @return The indices of selected features.
	 */
	public Matrix optimalW() {

		long n = data.getRowCount();

		Matrix W = Matrix.Factory.emptyMatrix();

		Matrix stdData = data.standardize(Ret.NEW, 0);

		for (int i = 0; i < n; i++) {
			W = W.appendHorizontally(Ret.LINK,
					new LARs(responseMatrix.selectColumns(Ret.NEW, i), stdData)
							.getOptSolv());
			System.out.println(i+"-th for: ");
		}
		this.W = W;

		return W;
	}

	public int[] computeFeatureRanking() {
		int n = (int) data.getRowCount();
		int d = (int) data.getColumnCount();

		double[] score = new double[d];

		for (int i = 0; i < d; i++) {
			double tMaxValue = Double.MIN_VALUE;
			for (int j = 0; j < n; j++) {
				double tValue = W.getAsDouble(i, j);
				if (Math.abs(tValue) > tMaxValue) {
					tMaxValue = Math.abs(tValue);
				}
			}
			score[i] = Math.abs(tMaxValue);
		}
		featureSubset = ufs.utils.Utils.argSort(score, Order.DESC);
		return featureSubset;
	}

	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays
				.copyOf(featureSubset, numFeatures)));
	}

	@Override
	public void middleProcess() {
		computeWeightMatrix();
		System.out.println("WeightMatrix");
		computeY();
		System.out.println("Y");
		optimalW();
		computeFeatureRanking();
	}
}
