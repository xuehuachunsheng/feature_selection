package ufs.featureselection.impl;
import java.io.RandomAccessFile;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.general.algorithm.LARs;
import ufs.utils.ConstValues;
import ufs.utils.Utils;
import ufs.utils.Utils.Order;

import java.io.File;

/**
 * The effective distance-based Laplacian Score method implementation. See
 * "Feature selection with efective distance" for more detail. The dataset is
 * not standardized.
 * 
 * @author Yanxue
 *
 */
public class Edls extends UnsupervisedFeatureSelection {

	/**
	 * The sparse coefficient matrix.
	 */
	protected Matrix coefficientMatrix;

	/**
	 * The weight matrix of the graph models the local structure of the data
	 * space.
	 */
	protected Matrix weightMatrix;

	/**
	 * The weight matrix constant. weightMatrix[i][j] = exp(-||x_i - x_j||^2 /
	 * t). 
	 */
	protected double t;

	/**
	 * Penalty factor.
	 */
	protected double rho;

	/**
	 * The max iterative number for computing coefficient matrix.
	 */
	protected double maxIterativeNum;

	/**
	 * The effective distance-based Laplacian score of each features.
	 */
	protected double[] edlsScore;

	public Edls(Matrix pData, int pNumFeatures) {
		super(pData, pNumFeatures);
		defaultInitialize();
	}

	protected void defaultInitialize() {
		t = ConstValues.LS_CONSTANT;
		rho = ConstValues.RHO;
		maxIterativeNum = ConstValues.MAX_ITE_NUM;
	}

	/**
	 * ADMM method in SPFS
	 * 
	 * @return
	 */
	/**
	 * public Matrix computeCoefficientMatrix() { long n = data.getRowCount();
	 * long d = data.getColumnCount();
	 * 
	 * Matrix rho_E_nsub1 = Matrix.Factory.eye(n - 1, n - 1).times(rho); Matrix
	 * W = Matrix.Factory.emptyMatrix();
	 * 
	 * for (int i = 0; i < n; i++) { // long c_time =
	 * System.currentTimeMillis(); System.out.println(i + "th for: "); //
	 * Initial assignment Matrix Z = Matrix.Factory.randn(n - 1, 1); Matrix Y =
	 * Matrix.Factory.randn(n - 1, 1); Matrix X_subi = data.deleteRows(Ret.NEW,
	 * i);
	 * 
	 * // X_(-i)^T * X_(-i) 1.5s time costs. // 10_0000_0000 basic multiply //
	 * operations. // This sentences is the Performance Bottle-Neck //
	 * System.out.println(X_subi.getColumnCount() + " " + //
	 * X_subi.getRowCount()); Matrix X_subi_subi =
	 * X_subi.mtimes(X_subi.transpose(Ret.LINK)); // X_(-i)^T * X_i
	 * 
	 * Matrix X_subi_i = X_subi.mtimes(data.selectRows(Ret.LINK, i)
	 * .transpose(Ret.LINK)); // System.out.println("Compute X_subi_subi,
	 * X_subi_i: " // + (System.currentTimeMillis() - c_time) + "ms"); Matrix Vi
	 * = Matrix.Factory.randn(n - 1, 1); Matrix V_i_1 = null; // Iterating
	 * process int counter = 0; do { V_i_1 = Vi; Vi =
	 * X_subi_subi.plus(rho_E_nsub1).inv()
	 * .mtimes(X_subi_i.plus(Z.times(rho).minus(Y))); Z =
	 * S_lambda(Vi.plus(Y.times(1.0 / rho)), 1.0 / rho); Y =
	 * Y.plus(Vi.minus(Z).times(rho)); counter++; } while
	 * (V_i_1.minus(Vi).normF() > ConstValues.CONVERG_VALUE && counter <
	 * ConstValues.MAX_ITE_NUM);
	 * 
	 * // Reconstruct Wi by Vi Matrix tWi = Matrix.Factory.zeros(n, 1); for (int
	 * j = 0; j < i; j++) { tWi.setAsDouble(Vi.getAsDouble(j, 0), j, 0); }
	 * tWi.setAsDouble(0, i, 0); for (int j = i; j < n - 1; j++) {
	 * tWi.setAsDouble(Vi.getAsDouble(j, 0), j + 1, 0); } // Reconstruct W
	 * matrix // System.out.println(tWi); W = W.appendHorizontally(Ret.LINK,
	 * tWi); // System.out.println("NumIterative: " + counter); //
	 * System.out.println((System.currentTimeMillis() - c_time) + "ms"); }
	 * coefficientMatrix = W; System.out.println(coefficientMatrix); return W; }
	 */

	/**
	 * Solving this problem by using Lars method.
	 * 
	 * @return
	 */
	public Matrix computeCoefficientMatrix() {
		int n = (int) data.getRowCount();
		int m = (int) data.getColumnCount();
		Matrix data_T = data.transpose();
		Matrix P = Matrix.Factory.emptyMatrix();
		for (int i = 0; i < n; i++) {
			
			// By using a broaden Matrix to solve the sum-to-one constrint. 
			// _xi_ = [xi^T, 1]^T
			Matrix _xi_ = data.selectRows(Ret.LINK, i).appendHorizontally(Ret.LINK, Matrix.Factory.ones(1, 1)).transpose(Ret.NEW);

			// A = [X, 1^T]^T
			Matrix A = data_T.deleteColumns(Ret.LINK, i).appendVertically(Ret.LINK, Matrix.Factory.ones(1, n - 1));
			
			LARs lars = new LARs(_xi_, A);
			Matrix pi = lars.getOptSolv();
			Matrix _pi_ = Matrix.Factory.zeros(n, 1);
			for (int j = 0; j < i; j++) {
				_pi_.setAsDouble(pi.getAsDouble(j, 0), j, 0);
			}
			_pi_.setAsDouble(0, i, 0);
			for (int j = i + 1; j < n; j++) {
				_pi_.setAsDouble(pi.getAsDouble(j - 1, 0), j, 0);
			}
			P = P.appendHorizontally(Ret.LINK, _pi_);
//			System.out.println(i + "-th for: ");
		}

		coefficientMatrix = P;

		return P;
	}

	/**
	 * Compute S_lambda value by given column vector.
	 * 
	 * @param pColumnVector
	 *            The given column vector.
	 * @param lambda
	 *            the value of lambda.
	 * @return The function value of S_lambda.
	 */
	public static Matrix S_lambda(Matrix pColumnVector, double lambda) {
		Matrix resultMatrix = Matrix.Factory.zeros(pColumnVector.getRowCount(), 1);
		for (int i = 0; i < pColumnVector.getRowCount(); i++) {
			if (pColumnVector.getAsDouble(i, 0) > lambda) {
				resultMatrix.setAsDouble(pColumnVector.getAsDouble(i, 0) - lambda, i, 0);
			} else if (pColumnVector.getAsDouble(i, 0) < -lambda) {
				resultMatrix.setAsDouble(pColumnVector.getAsDouble(i, 0) + lambda, i, 0);
			}
			// else 0
		}
		return resultMatrix;
	}

	/**
	 * Compute the ES matrix.
	 * 
	 * @return
	 */
	public Matrix computeWeightMatrix() {
		long n = data.getRowCount();
		
		Matrix normCoefficientMatrix = coefficientMatrix.normalize(Ret.NEW, 0);

		//
		Matrix ES = Matrix.Factory.zeros(n, n);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double pij = normCoefficientMatrix.getAsDouble(i, j);
				if (pij > 1e-6) {
					double temp = 1 - Math.log(pij);
					ES.setAsDouble(Math.pow(Math.E, -temp * temp / t), i, j);
				}
			}
		}
		// To avoid the log 0, we modify the pi to pi+0.01
//		try {
//			RandomAccessFile raf = new RandomAccessFile("src/data/test/test.txt", "rw");
//			raf.writeBytes(ES.toString());
//			raf.close();
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
		weightMatrix = ES;
		return ES;
	}

	public void modifyBandwidthParam() {
		long n = data.getRowCount();
		Matrix normCoefficientMatrix = coefficientMatrix.normalize(Ret.NEW, 0);

		int numNonzero = 0;
		double sum = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double pij = normCoefficientMatrix.getAsDouble(i, j);
				if (pij > 1e-6) {
					numNonzero++;
					sum += 1 - Math.log(pij);
				}
			}
		}
		t = sum / numNonzero;
	}

	public double[] computeEdls() {
		int n = (int) data.getRowCount();
		int d = (int) data.getColumnCount();
		Matrix unitColumnVector = Matrix.Factory.ones(n, 1);
		Matrix DVector = weightMatrix.mtimes(unitColumnVector);
		double dSum = DVector.sum(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);
		Matrix D = DVector.diag(Ret.NEW);
		Matrix L = D.minus(weightMatrix);
		double[] tLaplacianScore = new double[d];
		for (int i = 0; i < data.getColumnCount(); i++) {
			Matrix fr = data.selectColumns(Ret.NEW, i);
			Matrix _fr_ = fr.minus(unitColumnVector.times(fr.transpose().mtimes(DVector).getAsDouble(0, 0) / dSum));
			tLaplacianScore[i] = _fr_.transpose().mtimes(L).mtimes(_fr_).getAsDouble(0, 0)
					/ _fr_.transpose().mtimes(D).mtimes(_fr_).getAsDouble(0, 0);
		}
		edlsScore = tLaplacianScore;
		return edlsScore;
	}

	/**
	 * The features with smaller laplacian score are more important.
	 */
	public int[] computeFeatureRanking() {
		featureSubset = Utils.argSort(edlsScore, Order.ASC);
		return featureSubset;
	}

	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays.copyOf(featureSubset, numFeatures)));
	}

	@Override
	public int[] getFeatureSubset() {
		return Arrays.copyOf(featureSubset, numFeatures);
	}

	public double getT() {
		return t;
	}

	public void setT(double t) {
		this.t = t;
	}

	public Matrix getWeightMatrix() {
		return weightMatrix;
	}

	public double[] getEdls() {
		return edlsScore;
	}

	public static void main(String[] args) throws Exception {

		Matrix m = Utils.loadMatrix2DFromMat(
				new File(ConstValues.DATA_MATRIX_PATH + ConstValues.STD_WARPAR_MATRIX_130$2400$10), "X");

		Edls ls = new Edls(m, 20);
		ls.middleProcess();
		System.out.println(Arrays.toString(ls.getFeatureSubset()));
	}

	@Override
	public void middleProcess() {
		computeCoefficientMatrix();
		modifyBandwidthParam();
		computeWeightMatrix();
		computeEdls();
		computeFeatureRanking();
	}

}
