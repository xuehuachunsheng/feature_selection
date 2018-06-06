package ufs.general.algorithm;

import java.util.SortedSet;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.collections.composite.SortedListSet;

import ufs.utils.ConstValues;

/**
 * Least Angle Regressive algorithm. It solves Lasso problem as
 * 
 * <pre>
 * 	argmin_{x} ||y - Wx||_2^2 + ||x||_1,
 * </pre>
 * 
 * where y is standardize and W is standardize by column. <br>
 * 
 * Reference: Matrix Analysis and Applications (2nd Edition) P:386-389.
 * 
 * @author Yanxue
 *
 */
public class LARs {

	/**
	 * Input: The standardized observed column vector.
	 */
	Matrix y;

	/**
	 * Input: The column-standardized dictionary Matrix.
	 */
	Matrix W;

	/**
	 * Output: Regressive coefficient column vector.
	 */
	Matrix x;
	/**
	 * Iterative Parameter: Convergent value-Epsilon
	 */
	double convergeValue;

	/**
	 * Iterative Parameter: max number of iterative. If the number of iterative
	 * greater than this number, it stops.
	 */
	int maxIterativeNum;

	public LARs(Matrix pY, Matrix pW) {
		// the response vector has mean 0
		y = pY.standardize(Ret.NEW, 0);
		// The covariates is standardized to have mean 0 and unit length.
		W = pW.standardize(Ret.NEW, 0);
		defaultInitialize();
	}

	private void defaultInitialize() {
		// rho = ConstValues.RHO;
		convergeValue = ConstValues.CONVERG_VALUE;
		maxIterativeNum = ConstValues.MAX_ITE_NUM;
	}

	/**
	 * Obtain the x.
	 * 
	 * @return
	 */
	public Matrix getOptSolv() {

		int n = (int) W.getRowCount();
		int d = (int) W.getColumnCount();

		if (x != null) {
			return x;
		}

		Matrix x_k = Matrix.Factory.rand(d, 1);
		Matrix x_k_1 = Matrix.Factory.zeros(d, 1);
		Matrix y_k = Matrix.Factory.zeros(n, 1);
		Matrix dicW_k = W;
		Matrix c_k = null;
		SortedSet<Integer> labelSet = new SortedListSet<>();
		int tIteNum = 0;
		while (x_k.minus(x_k_1).normF() > convergeValue && tIteNum++ < maxIterativeNum) {
			x_k_1 = x_k;
			// 1) 2400*1
			c_k = W.transpose().mtimes(y.minus(y_k));
			// 2)
			double[] maxCorrCoeAndLabel = maxCorrelationCoefficient(c_k);
			double C = maxCorrCoeAndLabel[0];
			labelSet.add((int) maxCorrCoeAndLabel[1]);
			
			// 3)
			Integer[] labelArray = labelSet.toArray(new Integer[0]);
			//System.out.println("labelArray: " + Arrays.toString(labelArray));
			dicW_k = W.selectColumns(Ret.NEW, labelArray[0]).times(Math.signum(c_k.getAsDouble(labelArray[0], 0)));
			for (int i = 1; i < labelArray.length; i++) {
				dicW_k = dicW_k.appendHorizontally(Ret.LINK, W.selectColumns(Ret.NEW, labelArray[i]).times(Math.signum(c_k.getAsDouble(labelArray[i], 0))));
			}

			// 4)
			Matrix G_k = dicW_k.transpose().mtimes(dicW_k);
			if(G_k.isSingular()) {
				x = x_k;
				return x;
			}
			Matrix G_k_inv = G_k.inv();
			Matrix ones = Matrix.Factory.ones(labelSet.size(), 1);
			double a_k = 1 / Math.sqrt(ones.transpose().mtimes(G_k_inv)
					.mtimes(ones).getAsDouble(0, 0));
			Matrix omega_k = G_k_inv.mtimes(ones).times(a_k);
			Matrix u_k = dicW_k.mtimes(omega_k);

			// 5)
			Matrix eva_x = G_k.inv().mtimes(dicW_k.transpose()).mtimes(y);
			Matrix b = W.transpose().mtimes(u_k);

			// 5') 6')
			Matrix eva_x_plus = Matrix.Factory.zeros(d, 1);
			Matrix omega_k_plus = Matrix.Factory.zeros(d, 1);
			// Matrix b_plus = Matrix.Factory.zeros(d, 1);
			int tCount = 0;
			for (int i : labelSet) {
				eva_x_plus.setAsDouble(eva_x.getAsDouble(tCount, 0), i, 0);
				omega_k_plus.setAsDouble(
						omega_k.getAsDouble(tCount, 0)
								* Math.signum(c_k.getAsDouble(i, 0)), i, 0);
				// b_plus.setAsDouble(b.getAsDouble(tCount, 0), i, 0);
				tCount++;
			}
			x_k = eva_x_plus;
			
			// 6)
			double[] tArray = new double[(int) (2 * c_k.getRowCount())];
			for (int i = 0; i < c_k.getRowCount(); i++) {
				if (!labelSet.contains(i)) {
					tArray[2 * i] = (C - c_k.getAsDouble(i, 0))
							/ (a_k - b.getAsDouble(i, 0));
					tArray[2 * i + 1] = (C + c_k.getAsDouble(i, 0))
							/ (a_k + b.getAsDouble(i, 0));
					
				}
			}
			double[] hat_gama_and_j = min_plus_and_index(tArray);
			double hat_gama = hat_gama_and_j[0];
			int hat_j = (int) hat_gama_and_j[1] / 2; 
			
			
			tArray = new double[(int) c_k.getRowCount()];
			tCount = 0;
			for (int i : labelSet) {
				tArray[i] = -eva_x_plus.getAsDouble(i, 0)
						/ omega_k_plus.getAsDouble(i, 0);
			}

			double[] wave_gama_and_j = min_plus_and_index(tArray);
			double wave_gama = wave_gama_and_j[0];
			int wave_j = (int) wave_gama_and_j[1];
			
			// 7)
			if (wave_gama < hat_gama) {
				y_k = y_k.plus(u_k.times(wave_gama));
				labelSet.remove(wave_j);
			} else {
				y_k = y_k.plus(u_k.times(hat_gama));
				labelSet.add(hat_j);
			}
		}
		x = x_k;
		return x;
	}

	/**
	 * Compute the Maximum correlation coefficient (C) and indices. indices =
	 * [C, index1]
	 * 
	 * @param cVector
	 * @return
	 */
	private double[] maxCorrelationCoefficient(Matrix cVector) {
		//System.out.println(cVector.getRowCount() + " " + cVector.getColumnCount());
		double tC = Double.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < cVector.getRowCount(); i++) {
			if (tC < Math.abs(cVector.getAsDouble(i, 0))) {
				tC = Math.abs(cVector.getAsDouble(i, 0));
				index = i;
			}
		}

		return new double[] { tC, index };
	}

	/**
	 * min+ {`}
	 * 
	 * @param array
	 * @return
	 */
	private double[] min_plus_and_index(double[] array) {
		double tMin = Double.MAX_VALUE;
		int index = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > 0 && array[i] < tMin) {
				tMin = array[i];
				index = i;
			}
		}
		return new double[]{tMin, index};
	}
	public static void main(String[] args) {
		Matrix W = Matrix.Factory.randn(10, 50);
		Matrix Y = Matrix.Factory.randn(10, 1);
		LARs lars = new LARs(Y, W);
		System.out.println(lars.getOptSolv());
		System.out.println(W.sum(Ret.NEW, Matrix.ALL, false));
		System.out.println(W.mtimes(lars.getOptSolv()).minus(Y).sum(Ret.NEW, 0, false));
		
	}
}
