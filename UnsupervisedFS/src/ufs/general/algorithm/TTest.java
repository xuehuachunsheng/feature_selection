package ufs.general.algorithm;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * The T-Test basic implementation.
 * 
 * @author wuyanxue
 *
 */
public class TTest {

	Matrix a;

	Matrix b;

	/**
	 * Input two column vectors as two groups of samples to construct the TTest
	 * object.
	 * 
	 * @param a
	 *            A column vector. One of group of samples.
	 * @param b
	 *            A column vector. Another group of samples.
	 */
	public TTest(Matrix pA, Matrix pB) {
		a = pA;
		b = pB;
	}

	public double independentTTest() {
		double meanA = a.mean(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);
		double meanB = b.mean(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);

		double varA = a.var(Ret.NEW, Matrix.ALL, false, false).getAsDouble(0, 0);
		double varB = b.var(Ret.NEW, Matrix.ALL, false, false).getAsDouble(0, 0);

		double sizeA = a.getRowCount();
		double sizeB = b.getRowCount();

		return (meanA - meanB)
				/ Math.sqrt((1 / sizeA + 1 / sizeB) * ((sizeA - 1) * varA * varA + (sizeB - 1) * varB * varB
				) / (sizeA + sizeB - 2));
	}
	
	public double pairedTTest() {
		return a.appendHorizontally(Ret.NEW, b).pairedTTest(Ret.NEW).getAsDouble(0, 1);
	}

	public static void main(String[] args) {
		Matrix matrix = Matrix.Factory.rand(5, 1);
		Matrix matrix2 = Matrix.Factory.rand(5, 1);
		System.out.println(matrix.appendHorizontally(Ret.NEW, matrix2).pairedTTest(Ret.NEW));
	}
}
