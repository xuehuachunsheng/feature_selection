package ufs.cluster.evaluate.impl;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;


/**
 * The basic implementation of accuracy, which is solved by KM (Hungary)
 * algorithm.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class Accuracy extends Purity {

	public Accuracy(Matrix pData, int[] predictLabels, int[] realLabels) {
		super(pData, predictLabels, realLabels);
		coefficientMatrixGenerated();
	}

	@Override
	public double precision() {
		Matrix M = Matrix.Factory.fill(
				coefficientMatrix.max(Ret.NEW, Matrix.ALL).getAsInt(0, 0),
				coefficientMatrix.getRowCount(),
				coefficientMatrix.getColumnCount());
		Matrix tMatrix = M.minus(coefficientMatrix);
//		System.out.println(coefficientMatrix);
		Hungary h = new Hungary(tMatrix);
		h.findMinMatch();
		int[] mapIndices = h.mapIndices;
		double sum = 0;
		for (int i = 0; i < mapIndices.length && i < tMatrix.getRowCount(); i++) {
			if(mapIndices[i] >= tMatrix.getColumnCount() || mapIndices[i] == -1) {
				continue;
			}
//			System.out.println(coefficientMatrix.getAsDouble(i, mapIndices[i]));
			sum += coefficientMatrix.getAsDouble(i, mapIndices[i]);
		}
		return sum / data.getRowCount();
	}
	
}
