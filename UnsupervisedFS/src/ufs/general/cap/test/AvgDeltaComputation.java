package ufs.general.cap.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * Compute the average error of 100 times experiments.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 */
public class AvgDeltaComputation {
	public static void main(String[] args) {
		
		int numExperiment = 100;

		Matrix[] resMat = new Matrix[numExperiment];
		for (int i = 0; i < numExperiment; i++) {
			resMat[i] = ufs.utils.Utils.load2DMatrixFromTxt(
					new File("src/data/mat/cap/kanddelta/NNPFS_K_Delta_" + i
							+ "th.data"), " ", Double.class);
		}

		int numData = (int) resMat[0].getRowCount();
		Matrix[] reconstructMatrix = new Matrix[numData];
		for (int i = 0; i < numData; i++) {
			reconstructMatrix[i] = Matrix.Factory.emptyMatrix();
			for (int j = 0; j < numExperiment; j++) {
				reconstructMatrix[i] = reconstructMatrix[i].appendVertically(
						Ret.NEW, resMat[j].selectRows(Ret.NEW, i));
			}
		}

		Matrix avgDelta = Matrix.Factory.emptyMatrix();
		Matrix floatRange = Matrix.Factory.emptyMatrix();

		for (int i = 0; i < numData; i++) {
			avgDelta = avgDelta.appendVertically(Ret.NEW,
					reconstructMatrix[i].mean(Ret.NEW, 0, false));
			Matrix minX = reconstructMatrix[i].min(Ret.NEW, 0);
			Matrix maxX = reconstructMatrix[i].max(Ret.NEW, 0);
			floatRange = floatRange.appendVertically(Ret.NEW, maxX.minus(minX)
					.divide(2));
		}
		System.out.println(avgDelta.normalize(Ret.NEW, Matrix.COLUMN));
		System.out.println(avgDelta);
		
	}
}
