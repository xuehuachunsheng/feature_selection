package ufs.general.cap.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.impl.Cap;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class AvgErrorComputed {
	public static void main(String[] args) throws Exception {
		String dataset = ConstValues.CARCINOM_MATRIX_174$9182$11;

		Matrix X = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "X");
		Matrix Y = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "Y");
		Matrix centroNeighbors = Utils.load2DMatrixFromTxt(
				new File("src/data/mat/cap/centromapping/neighbors/"
						+ dataset.substring(0, dataset.length() - 4)
						+ "_Neighbors" + X.getRowCount() + ".data"), " ",
				Integer.class);
		Matrix errorMatrix = Matrix.Factory.zeros(18, 1);
		for (int k = 3; k <= 20; k++) {
			long[] numNeighbors = new long[k];
			for (int i = 0; i < numNeighbors.length; i++) {
				numNeighbors[i] = i;
			}
			double tSum = 0;
			for (int i = 0; i < 20; i++) {

				Cap cap = new Cap(X, centroNeighbors.selectColumns(Ret.NEW, numNeighbors), 0);
				cap.centralize();
				cap.centrosymmetricMapping();
				cap.optimalW();
				
				tSum += cap.getAvgFittingError();
			}
			errorMatrix.setAsDouble(tSum / 20, k - 3, 0);
			System.out.print(tSum / 20 + "\t");
		}
		System.out.println("\r\n" + errorMatrix.normalize(Ret.NEW, Matrix.ROW));

	}
}
