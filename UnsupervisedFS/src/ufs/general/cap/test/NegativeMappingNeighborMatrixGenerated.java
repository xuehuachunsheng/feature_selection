package ufs.general.cap.test;

import java.io.File;

import org.ujmp.core.Matrix;

import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class NegativeMappingNeighborMatrixGenerated {
	public static void main(String[] args) throws Exception {
		System.out.println("Cap Centrosymmetric mapping Neighbor Computing..");
		for(String dataSet : ConstValues.DATA_MATRIX) {
			String matDataSet = ConstValues.DATA_MATRIX_PATH + dataSet;
			Matrix X = Utils.loadMatrix2DFromMat(new File(matDataSet), "X");
			int numNeighbors = (int) X.getRowCount();
			String neighborFileName = "src/data/mat/cap/centromapping/neighbors/"
					+ dataSet.subSequence(0, dataSet.length() - 4) + "_Neighbors"
					+ numNeighbors + ".data";
			System.out.println(dataSet);
			Utils.writeMatrixToTxt(Utils.kNeighborsIndicesMatrix(X,
					Utils.negativeMatrix(X), numNeighbors), neighborFileName, " ",
					Integer.class);
			System.out.println("Over");
		}
		

	}
}
