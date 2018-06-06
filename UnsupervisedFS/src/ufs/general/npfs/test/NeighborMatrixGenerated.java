package ufs.general.npfs.test;

import java.io.File;
import org.ujmp.core.Matrix;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class NeighborMatrixGenerated {
	public static void main(String[] args) throws Exception {

		System.out.println("NPFS Neighbor Computing..");
		
		for (int i = 5; i < ConstValues.DATA_MATRIX.length; i++) {
			
			String dataset = ConstValues.DATA_MATRIX[i];
//			System.out.println(dataset);
			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			int numNeighbors = (int) X.getRowCount() - 1;
			String neighborFileName = "src/data/mat/npfs/neighbors/" + dataset.substring(0, dataset.length() - 4)
					+ "_Neighbors" + numNeighbors + ".data";
//			Utils.writeMatrixToTxt(Utils.kNeighborsIndicesMatrix(X, numNeighbors), neighborFileName, " ",
//					Integer.class);
//			System.out.println("Over");

			long c_time = System.currentTimeMillis();
			Utils.kNeighborsIndicesMatrix(X, numNeighbors);
			System.out.print((System.currentTimeMillis() - c_time)+"\t");
		}
		
	}
}
