package ufs.general.test;

import java.io.File;

import org.ujmp.core.Matrix;

import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class NeighborMatrixGenerated {
	public static void main(String[] args) throws Exception {
		String dataset = ConstValues.GLIOMA_MATRIX_50$4434$4;
		
		Matrix X = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "X");
		
		X = Utils.centralize(X, 0);
		
		Matrix negX = Utils.negativeMatrix(X);
		
		Utils.writeMatrixToTxt(Utils.kNeighborsIndicesMatrix(X, negX, (int)X.getRowCount()), "src/data/mat/cap/centromapping/neighbors/"+dataset.substring(0, dataset.length()-4)+"_Neighbors"+X.getRowCount()+".data", " ", Integer.class);
		
	}
}
