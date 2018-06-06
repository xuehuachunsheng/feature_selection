package ufs.general.cap.test;

import java.io.File;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.jmatio.ImportMatrixMAT;

import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class Test {
	public static void main(String[] args) throws Exception {
		String dataset = ConstValues.DATA_MATRIX[2];

		Matrix X = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "X");
		Matrix Y = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "Y");
		Matrix features = Utils.load2DMatrixFromTxt(
				new File("src/data/mat/npfs/featureranking/"
						+ dataset.substring(0, dataset.length() - 4)
						+ "_Neighbors" + 9
						+ "rho=2.0CV=0.01features_desc_0th.data"), " ",
				Integer.class);
		
		long[] selectedFeatures = Arrays.copyOf(features.transpose()
				.toLongArray()[0], 20);
		
		Matrix fsX = X.selectColumns(Ret.NEW, selectedFeatures);
		File f = new File("src/data/mat/npfs/fsmat/" + dataset.substring(0, dataset.length() - 4) + "_NumFeatures20_NPFS_X.mat");
		f.createNewFile();
		Utils.writeMatrix2DToMat(f, fsX, "X");
		File yf = new File("src/data/mat/npfs/fsmat/" + dataset.substring(0, dataset.length() - 4) + "_NumFeatures20_NPFS_Y.mat");
		yf.createNewFile();
		Utils.writeMatrix2DToMat(yf, Y, "Y");
		
	}
}
