package ufs.general.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestBaseLine {
	public static void main(String[] args) throws Exception {
		for (int i = 0; i < ConstValues.STD_DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 10;

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			// CAP
			double tACCSum = 0;
			for (int j = 0; j < numRepeat; j++) {
				Cluster cluster = new KMeansInWeka(X, numClusters);
				cluster.setRealLabels(Y.transpose().toIntArray()[0]);
				cluster.cluster();
				tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
			}
			System.out.println(tACCSum / numRepeat);

		}
	}
}
