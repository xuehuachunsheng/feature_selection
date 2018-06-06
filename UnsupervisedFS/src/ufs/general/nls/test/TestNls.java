package ufs.general.nls.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Nls;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestNls {
	public static void main(String[] args) throws Exception {

		int numNeis = 9;

		// Feature Selection
		for (int i = 1; i < ConstValues.STD_DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 20;

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/"
					+ dataset.substring(0, dataset.length() - 4) + "_Neighbors" + (X.getRowCount() - 1) + ".data"), " ",
					Integer.class);
			long[] numNeighbors = new long[numNeis];
			for (int j = 0; j < numNeighbors.length; j++) {
				numNeighbors[j] = j;
			}
			Nls edls = new Nls(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
			edls.middleProcess();
			double tACCSum = 0;

			// CAP
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				edls.setNumFeatures(numFeatures);

				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(edls.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();

					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
				}
			}
			System.out.println(tACCSum / numRepeat);

		}

	}
}
