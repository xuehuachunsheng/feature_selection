package ufs.general.nnls.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Nls;
import ufs.featureselection.impl.Nnls;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestNNLS {
	public static void main(String[] args) throws Exception {
		System.out.println("9 neighbors NNLS");

		// Feature Selection
		for (int i = 0; i < ConstValues.STD_DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 20;

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/"
					+ dataset.substring(0, dataset.length() - 4) + "_Neighbors" + (X.getRowCount() - 1) + ".data"), " ",
					Integer.class);
			long[] numNeighbors = new long[9];
			for (int j = 0; j < numNeighbors.length; j++) {
				numNeighbors[j] = j;
			}
			Nnls nnls = new Nnls(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
			nnls.middleProcess();
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				nnls.setNumFeatures(numFeatures);
				double tACCSum = 0;

				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(nnls.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();

					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
				}
				System.out.println(tACCSum / numRepeat);
			}
		}

	}
}
