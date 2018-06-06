package ufs.general.ls.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.LaplacianScoreForUnsupervisedFS;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestLs {
	public static void main(String[] args) throws Exception {
		int dsID = 4;

		String dataset = ConstValues.DATA_MATRIX[dsID];
		int numClusters = ConstValues.NUM_CLUSTERS[dsID];
		System.out.println(dataset);

		// Feature Selection
		for (int numNeis = 3; numNeis <= 15; numNeis++) {
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

			for (double t = 1e-4; t <= 1e4 + 1; t *= 10) {

				LaplacianScoreForUnsupervisedFS cap = new LaplacianScoreForUnsupervisedFS(X,
						neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
				cap.setT(t);
				cap.middleProcess();
				double tACCSum = 0;

				for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
					cap.setNumFeatures(numFeatures);

//					for (int j = 0; j < numRepeat; j++) {
						Cluster cluster = new KMeansInWeka(cap.getDataAfterFeaturesSelected(), numClusters);

						cluster.setRealLabels(Y.transpose().toIntArray()[0]);
						cluster.cluster();

						tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
//					}

				}
				System.out.print(tACCSum / 10 + "\t");
			}
			System.out.println();
		}
	}
}
