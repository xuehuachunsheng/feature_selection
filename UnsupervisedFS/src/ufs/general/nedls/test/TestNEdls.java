package ufs.general.nedls.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeans;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Edls;
import ufs.featureselection.impl.LaplacianScoreForUnsupervisedFS;
import ufs.featureselection.impl.NEdls;
import ufs.featureselection.impl.Npfs;
import ufs.general.test.TestBaseLine;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestNEdls {
	public static void main(String[] args) throws Exception {
		System.out.println("NEDLS");

		// Feature Selection
		for (int i = 0; i < ConstValues.STD_DATA_MATRIX.length; i++) {

			String dataset = ConstValues.STD_DATA_MATRIX[i];
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
			NEdls edls = new NEdls(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
			edls.middleProcess();

			// CAP
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				edls.setNumFeatures(numFeatures);
				double tACCSum = 0;

				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(edls.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();

					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
				}
				System.out.println(tACCSum / numRepeat);
			}
		}
	}
}