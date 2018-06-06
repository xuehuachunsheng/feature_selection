package ufs.general.mcfs.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Mcfs;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestMcfs {
	public static void main(String[] args) throws Exception {
		System.out.println("7 Neighbors MCFS");
		System.out.println("lambda: " + ConstValues.LS_CONSTANT);
		for (int i = 7; i < 8; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 20;
			
			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/"
					+ dataset.substring(0, dataset.length() - 4) + "_Neighbors" + (X.getRowCount() - 1) + ".data"), " ",
					Integer.class);
			long[] numNeighbors = new long[7];
			for (int j = 0; j < numNeighbors.length; j++) {
				numNeighbors[j] = j;
			}
			Mcfs cap = new Mcfs(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
			cap.middleProcess();
			// CAP
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				cap.setNumFeatures(numFeatures);
				double tACCSum = 0;
				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(cap.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();

					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
				}
				System.out.println(tACCSum / numRepeat);
			}

		}
		
	}

	// System.out.println("Matrix: \r\n" + m + "\r\n");
}
