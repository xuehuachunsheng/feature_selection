package ufs.general.cap.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Cap;
import ufs.general.test.TestBaseLine;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestACWrtNF {
	public static void main(String[] args) throws Exception {
		System.out.println("9 Neighbors CAP");

		int i = 5;
		String dataset = ConstValues.DATA_MATRIX[i];
		System.out.println(dataset);
		int numClusters = ConstValues.NUM_CLUSTERS[i];
		int numRepeat = 20;

		Matrix X = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "X");
		Matrix Y = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH + dataset), "Y");
		
		Matrix neighbors = Utils.load2DMatrixFromTxt(
				new File("src/data/mat/cap/centromapping/neighbors/"
						+ dataset.substring(0, dataset.length() - 4)
						+ "_Neighbors" + X.getRowCount() + ".data"), " ",
				Integer.class);
		long[] numNeighbors = new long[9];
		for (int j = 0; j < numNeighbors.length; j++) {
			numNeighbors[j] = j;
		}
		Cap cap = new Cap(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
		cap.middleProcess();

		// CAP
		for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
			cap.setNumFeatures(numFeatures);
			double tACCSum = 0;
			for (int j = 0; j < numRepeat; j++) {
				Cluster cluster = new KMeansInWeka(
						cap.getDataAfterFeaturesSelected(), numClusters);

				cluster.setRealLabels(Y.transpose().toIntArray()[0]);
				cluster.cluster();

				tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
			}
			System.out.println(tACCSum / numRepeat);
		}
	}

}
