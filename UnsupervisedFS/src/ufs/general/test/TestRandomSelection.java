package ufs.general.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Cap;
import ufs.featureselection.impl.RandomSelection;
import ufs.general.test.TestBaseLine;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestRandomSelection {
	public static void main(String[] args) throws Exception {
		System.out.println("Randomly Selection");
		for (int i = 0; i < ConstValues.STD_DATA_MATRIX.length; i++) {

			String dataset = ConstValues.STD_DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 20;

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			// CAP
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				RandomSelection cap = new RandomSelection(X, numFeatures);
				cap.middleProcess();
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

}
