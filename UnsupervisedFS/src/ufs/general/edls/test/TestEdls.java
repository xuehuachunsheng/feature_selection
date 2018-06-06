package ufs.general.edls.test;

import java.io.File;

import org.ujmp.core.Matrix;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Edls;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestEdls {
	public static void main(String[] args) throws Exception {
		System.out.println("EDLS");
		
		// Feature Selection
		for (int i = 1; i < ConstValues.DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
//			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];
			int numRepeat = 20;

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");
			
			Edls edls = new Edls(X, 0);
			edls.middleProcess();
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
