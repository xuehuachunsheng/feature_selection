package ufs.general.mv.test;

import java.io.File;

import org.ujmp.core.Matrix;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.MaxVariance;
import ufs.general.test.TestBaseLine;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestMv {
	public static void main(String[] args) throws Exception {
		
		System.out.println("Max Variance");
		
		// Feature Selection
		int numRepeat = 20;
		for (int i = 0; i < ConstValues.DATA_MATRIX.length; i++) {

			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println(dataset);
			int numClusters = ConstValues.NUM_CLUSTERS[i];

			Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");
			
			MaxVariance mv = new MaxVariance(X, 0);
			mv.middleProcess();
			
			// CAP
			for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
				mv.setNumFeatures(numFeatures);
				double tACCSum = 0;
				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(mv.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();

					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
				}
				System.out.println(tACCSum / numRepeat);
			}
		}
		
	}

}
