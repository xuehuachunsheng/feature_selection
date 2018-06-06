package ufs.general.cluster.test;

import java.io.File;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeans;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.featureselection.impl.LaplacianScoreForUnsupervisedFS;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestKMeans {
	public static void main(String[] args) throws Exception {
		// BaseLine
		int[] numClusters = { 10, 4, 40, 10, 15, 20, 26, 2 };

		for (int i = 0; i < 8; i++) {

			String dataset = ConstValues.DATA_MATRIX[i];

			Matrix X = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "Y");
			// Matrix capNeighbors = Utils.load2DMatrixFromTxt(
			// new File("src/data/mat/cap/centromapping/neighbors/"
			// + dataset.substring(0, dataset.length() - 4)
			// + "_Neighbors" + X.getRowCount() + ".data"), " ",
			// Integer.class);

			double tSum = 0;
			Cluster cluster = new KMeans(X, numClusters[i]);
			weka.core.Instances instances = Utils
					.matrixWithoutLabelToInstances(X, "Basehock");
			weka.clusterers.SimpleKMeans cl = new weka.clusterers.SimpleKMeans();
			cl.setNumClusters(numClusters[i]);
			cl.buildClusterer(instances);
			int[] preditLabels = new int[instances.numInstances()];
			for (int k = 0; k < instances.numInstances(); k++) {
				preditLabels[k] = cl.clusterInstance(instances.instance(k));
			}

			cluster.setRealLabels(Y.transpose().toIntArray()[0]);

			cluster.setPredictLabels(preditLabels);

			tSum = cluster.getEvaluationResult(EvaluationIndexType.NMI);

			System.out.print(tSum + "\t");
		}

	}
}
