package ufs.general.npfs.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeansInWeka;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Cap;
import ufs.featureselection.impl.Npfs;
import ufs.general.test.TestBaseLine;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestNpfs {
	public static void main(String[] args) throws Exception {
		//System.out.println("7 Neighbors NPFS");

		for (int numNei = 3; numNei <= 15; numNei++) {
			// Feature Selection
			for (int i = 1; i < ConstValues.STD_DATA_MATRIX.length; i++) {
				String dataset = ConstValues.DATA_MATRIX[i];
				
				int numClusters = ConstValues.NUM_CLUSTERS[i];
				int numRepeat = 20;
				Matrix X = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "X");
				Matrix Y = Utils.loadMatrix2DFromMat(new File(ConstValues.DATA_MATRIX_PATH + dataset), "Y");

				Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/"
						+ dataset.substring(0, dataset.length() - 4) + "_Neighbors" + (X.getRowCount() - 1) + ".data"),
						" ", Integer.class);
				long[] numNeighbors = new long[numNei];
				for (int j = 0; j < numNeighbors.length; j++) {
					numNeighbors[j] = j;
				}
				Npfs cap = new Npfs(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
				cap.middleProcess();
				double tACCSum = 0;
				// Npfs
				for (int numFeatures = 20; numFeatures <= 200; numFeatures += 20) {
					cap.setNumFeatures(numFeatures);
					
					for (int j = 0; j < numRepeat; j++) {
						Cluster cluster = new KMeansInWeka(cap.getDataAfterFeaturesSelected(), numClusters);

						cluster.setRealLabels(Y.transpose().toIntArray()[0]);
						cluster.cluster();

						tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
					}
					

				}
				System.out.print(tACCSum / 200 + "\t");
			}
			System.out.println();
		}
	}

}
