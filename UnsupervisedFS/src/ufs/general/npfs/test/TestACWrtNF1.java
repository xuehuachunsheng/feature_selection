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

public class TestACWrtNF1 {
	public static void main(String[] args) throws Exception {
		System.out.println("9 Neighbors NPFS");

		// Feature Selection
		int numRepeat = 20;
		Matrix resultsACC = Matrix.Factory.zeros(9, ConstValues.DATA_MATRIX.length);
		Matrix resultsNMI = Matrix.Factory.zeros(9, ConstValues.DATA_MATRIX.length);
		// Feature Selection
		for (int i = 0; i < ConstValues.DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.print(dataset + "\t");
		}
		System.out.println();
		for (int i = 0; i < ConstValues.DATA_MATRIX.length; i++) {
			String dataset = ConstValues.DATA_MATRIX[i];
			int numClusters = 10;

			Matrix X = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "Y");

			Matrix neighbors = Utils.load2DMatrixFromTxt(
					new File("src/data/mat/npfs/neighbors/"
							+ dataset.substring(0, dataset.length() - 4)
							+ "_Neighbors" + (X.getRowCount()-1) + ".data"), " ",
					Integer.class);
			long[] numNeighbors = new long[9];
			for (int j = 0; j < numNeighbors.length; j++) {
				numNeighbors[j] = j;
			}
			X = Utils.centralize(X, 0);
			Npfs cap = new Npfs(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
			cap.middleProcess();

			// CAP
			for (int numFeatures = 20; numFeatures <= 100; numFeatures += 10) {
				cap.setNumFeatures(numFeatures);
				double tACCSum = 0;
				double tNMISum = 0;
				for (int j = 0; j < numRepeat; j++) {
					Cluster cluster = new KMeansInWeka(
							cap.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();
					
					tACCSum += cluster.getEvaluationResult(EvaluationIndexType.ACC);
					tNMISum += cluster.getEvaluationResult(EvaluationIndexType.NMI);
				}
				resultsACC.setAsDouble(tACCSum / numRepeat, (numFeatures-20) / 10, i);
				resultsNMI.setAsDouble(tNMISum / numRepeat, (numFeatures-20) / 10, i);
			}
		}
		
		System.out.println(resultsACC);
		System.out.println(resultsNMI);

	}

}
