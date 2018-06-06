package ufs.general.capnpfs.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeans;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.ConcurrentCapNpfs;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestKMeans {
	public static void main(String[] args) throws Exception {
		int[] numClusters = { 10, 4, 40, 10, 15, 20, 26, 2 };

		for (int i = 1; i < 8; i++) {

			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println("-----------------------" + dataset
					+ "---------------");
			Matrix X = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "Y");
			Matrix capNeighbors = Utils.load2DMatrixFromTxt(
					new File("src/data/mat/cap/centromapping/neighbors/"
							+ dataset.substring(0, dataset.length() - 4)
							+ "_Neighbors" + X.getRowCount() + ".data"), " ",
					Integer.class);
			Matrix neighbors = Utils.load2DMatrixFromTxt(
					new File("src/data/mat/npfs/neighbors/"
							+ dataset.substring(0, dataset.length() - 4)
							+ "_Neighbors" + (X.getRowCount() - 1) + ".data"),
					" ", Integer.class);

			Matrix acc = Matrix.Factory.zeros(9, 18);
			Matrix nmi = Matrix.Factory.zeros(9, 18);
			for (int numFeatures = 20; numFeatures <= 100; numFeatures += 10) {
				long[] selectedColumns = new long[9];
				for (int j = 0; j < 9; j++) {
					selectedColumns[j] = j;
				}
				ConcurrentCapNpfs ccn = new ConcurrentCapNpfs(X,
						neighbors.selectColumns(Ret.NEW, selectedColumns),
						capNeighbors.selectColumns(Ret.NEW, selectedColumns),
						numFeatures);

				ccn.centralize();
				ccn.centrosymmetricMapping();
				ccn.optimalW();
				ccn.optimalW_();
				for (double alpha = 0; alpha <= 2.01; alpha += 0.1) {
					ccn.setAlpha(alpha);
					ccn.computeFeatureRanking();
					long[] selectedFeatures = Utils.intArrayToLongType(ccn
							.getFeatureSubset());
					double tACCSum = 0;
					double tNMISum = 0;
					// for (int j = 0; j < 10; j++) {

					Cluster cluster = new KMeans(X.selectColumns(Ret.NEW,
							selectedFeatures), numClusters[i]);
					// Cluster cluster = new KMeans(
					// ufs.getDataAfterFeaturesSelected(), numClusters);

					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
					cluster.cluster();
					tACCSum += cluster
							.getEvaluationResult(EvaluationIndexType.PURITY);
					// tNMISum += cluster
					// .getEvaluationResult(EvaluationIndexType.NMI);

					// }
					System.out.print(tACCSum + "\t");
					// acc.setAsDouble(tACCSum / 20, (numFeatures - 20) / 10,
					// numNeighbors - 3);
					// nmi.setAsDouble(tNMISum / 20, (numFeatures - 20) / 10,
					// numNeighbors - 3);
				}
				System.out.println();
			}

			System.out.println("ACC: ");
			System.out.println(acc);
			System.out.println("\r\nNMI: ");
			System.out.println(nmi);

			Utils.writeMatrixToTxt(
					acc,
					"src/data/_20170203results/"
							+ dataset.substring(0, dataset.length() - 4)
							+ "_ACC.data", "\t", Double.class);
			Utils.writeMatrixToTxt(
					nmi,
					"src/data/_20170203results/"
							+ dataset.substring(0, dataset.length() - 4)
							+ "_NMI.data", "\t", Double.class);

		}

	}
}
