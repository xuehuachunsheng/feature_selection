package ufs.general.cap.test;

import java.io.File;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeans;
import ufs.cluster.evaluate.EvaluationIndexType;

import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestACCWRTNFANDNN {
	public static void main(String[] args) throws Exception {
		int[] numClusters = { 10, 4, 40, 10, 15, 20, 26, 2 };

		for (int i = 2; i < 8; i++) {

			String dataset = ConstValues.DATA_MATRIX[i];
			System.out.println("-----------------------" + dataset
					+ "---------------");
			Matrix X = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File(
					ConstValues.DATA_MATRIX_PATH + dataset), "Y");
			// Matrix capNeighbors = Utils.load2DMatrixFromTxt(
			// new File("src/data/mat/cap/centromapping/neighbors/"
			// + dataset.substring(0, dataset.length() - 4)
			// + "_Neighbors" + X.getRowCount() + ".data"), " ",
			// Integer.class);
			Matrix acc = Matrix.Factory.zeros(9, 18);
			Matrix nmi = Matrix.Factory.zeros(9, 18);
			for (int numFeatures = 20; numFeatures <= 100; numFeatures += 10) {
				for (int numNeighbors = 3; numNeighbors <= 20; numNeighbors++) {
					long[] selectedColumns = new long[numNeighbors];
					for (int j = 0; j < numNeighbors; j++) {
						selectedColumns[j] = j;
					}
					Matrix features = Utils
							.load2DMatrixFromTxt(
									new File(
											"src/data/mat/cap/featureranking/"
													+ dataset
															.substring(
																	0,
																	dataset.length() - 4)
													+ "_Neighbors"
													+ numNeighbors
													+ "rho=2.0CV=0.01features_desc_0th.data"),
									" ", Integer.class);
					long[] selectedFeatures = Arrays.copyOf(features
							.transpose().toLongArray()[0], numFeatures);
					// Cap ufs = new Cap(X, capNeighbors.selectColumns(Ret.NEW,
					// selectedColumns), numFeatures);
					// ufs.centralize();
					// ufs.centrosymmetricMapping();
					// ufs.optimalW();
					// ufs.computeFeatureRanking();
					double tACCSum = 0;
					double tNMISum = 0;
					for (int j = 0; j < 20; j++) {

						Cluster cluster = new KMeans(X.selectColumns(Ret.NEW,
								selectedFeatures), numClusters[i]);
						// Cluster cluster = new KMeans(
						// ufs.getDataAfterFeaturesSelected(), numClusters);

						cluster.setRealLabels(Y.transpose().toIntArray()[0]);
						cluster.cluster();
						tACCSum += cluster
								.getEvaluationResult(EvaluationIndexType.PURITY);
						tNMISum += cluster
								.getEvaluationResult(EvaluationIndexType.NMI);

					}
					acc.setAsDouble(tACCSum / 20, (numFeatures - 20) / 10,
							numNeighbors - 3);
					nmi.setAsDouble(tNMISum / 20, (numFeatures - 20) / 10,
							numNeighbors - 3);
				}
				
			}
			
			System.out.println("ACC: ");
			System.out.println(acc);
			System.out.println("\r\nNMI: ");
			System.out.println(nmi);
			
			Utils.writeMatrixToTxt(acc, "src/data/_20170203results/" + dataset
					.substring(
							0,
							dataset.length() - 4)+"_ACC.data", "\t", Double.class);
			Utils.writeMatrixToTxt(nmi, "src/data/_20170203results/" + dataset
					.substring(
							0,
							dataset.length() - 4)+"_NMI.data", "\t", Double.class);
			
		}
	}
}
