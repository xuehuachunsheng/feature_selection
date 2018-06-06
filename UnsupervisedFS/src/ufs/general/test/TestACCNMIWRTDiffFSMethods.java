package ufs.general.test;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.algorithm.impl.KMeans;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.featureselection.impl.Cap;
import ufs.utils.ConstValues;
import ufs.utils.Utils;

public class TestACCNMIWRTDiffFSMethods {
	public static void main(String[] args) throws Exception {
//		System.out.println("9 Neighbors Cap");
//		// Feature Selection
//		int[] numClusters = { 10, 4, 40, 10, 15, 20, 26, 2 };
//		long[] numNeighbors = new long[9];
//		for (int i = 0; i < numNeighbors.length; i++) {
//			numNeighbors[i] = i;
//		}
//		for (int i = 0; i < 8; i++) {
//
//			String dataset = ConstValues.CARCINOM_MATRIX_174$9182$11;
//
//			Matrix X = Utils.loadMatrix2DFromMat(new File(
//					ConstValues.DATA_MATRIX_PATH + dataset), "X");
//			Matrix Y = Utils.loadMatrix2DFromMat(new File(
//					ConstValues.DATA_MATRIX_PATH + dataset), "Y");
//			
//			Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/"+dataset.substring(0, dataset.length() - 4)+"_Neighbors173.data"), " ", Integer.class);
////			Cap cap = new Cap(X, neighbors.selectColumns(Ret.NEW, numNeighbors), 0);
////			System.out.println("Cap constructed");
////			cap.centralize();
////			cap.centrosymmetricMapping();
////			cap.optimalW();
////			System.out.println("W constructed");
////			cap.computeFeatureRanking();
////			System.out.println("ranking computed");
//			
//			// NPFS
//			for (int numFeatures = 20; numFeatures <= 100; numFeatures += 10) {
////				long[] selectedFeatures = Arrays.copyOf(features.transpose()
////						.toLongArray()[0], numFeatures);
////				cap.setNumFeatures(numFeatures);
//				double tSum = 0;
//				for (int j = 0; j < 20; j++) {
//
////					Cluster cluster = new KMeans(cap.getDataAfterFeaturesSelected(), 11);
////					weka.core.Instances instances = Utils.matrixWithoutLabelToInstances(ls.getDataAfterFeaturesSelected(), "Basehock");
////					weka.clusterers.SimpleKMeans cl = new weka.clusterers.SimpleKMeans();
////					cl.setNumClusters(numClusters[i]);
////					cl.buildClusterer(instances);
////					int[] preditLabels = new int[instances.numInstances()];
////					for (int k = 0; k < instances.numInstances(); k++) {
////						preditLabels[k] = cl.clusterInstance(instances.instance(k));
////					}
//					cluster.setRealLabels(Y.transpose().toIntArray()[0]);
////					cluster.setPredictLabels(preditLabels);
//					cluster.cluster();
//					
//					tSum += cluster
//							.getEvaluationResult(EvaluationIndexType.PURITY);
//
//				}
//				System.out.print(tSum / 20 + "\t");
//
//			}
//			System.out.println();
//		}
	}

}
