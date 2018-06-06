package ufs.cluster.algorithm.impl;

import org.ujmp.core.Matrix;

import ufs.cluster.algorithm.Cluster;
import ufs.utils.Utils;

public class KMeansInWeka extends Cluster {
	weka.clusterers.SimpleKMeans cluster = new weka.clusterers.SimpleKMeans();

	public KMeansInWeka(Matrix pData, int k) {
		super(pData, k);
		try {
			cluster.setNumClusters(k);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void cluster() {
		weka.core.Instances instances = Utils.matrixWithoutLabelToInstances(
				data, "Relation");
		
		try {
			cluster.buildClusterer(instances);
			int[] tPredictLabels = new int[instances.numInstances()];
			for (int k = 0; k < instances.numInstances(); k++) {
				tPredictLabels[k] = cluster.clusterInstance(instances
						.instance(k));
			}
			predictLabels = tPredictLabels;
			centers = Utils.instancesToMatrixWithoutLabel(cluster.getClusterCentroids());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
