package ufs.cluster.algorithm.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.utils.Utils;

/**
 * The basic implementation of KMeans.
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class KMeans extends Cluster {

	public KMeans(Matrix data, int k) {
		super(data, k);
	}

	@Override
	public void cluster() {
		int[] tCenterIndices = Arrays
				.copyOfRange(
						Utils.randomPermutationArray(0,
								(int) data.getRowCount()), 0, k);
		Matrix tCenters = data.selectRows(Ret.NEW,
				Utils.intArrayToLongType(tCenterIndices));
		int n = (int) data.getRowCount();

		int[] tPredictLabels = new int[n];
		Matrix _tCenters = Matrix.Factory.rand(tCenters.getRowCount(), tCenters.getColumnCount());
		
		while (tCenters.minus(_tCenters).normF() > 1e-6) {// Convergence condition judgment.
			_tCenters = tCenters;// Reassignment
			int[] tCounters = new int[k];
			for (int i = 0; i < n; i++) {
				double tMinDist = Double.MAX_VALUE;
				int tCluster = 0;
				for (int j = 0; j < k; j++) {
					double tDist = data.selectRows(Ret.NEW, i)
							.minus(tCenters.selectRows(Ret.NEW, j)).normF();
					if (tDist < tMinDist) {
						tMinDist = tDist;
						tCluster = j;
					}
				}
				tPredictLabels[i] = tCluster;
				tCounters[tCluster]++;
			}
			long[][] tPredictIndices = new long[k][];// Reallocate
			for (int i = 0; i < tPredictIndices.length; i++) {
				tPredictIndices[i] = new long[tCounters[i]];
			}
			tCounters = new int[k];
			for (int i = 0; i < tPredictLabels.length; i++) {
				tPredictIndices[tPredictLabels[i]][tCounters[tPredictLabels[i]]++] = i;
			}
			tCenters = Matrix.Factory.emptyMatrix();
			for (int i = 0; i < tPredictIndices.length; i++) {
				tCenters = tCenters.appendVertically(
						Ret.NEW,
						data.selectRows(Ret.LINK, tPredictIndices[i]).mean(
								Ret.NEW, 0, false));
			}
		}
		centers = tCenters;
		predictLabels = tPredictLabels;
	}

}
