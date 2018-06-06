package ufs.cluster.evaluate.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.evaluate.InnerIndex;

/**
 * The basic implementation of Dunn Index.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class DunnIndex extends InnerIndex {

	/**
	 * The diameter of each cluster. diamC[i] stores the diameter (Farthest
	 * distance of two samples) of the cluster corresponding to those whom
	 * labels are equals to sample located at centerIndices[i]. This variable is
	 * computed by given X, predictLabels and centerIndices. It is used in DI.
	 * 
	 * We assign
	 * 
	 * <pre>
	 * 	diamC[i] = max_{1<=j<k<=|C_i|} dist(x_j, x_k)
	 * </pre>
	 * 
	 * It is a row vector.
	 */
	Matrix diamC;

	/**
	 * The minimum distance of two samples in two different clusters. Likewise,
	 * dMin[i][j] stores the minimum distance between two samples located in C_i
	 * and C_j, respectively.
	 * 
	 * We assign
	 * 
	 * <pre>
	 * 	dMin[i][j] = min_{x_i ∈ C_i, x_j ∈ C_j} dist(x_i, x_j)
	 * </pre>
	 * 
	 * It is also a symmetric matrix.
	 */
	Matrix dMin;

	public DunnIndex(Matrix pData, Matrix pDistData, int[] pPredictLabels,
			Matrix centers) {
		super(pData, pDistData, pPredictLabels, centers);
		paramConstGenerated();
	}

	public void paramConstGenerated() {
		// X = pX;
		int numCenters = (int) centers.getRowCount();
		// <ClusterIndex, List<Integer>>
		Map<Integer, List<Integer>> map = new HashMap<>();

		for (int i = 0; i < numCenters; i++) {
			List<Integer> list = new ArrayList<>();
			map.put(i, list);
		}

		for (int i = 0; i < predictLabels.length; i++) {
			map.get(predictLabels[i]).add(i);
		}

		// Compute dimC and dMin

		Matrix tDiamC = Matrix.Factory.zeros(1, numCenters);
		Matrix tDMin = Matrix.Factory.zeros(numCenters, numCenters);

		for (int i = 0; i < numCenters; i++) {
			Integer[] indices = map.get(i).toArray(new Integer[0]);
			double tDiamCi = Double.MIN_VALUE;
			for (int j = 0; j < indices.length; j++) {
				for (int k = j + 1; k < indices.length; k++) {
					double tDist = distData.getAsDouble(indices[j], indices[k]);
					if (tDiamCi < tDist) {
						tDiamCi = tDist;
					}
				}
			}
			tDiamC.setAsDouble(tDiamCi, 0, i);
			for (int j = i + 1; j < numCenters; j++) {
				Integer[] _indices = map.get(j).toArray(new Integer[0]);
				double tDMinDist = Double.MAX_VALUE;
				for (int k = 0; k < indices.length; k++) {
					for (int m = 0; m < _indices.length; m++) {
						if (tDMinDist > distData.getAsDouble(indices[k],
								_indices[m])) {
							tDMinDist = distData.getAsDouble(indices[k],
									_indices[m]);
						}
					}
				}
				tDMin.setAsDouble(tDMinDist, i, j);
				tDMin.setAsDouble(tDMinDist, j, i);
			}
		}

		dMin = tDMin;
		diamC = tDiamC;

	}

	@Override
	public double precision() {
		double tDI = Double.MIN_VALUE;
		double tMaxDiam = diamC.max(Ret.NEW, Matrix.ALL).getAsDouble(0, 0);
		for (int i = 0; i < centers.getRowCount(); i++) {
			for (int j = i + 1; j < centers.getRowCount(); j++) {
				if (tDI < dMin.getAsDouble(i, j)) {
					tDI = dMin.getAsDouble(i, j);
				}
			}
		}
		return tDI / tMaxDiam;
	}

}
