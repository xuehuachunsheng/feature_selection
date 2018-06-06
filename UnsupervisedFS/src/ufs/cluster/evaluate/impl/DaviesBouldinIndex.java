package ufs.cluster.evaluate.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.evaluate.InnerIndex;

/**
 * The basic implementation of Davies-Bouldin Index.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class DaviesBouldinIndex extends InnerIndex {

	/**
	 * The average distance in each cluster. avgC[i] stores the average distance
	 * of the cluster corresponding to those whom labels are equals to sample
	 * located at centerIndices[i]. This variable is computed by given X,
	 * predictLabels and centerIndices. It is used in DBI.
	 * 
	 * We assign
	 * 
	 * <pre>
	 * 	avgC[i] = 2*(Sum_{1<=j<k<=C_i} dist(x_j, x_k)) / (|C_i|*(|C_i| - 1))
	 * </pre>
	 * 
	 * It is a row vector.
	 */
	Matrix avgC;

	/**
	 * The distance among those centerIndices. dCen[i][j] stores the distance
	 * between the samples located in centerIndices[i] and center[j]. Obviously,
	 * it is a symmetric matrix. It is used in DBI.
	 * 
	 * We assign
	 * 
	 * <pre>
	 * dCen[i][j] = dist(x_centerIndices[i], x_centerIndices[j])
	 * </pre>
	 */
	Matrix dCen;

	public DaviesBouldinIndex(Matrix pData, Matrix pDistData,
			int[] predictLabels, Matrix centers) {
		super(pData, pDistData, predictLabels, centers);
		paramConstGenerated();
	}

	public void paramConstGenerated() {
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

		// Compute avgC and dCen
		Matrix tAvgC = Matrix.Factory.zeros(1, numCenters);
		Matrix tDCen = Matrix.Factory.zeros(numCenters, numCenters);
		for (int i = 0; i < numCenters; i++) {
			Integer[] indices = map.get(i).toArray(new Integer[0]);
			double tSumDist = 0;
			for (int j = 0; j < indices.length; j++) {
				for (int k = j + 1; k < indices.length; k++) {
					double tDist = distData.getAsDouble(indices[j], indices[k]);
					tSumDist += tDist;
				}
			}
			tAvgC.setAsDouble(indices.length == 1 ? 0 : tSumDist * 2
					/ (indices.length * (indices.length - 1)), 0, i);

			for (int j = i + 1; j < numCenters; j++) {
				// We use the norm2 for computing the distance of two vectors
				double tDistance = centers.selectRows(Ret.NEW, i)
						.minus(centers.selectRows(Ret.NEW, j)).normF();
				tDCen.setAsDouble(tDistance, i, j);
				tDCen.setAsDouble(tDistance, j, i);
			}
		}
		avgC = tAvgC;
		dCen = tDCen;
	}

	@Override
	public double precision() {
		double tDBI = 0;
		for (int i = 0; i < centers.getRowCount(); i++) {
			double tMax = Double.MIN_VALUE;
			for (int j = 0; j < centers.getRowCount(); j++) {
				if (j != i) {
					double tValue = (avgC.getAsDouble(0, i) + avgC.getAsDouble(
							0, j)) / dCen.getAsDouble(i, j);
					if (tMax < tValue) {
						tMax = tValue;
					}
				}
			}
			tDBI += tMax;
		}
		return tDBI / centers.getRowCount();
	}

	public Matrix getAvgC() {
		return avgC;
	}

	public Matrix getdCen() {
		return dCen;
	}

}
