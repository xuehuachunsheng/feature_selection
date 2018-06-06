package ufs.cluster.evaluate.impl;

import org.ujmp.core.Matrix;

import ufs.cluster.evaluate.PairwiseOuterIndex;

/**
 * The basic implementation of Jaccard Coefficient.<br>
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 */
public class JaccardCoefficient extends PairwiseOuterIndex {

	public JaccardCoefficient(Matrix pData, int[] pPredictLabels,
			int[] pRealLabels) {
		super(pData, pPredictLabels, pRealLabels);
	}

	@Override
	public double precision() {
		return (ss.size() + 0.0) / (ss.size() + sd.size() + ds.size());
	}
}
