package ufs.cluster.evaluate.impl;

import org.ujmp.core.Matrix;

import ufs.cluster.evaluate.OuterIndex;
import ufs.utils.Utils;

/**
 * The basic implementation of Normalized Mutual Information.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class NormalizedMutualInformation extends OuterIndex {

	/**
	 * The mutual information of the predict clusters and real clusters.
	 */
	double mutualInformation;

	/**
	 * The entropy of predict clusters.
	 */
	double entropyOfPredict;

	/**
	 * The entropy of real clusters.
	 */
	double entropyOfReal;

	Matrix coefficientMatrix;

	public NormalizedMutualInformation(Matrix pData, int[] pPredictLabels,
			int[] pRealLabels) {
		super(pData, pPredictLabels, pRealLabels);
		paramConstGenerated();
	}

	private void paramConstGenerated() {

		int numClusterPred = Utils.maxValueAndIndex(predictLabels)[0] + 1;
		int numClusterReal = Utils.maxValueAndIndex(realLabels)[0] + 1;
		int n = predictLabels.length;
		Matrix tCoefficientMatrix = Matrix.Factory.zeros(numClusterPred,
				numClusterReal);
		int[] predictClusterSize = new int[numClusterPred];

		int[] realClusterSize = new int[numClusterReal];

		for (int i = 0; i < predictLabels.length; i++) {
			int tValue = tCoefficientMatrix.getAsInt(predictLabels[i],
					realLabels[i]);
			tCoefficientMatrix.setAsInt(tValue + 1, predictLabels[i],
					realLabels[i]);
			predictClusterSize[predictLabels[i]]++;
			realClusterSize[realLabels[i]]++;
		}
		coefficientMatrix = tCoefficientMatrix;
		mutualInformation = 0;
		entropyOfPredict = 0;
		entropyOfReal = 0;
		for (int i = 0; i < numClusterPred; i++) {
			for (int j = 0; j < numClusterReal; j++) {
				if(tCoefficientMatrix.getAsDouble(i, j) == 0) {
					continue;
				}
				if(predictClusterSize[i] * realClusterSize[j] == 0) {
					continue;
				}
				mutualInformation += tCoefficientMatrix.getAsDouble(i, j)
						* Math.log(n * tCoefficientMatrix.getAsDouble(i, j)
								/ (predictClusterSize[i] * realClusterSize[j]))
						/ Math.log(2);
			}
		}
		for (int i = 0; i < predictClusterSize.length; i++) {
			if(predictClusterSize[i] == 0) {
				continue;
			}
			entropyOfPredict += predictClusterSize[i]
					* Math.log(predictClusterSize[i] / (n + 0.0)) / Math.log(2);
		}
		for (int i = 0; i < realClusterSize.length; i++) {
			if(realClusterSize[i] == 0) {
				continue;
			}
			entropyOfReal += realClusterSize[i]
					* Math.log(realClusterSize[i] / (n + 0.0)) / Math.log(2);
		}
	}

	@Override
	public double precision() {
		if(mutualInformation == 0 || entropyOfPredict == 0 || entropyOfReal == 0) {
			return 0;
		}
		return mutualInformation / Math.sqrt(entropyOfPredict * entropyOfReal);
	}

	public double getMutualInformation() {
		return mutualInformation;
	}

	public double getEntropyOfPredict() {
		return entropyOfPredict;
	}

	public double getEntropyOfReal() {
		return entropyOfReal;
	}

	public Matrix getCoefficientMatrix() {
		return coefficientMatrix;
	}

}
