package ufs.cluster.evaluate;

import org.ujmp.core.Matrix;

/**
 * It is assumed that the class labels is unknown. The outer indices computed
 * does not use the class labels.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public abstract class InnerIndex implements ClusterEvaluation {

	/**
	 * The data matrix. Each row is a sample.
	 */
	protected Matrix data;

	/**
	 * The distance matrix of the data. It is a symmetric squre matrix that
	 * distData[i][j] stores the distance of the i-th sample and the j-th
	 * sample.
	 */
	protected Matrix distData;

	/**
	 * The center samples. Each row is a sample.
	 */
	protected Matrix centers;

	/**
	 * Predict labels.
	 */
	protected int[] predictLabels;

	public InnerIndex(Matrix pData, Matrix pDistData, int[] pPredictLabels,
			Matrix pCenters) {
		data = pData;
		distData = pDistData;
		predictLabels = pPredictLabels;
		centers = pCenters;
	}

	public Matrix getData() {
		return data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}

	public Matrix getDistData() {
		return distData;
	}

	public void setDistData(Matrix distData) {
		this.distData = distData;
	}

	public Matrix getCenters() {
		return centers;
	}

	public int[] getPredictLabels() {
		return predictLabels;
	}

	public void setPredictLabels(int[] predictLabels) {
		this.predictLabels = predictLabels;
	}

}
