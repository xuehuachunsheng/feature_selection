package ufs.cluster.evaluate;

import org.ujmp.core.Matrix;

/**
 * It is assumed that the class labels is known.
 * The outer indices computed uses the class labels.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public abstract class OuterIndex implements ClusterEvaluation {

	/**
	 * The data matrix.
	 * Each row is a sample.
	 */
	protected Matrix data;
	
	/**
	 * Predict and real labels. The length of this variable must be equal with
	 * each other.
	 */
	protected int[] predictLabels, realLabels;


	public OuterIndex(Matrix pData, int[] pPredictLabels, int[] pRealLabels) {
		data = pData;
		predictLabels = pPredictLabels;
		realLabels = pRealLabels;
	}


	public Matrix getData() {
		return data;
	}


	public void setData(Matrix data) {
		this.data = data;
	}


	public int[] getPredictLabels() {
		return predictLabels;
	}


	public void setPredictLabels(int[] predictLabels) {
		this.predictLabels = predictLabels;
	}


	public int[] getRealLabels() {
		return realLabels;
	}


	public void setRealLabels(int[] realLabels) {
		this.realLabels = realLabels;
	}
	
}
