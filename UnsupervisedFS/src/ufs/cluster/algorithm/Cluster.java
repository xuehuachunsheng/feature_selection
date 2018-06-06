package ufs.cluster.algorithm;

import org.ujmp.core.Matrix;

import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.cluster.evaluate.impl.ClusterEvaluationDelegate;

/**
 * The basic implementation of a general cluster.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public abstract class Cluster {

	/**
	 * The data matrix, each row of which is a sample.
	 */
	protected Matrix data;

	/**
	 * The number of clusters.
	 */
	protected int k;

	/**
	 * The predict labels.
	 */
	protected int[] predictLabels;

	/**
	 * The real labels.
	 */
	protected int[] realLabels;

	/**
	 * The centers, each row of which is a center (virtual) sample. 
	 */
	protected Matrix centers;
	
	public Cluster(Matrix pData) {
		data = pData;
	}

	public Cluster(Matrix data, int k) {
		super();
		this.data = data;
		this.k = k;
	}

	public abstract void cluster();

	/**
	 * Before using this method, you should preserve the completeness of this cluster.
	 * E.g. If the et == JC, this cluster should contains real labels.
	 * @param et
	 * @return
	 */
	public double getEvaluationResult(EvaluationIndexType et) {
		return new ClusterEvaluationDelegate(this, et).precision(); 
	}
	
	public Matrix getData() {
		return data;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public int[] getPredictLabels() {
		return predictLabels;
	}

	public int[] getRealLabels() {
		return realLabels;
	}

	public void setRealLabels(int[] realLabels) {
		this.realLabels = realLabels;
	}

	public Matrix getCenters() {
		return centers;
	}

	public void setPredictLabels(int[] preditLabels) {
		predictLabels = preditLabels;
	}

}
