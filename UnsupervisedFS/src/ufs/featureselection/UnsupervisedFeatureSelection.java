package ufs.featureselection;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * The general framework of unsupervised feature selection.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 13, 2017 <br>
 * Last Modified Time: Jan. 13, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public abstract class UnsupervisedFeatureSelection implements FeatureSelection {

	/**
	 * The original data matrix. Each row is a sample.
	 */
	protected Matrix data;

	/**
	 * The number of selected features
	 */
	protected int numFeatures;

	/**
	 * The feature subset. Normally, it is unsorted. In its implementation, you
	 * can sort it by each features importance.
	 */
	protected int[] featureSubset;

	public UnsupervisedFeatureSelection(Matrix pData, int pNumFeatures) {
		super();
		this.data = pData;
		numFeatures = pNumFeatures;
	}

	public int getNumFeatures() {
		return numFeatures;
	}

	public void setNumFeatures(int numFeatures) {
		this.numFeatures = numFeatures;
	}

	public Matrix getData() {
		return data;
	}

	@Override
	public int[] getFeatureSubset() {
		return featureSubset;
	}

	@Override
	public String toString() {
		return "UnsupervisedFeatureSelection [\r\ndata=" + data
				+ ", \r\nnumFeatures=" + numFeatures + ", \r\nfeatureSubset="
				+ Arrays.toString(featureSubset) + "\r\n]";
	}

}
