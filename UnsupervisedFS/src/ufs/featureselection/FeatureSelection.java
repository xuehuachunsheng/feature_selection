package ufs.featureselection;

import org.ujmp.core.Matrix;

/**
 * The general framework of feature selection.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 13, 2017 <br>
 * Last Modified Time: Jan. 13, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public interface FeatureSelection {
	/**
	 * Get the feature subset.
	 */
	int[] getFeatureSubset();
	
	/**
	 * Get the data after features selected X_I, where I is the feature subset.
	 */
	Matrix getDataAfterFeaturesSelected();
	
	/**
	 * A method managing all the process of a feature selection method.
	 * It is a delegate method.
	 */
	void middleProcess();
}
