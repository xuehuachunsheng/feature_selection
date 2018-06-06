package ufs.featureselection.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.utils.Utils;
import ufs.utils.Utils.Order;

public class MaxVariance extends UnsupervisedFeatureSelection {

	double[] variances;

	public MaxVariance(Matrix pData, int pNumFeatures) {
		super(pData, pNumFeatures);
		variances = new double[(int) pData.getColumnCount()];
	}

	public double[] computeVariances() {
		Matrix meanValue = data.mean(Ret.NEW, 0, false);
		
		for (int i = 0; i < data.getColumnCount(); i++) {
			double tSum = 0;
			for (int j = 0; j < data.getRowCount(); j++) {
				tSum += (data.getAsDouble(j, i) - meanValue.getAsDouble(0, i))
						* (data.getAsDouble(j, i) - meanValue.getAsDouble(0, i));
			}
			variances[i] = tSum / data.getRowCount();
		}
		return variances;
	}
	
	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays
				.copyOf(featureSubset, numFeatures)));
	}
	
	/**
	 * The features with larger variances are more important.
	 */
	public int[] computeFeatureRanking() {
		featureSubset = Utils.argSort(variances, Order.DESC);
		return featureSubset;
	}
	
	@Override
	public int[] getFeatureSubset() {
		return Arrays.copyOf(featureSubset, numFeatures);
	}
	public static void main(String[] args) {
		Matrix m = Matrix.Factory.rand(3, 2);
		
		System.out.println(m);
		System.out.println(m.mean(Ret.NEW, 0, false));
		MaxVariance mv = new MaxVariance(m, 2);
		mv.computeVariances();
		System.out.println(Arrays.toString(mv.variances));
	}

	@Override
	public void middleProcess() {
		computeVariances();
		computeFeatureRanking();
	}
}
