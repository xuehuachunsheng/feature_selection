package ufs.featureselection.impl;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.utils.Utils;

public class RandomSelection extends UnsupervisedFeatureSelection {

	public RandomSelection(Matrix pData, int pNumFeatures) {
		super(pData, pNumFeatures);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays
				.copyOf(featureSubset, numFeatures)));
	}

	@Override
	public void middleProcess() {
		int[] features = Utils.randomPermutationArray(0,
				(int) data.getColumnCount());
		featureSubset = Arrays.copyOf(features, numFeatures);
	}
}
