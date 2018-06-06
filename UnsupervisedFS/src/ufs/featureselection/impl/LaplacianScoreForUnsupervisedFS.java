package ufs.featureselection.impl;


import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.featureselection.UnsupervisedFeatureSelection;
import ufs.utils.ConstValues;
import ufs.utils.Utils;
import ufs.utils.Utils.Order;

/**
 * The basic implementation of Laplacian score for unsupervised feature
 * selection. <br>
 * See 'Laplacian Score for Feature Selection' for more detail.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 14, 2017 <br>
 * Last Modified Time: Jan. 14, 2017 <br>
 * Progress: Done.<br>
 */
public class LaplacianScoreForUnsupervisedFS extends
		UnsupervisedFeatureSelection {

	/**
	 * The neighbor indices matrix. i-th row of this matrix is the i-th samples'
	 * neighbor indices in the original sample matrix.
	 */
	Matrix neighborIndicesMatrix;

	/**
	 * The weight matrix of the graph models the local structure of the data
	 * space.
	 */
	Matrix weightMatrix;

	/**
	 * The weight matrix constant. weightMatrix[i][j] = exp(-||x_i - x_j||^2 /
	 * t)
	 */
	double t;

	/**
	 * The Laplacian score of each features.
	 */
	double[] laplacianScore;

	public LaplacianScoreForUnsupervisedFS(Matrix pData,
			Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNumFeatures);
		//data = data.standardize(Ret.NEW, 0);  
		neighborIndicesMatrix = pNeighborIndicesMatrix;
		defaultInitialize();
	}

	public void defaultInitialize() {
		t = ConstValues.LS_CONSTANT;
		// Default to set the t as the square of the mean distance of neighbors.
//		int n = (int) data.getRowCount();
//		double sum = 0;
//		for (int i = 0; i < n; i++) {
//			for (int j = 0; j < neighborIndicesMatrix.getColumnCount(); j++) {
//				double tValue = data
//						.selectRows(Ret.LINK, i)
//						.minus(data.selectRows(Ret.LINK,
//								neighborIndicesMatrix.getAsInt(i, j))).normF();
////				tWeightMatrix.setAsDouble(Math.exp(-tValue * tValue / t), i,
////						neighborIndicesMatrix.getAsInt(i, j));
//				sum += tValue;
//			}
////			tWeightMatrix.setAsDouble(1, i, i);
//		}
//		t = sum / (n * neighborIndicesMatrix.getColumnCount());
////		t = t * t;
	}

	public Matrix computeWeightMatrix() {
		int n = (int) data.getRowCount();
		Matrix tWeightMatrix = Matrix.Factory.zeros(n, n);
		
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < neighborIndicesMatrix.getColumnCount(); j++) {
				double tValue = data
						.selectRows(Ret.LINK, i)
						.minus(data.selectRows(Ret.LINK,
								neighborIndicesMatrix.getAsInt(i, j))).normF();
				tWeightMatrix.setAsDouble(Math.exp(-tValue * tValue / t), i,
						neighborIndicesMatrix.getAsInt(i, j));
			}
//			tWeightMatrix.setAsDouble(1, i, i);
		}
		weightMatrix = tWeightMatrix;
		
		return weightMatrix;
	}

	public double[] computeLaplacianScore() {
		int n = (int) data.getRowCount();
		int d = (int) data.getColumnCount();
		Matrix unitColumnVector = Matrix.Factory.ones(n, 1);
		Matrix DVector = weightMatrix.mtimes(unitColumnVector);
		double dSum = DVector.sum(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);
		Matrix D = DVector.diag(Ret.NEW);
		Matrix L = D.minus(weightMatrix);
		double[] tLaplacianScore = new double[d];
		for (int i = 0; i < data.getColumnCount(); i++) {
			Matrix fr = data.selectColumns(Ret.NEW, i);
			Matrix _fr_ = fr.minus(unitColumnVector.times(fr.transpose()
					.mtimes(DVector).getAsDouble(0, 0)
					/ dSum));
			tLaplacianScore[i] = _fr_.transpose().mtimes(L).mtimes(_fr_)
					.getAsDouble(0, 0)
					/ _fr_.transpose().mtimes(D).mtimes(_fr_).getAsDouble(0, 0);
		}
		laplacianScore = tLaplacianScore;
		return laplacianScore;
	}

	/**
	 * The features with smaller laplacian score are more important.
	 */
	public int[] computeFeatureRanking() {
		featureSubset = Utils.argSort(laplacianScore, Order.ASC);
		return featureSubset;
	}

	@Override
	public Matrix getDataAfterFeaturesSelected() {
		return data.selectColumns(Ret.NEW, Utils.intArrayToLongType(Arrays
				.copyOf(featureSubset, numFeatures)));
	}

	@Override
	public int[] getFeatureSubset() {
		return Arrays.copyOf(featureSubset, numFeatures);
	}

	public double getT() {
		return t;
	}

	public void setT(double t) {
		this.t = t;
	}

	public Matrix getNeighborIndicesMatrix() {
		return neighborIndicesMatrix;
	}

	public Matrix getWeightMatrix() {
		return weightMatrix;
	}

	public double[] getLaplacianScore() {
		return laplacianScore;
	}
	public static void main(String[] args) throws Exception {
		
		weka.core.Instances iris = new weka.core.Instances(new java.io.FileReader("src/data/arff/iris.arff"));
		iris.setClassIndex(4);
		Matrix m = Utils.instancesToMatrixWithLabel(iris);
		m = m.deleteColumns(Ret.NEW, 4);
		m = m.normalize(Ret.NEW, 0);

		LaplacianScoreForUnsupervisedFS ls = new LaplacianScoreForUnsupervisedFS(m, Utils.kNeighborsIndicesMatrix(m, 3), 4);
		ls.middleProcess();
		System.out.println(Arrays.toString(ls.getLaplacianScore()));
		System.out.println(Arrays.toString(ls.getFeatureSubset()));
				
	}

	@Override
	public void middleProcess() {
		computeWeightMatrix();
		computeLaplacianScore();
		computeFeatureRanking();
	}
}
