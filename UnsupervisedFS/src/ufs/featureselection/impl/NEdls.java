package ufs.featureselection.impl;

import java.io.File;
import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.reducedim.impl.Lle;
import ufs.utils.ConstValues;
import ufs.utils.Utils;


/**
 * Neighborhood preserving effective distance-based Feature selection.
 * @author Yanxue
 *
 */
public class NEdls extends Edls {

	/**
	 * R^n×k, where n is the number of samples and k is the number of neighbors.
	 * The element located in i-th row and j-th column stores the index of the
	 * j-th neighbor of the i-th sample.
	 * 
	 */
	protected Matrix neighborIndicesMatrix;
	
	public NEdls(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNumFeatures);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
	}
	
	@Override
	public Matrix computeCoefficientMatrix() {
		Lle lle = new Lle(data, neighborIndicesMatrix);
		coefficientMatrix = lle.computeCoefficientMatrix().transpose();
		// Diagonal to 1
		for (int i = 0; i < coefficientMatrix.getRowCount(); i++) {
			coefficientMatrix.setAsDouble(1,  i, i);
		}
		return coefficientMatrix;
	}

	public static void main(String[] args) throws Exception {

		Matrix m = Utils.loadMatrix2DFromMat(new File(
				ConstValues.DATA_MATRIX_PATH
						+ ConstValues.STD_WARPAR_MATRIX_130$2400$10), "X");
		Matrix neighbors = Utils.load2DMatrixFromTxt(new File("src/data/mat/npfs/neighbors/WarpAR10P_std_Neighbors129.data"), " ", Integer.class);
		int numNeighbors = 9;
		long[] selectNeighbors = new long[numNeighbors];
		for (int i = 0; i < selectNeighbors.length; i++) {
			selectNeighbors[i] = i;
		}
		NEdls ls = new NEdls(m, neighbors.selectColumns(Ret.NEW, selectNeighbors), 20);
		ls.middleProcess();
		System.out.println(Arrays.toString(ls.getEdls()));
		System.out.println(Arrays.toString(ls.getFeatureSubset()));

	}
}
