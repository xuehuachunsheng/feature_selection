package ufs.reducedim.impl;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.reducedim.DimensionalReduction;
import ufs.utils.ConstValues;

/**
 * A basic implementation of 'Laplacian Eigenmaps'. 
 * @author Yanxue
 *
 */
public class LE extends DimensionalReduction {

	/**
	 * The bandwidth of Gaussian kernel.
	 */
	double t;
	
	/**
	 * The number of dimension after reduced.
	 */
	int numDimension;
	
	/**
	 * The weight matrix W.
	 */
	Matrix weightMatrix;
	
	
	Matrix neighborIndicesMatrix;
	
	public LE(Matrix pData) {
		super(pData);
		defaultInitialize();
	}

	public LE(Matrix pData, Matrix pNeighborIndicesMatrix) {
		super(pData);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
	}
	
	public LE(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumDimension) {
		super(pData);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
		numDimension = pNumDimension;
	}
	public void defaultInitialize() {
		t = ConstValues.LS_CONSTANT;
		numDimension = 2;
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
		}
		weightMatrix = tWeightMatrix;
		
		return weightMatrix;
	}
	
	@Override
	public void middleProcess() {
		// TODO: 
	}

	public double getT() {
		return t;
	}

	public void setT(double t) {
		this.t = t;
	}

	public int getNumDimension() {
		return numDimension;
	}

	public void setNumDimension(int numDimension) {
		this.numDimension = numDimension;
	}

	public Matrix getNeighborIndicesMatrix() {
		return neighborIndicesMatrix;
	}

	public void setNeighborIndicesMatrix(Matrix neighborIndicesMatrix) {
		this.neighborIndicesMatrix = neighborIndicesMatrix;
	}

}
