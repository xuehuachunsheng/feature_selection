package ufs.reducedim;

import org.ujmp.core.Matrix;

/**
 * The general interface of dimensional reduction
 * 
 * @author Yanxue
 *
 */
public abstract class DimensionalReduction {

	protected Matrix data;

	protected Matrix reducedData;
	
	public Matrix getData() {
		return data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}

	public Matrix getReducedData() {
		return reducedData;
	}

	public DimensionalReduction(Matrix pData) {
		data = pData;
	}

	/**
	 * Compute the matrix after dimension reduced.
	 */
	public Matrix reduceDimension() {
		middleProcess();
		return reducedData;
	}

	
	public abstract void middleProcess();
}
