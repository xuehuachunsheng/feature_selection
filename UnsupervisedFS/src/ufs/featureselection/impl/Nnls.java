package ufs.featureselection.impl;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * Nonnegative Neighborhood Laplacian score.
 * 
 * @author Yanxue
 *
 */
public class Nnls extends Nls {

	public Nnls(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNeighborIndicesMatrix, pNumFeatures);
	}

	@Override
	public Matrix computeWeightMatrix() {
		Npfs npfs = new Npfs(data, neighborIndicesMatrix, 0);
		weightMatrix = npfs.optimalW().transpose(Ret.NEW);
		return weightMatrix;
	}
}
