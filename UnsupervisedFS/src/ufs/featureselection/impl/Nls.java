package ufs.featureselection.impl;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.reducedim.impl.Lle;

/**
 * LLE based Laplacian score.
 * @author Yanxue
 *
 */
public class Nls extends LaplacianScoreForUnsupervisedFS{

	public Nls(Matrix pData, Matrix pNeighborIndicesMatrix, int pNumFeatures) {
		super(pData, pNeighborIndicesMatrix, pNumFeatures);
		neighborIndicesMatrix = pNeighborIndicesMatrix;
	}
	
	@Override
	public Matrix computeWeightMatrix() {
		Lle lle = new Lle(data, neighborIndicesMatrix);
		weightMatrix = lle.computeCoefficientMatrix();
//		for (int i = 0; i < weightMatrix.getRowCount(); i++) {
//			weightMatrix.setAsDouble(1, i, i);
//		}
//		System.out.println("OK3");
		return weightMatrix;
	}
	
//	public double[] computeLaplacianScore() {
//		int n = (int) data.getRowCount();
//		int d = (int) data.getColumnCount();
//		Matrix unitColumnVector = Matrix.Factory.ones(n, 1);
//		
//		Matrix normWeightMatrix = weightMatrix.normalize(Ret.NEW, 1);
//		
//		Matrix DVector = normWeightMatrix.mtimes(unitColumnVector);
//		double dSum = DVector.sum(Ret.NEW, Matrix.ALL, false).getAsDouble(0, 0);
//		Matrix D = DVector.diag(Ret.NEW);
//		Matrix L = D.minus(normWeightMatrix);
//		double[] tLaplacianScore = new double[d];
//		for (int i = 0; i < data.getColumnCount(); i++) {
//			Matrix fr = data.selectColumns(Ret.NEW, i);
//			Matrix _fr_ = fr.minus(unitColumnVector.times(fr.transpose()
//					.mtimes(DVector).getAsDouble(0, 0)
//					/ dSum));
//			tLaplacianScore[i] = _fr_.transpose().mtimes(L).mtimes(_fr_)
//					.getAsDouble(0, 0)
//					/ _fr_.transpose().mtimes(D).mtimes(_fr_).getAsDouble(0, 0);
//		}
//		laplacianScore = tLaplacianScore;
//		return laplacianScore;
//	}
}
