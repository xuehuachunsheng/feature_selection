package ufs.general.test;

import org.ujmp.core.Matrix;

public class TestSVD {
	public static void main(String[] args) {
		int[][] m = new int[][]{
				{2, 15, 13, 4}, 
				{10, 4, 14, 15},
				{9, 14, 16, 13},
				
		};
		Matrix m1 = Matrix.Factory.importFromArray(m);
		
		Matrix[] svd = m1.svd();
		System.out.println(svd[0]);
		System.out.println(svd[1]);
		System.out.println(svd[2]);
		System.out.println(svd[0].mtimes(svd[1]).mtimes(svd[2].transpose()));
	}
}
