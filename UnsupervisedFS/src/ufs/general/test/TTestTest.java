package ufs.general.test;


import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.general.algorithm.TTest;

public class TTestTest {
	public static void main(String[] args) {
		double[][] array = {
				{0.4351,0.3930,0.2633,0.3012,0.6046,0.4700,0.5278,0.5276,0.4403},
				{0.3368,0.4675,0.3933,0.3552,0.5799,0.6040,0.5823,0.5212,0.4800},
				{0.4070,0.4888,0.3343,0.3921,0.5017,0.5820,0.5845,0.5114,0.4752},
				{0.3439,0.4865,0.3981,0.4436,0.4190,0.5980,0.3044,0.3842,0.4222},
				{0.5193,0.4665,0.3933,0.4358,0.5437,0.5860,0.5766,0.5233,0.4942},
				{0.5082,0.4785,0.3348,0.4242,0.4017,0.6040,0.6017,0.5242,0.4847},
				{0.5216,0.4753,0.3124,0.4115,0.5799,0.5960,0.5441,0.5165,0.4947},
		};
		Matrix matrix = Matrix.Factory.importFromArray(array).transpose(Ret.NEW);
		System.out.println(matrix);
		
		for (int i = 0; i < matrix.getColumnCount() - 2; i++) {
			System.out.println(1 - new TTest(matrix.selectColumns(Ret.NEW, matrix.getColumnCount()-2), matrix.selectColumns(Ret.NEW, i)).pairedTTest());
		}
	}
}
