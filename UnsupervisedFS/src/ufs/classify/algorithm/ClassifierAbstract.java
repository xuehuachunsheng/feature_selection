package ufs.classify.algorithm;

import org.ujmp.core.Matrix;

public abstract class ClassifierAbstract implements Classifier {
	
	Matrix data;
	
	int[] pLabels;
	
	int[] rLabels;
	
	int k;

	public ClassifierAbstract(Matrix pData) {
		data = pData;
	}
	
	public Matrix getData() {
		return data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}

	public int[] getpLabels() {
		return pLabels;
	}

	public void setpLabels(int[] pLabels) {
		this.pLabels = pLabels;
	}

	public int[] getrLabels() {
		return rLabels;
	}

	public void setrLabels(int[] rLabels) {
		this.rLabels = rLabels;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}
	
	
}