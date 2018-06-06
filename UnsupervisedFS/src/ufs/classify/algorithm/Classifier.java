package ufs.classify.algorithm;
/**
 * 
 * @author YanXue
 *
 */
public interface Classifier {
	
	int[] getClassifyResults();
	
	String getInfo();
	
	double getClassifyAccuracy();
	
}
