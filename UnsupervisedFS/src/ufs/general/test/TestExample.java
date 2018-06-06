package ufs.general.test;

import java.io.File;
import java.io.RandomAccessFile;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.utils.ConstValues;
import ufs.utils.Utils;
import weka.core.Instances;

public class TestExample {
	public static void main(String[] args) throws Exception{
		
		for (int i = 0; i < ConstValues.DATA_MATRIX.length; i++) {
			Matrix X = Utils.loadMatrix2DFromMat(new File("src/data/mat/"+ConstValues.DATA_MATRIX[i]), "X");
			Matrix Y = Utils.loadMatrix2DFromMat(new File("src/data/mat/"+ConstValues.DATA_MATRIX[i]), "Y");
			
			Instances instances = Utils.matrixWithLabelToInstances(X.appendHorizontally(Ret.LINK, Y), "relation");
			
			File file = new File("src/data/mat/arff/"+ConstValues.DATA_MATRIX[i].substring(0, ConstValues.DATA_MATRIX[i].length()-4)+".arff");
			if(!file.exists()) {
				file.createNewFile();
			}
			RandomAccessFile randomAccessFile= new RandomAccessFile(file, "rw");
			
			randomAccessFile.writeBytes(instances.toString());
			randomAccessFile.close();
		}
		
	}
}
