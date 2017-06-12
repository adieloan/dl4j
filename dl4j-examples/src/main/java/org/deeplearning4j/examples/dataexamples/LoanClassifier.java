package org.deeplearning4j.examples.dataexamples;


import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



/**
 * This example is intended to be a simple CSV classifier that seperates the training data
 * from the test data for the classification of animals. It would be suitable as a beginner's
 * example because not only does it load CSV data into the network, it also shows how to extract the
 * data and display the results of the classification, as well as a simple method to map the lables
 * from the testing data into the results.
 *
 * @author Clay Graham
 */
public class LoanClassifier {

    private static Logger log = LoggerFactory.getLogger(LoanClassifier.class);

       private static Map<Integer,String> classifiers = readEnumCSV("/DataExamples/loans/classifiers.csv");

    public static void main(String[] args){

        try {
        	int[][] hits=new int[2][2];
        	for(int i=0;i<2;++i)
        		for(int j=0;j<2;++j)
        			hits[i][j]=0;
        	List<Integer> dataResults=readKnownResult("/DataExamples/loans/result2data.csv");
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int labelIndex = 20;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2

            int batchSizeTraining = 1000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            DataSet trainingData = readCSVDataset(
                    "/DataExamples/loans/result1.csv",
                    batchSizeTraining, labelIndex, numClasses);

            // this is the data we want to classify
            int batchSizeTest = 265;
            DataSet testData = readCSVDataset("/DataExamples/loans/result2.csv",
                    batchSizeTest, labelIndex, numClasses);


            // make the data model for records prior to normalization, because it
            // changes the data.
            Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);


            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainingData);     //Apply normalization to the training data
            normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

            final int numInputs = 20;
            int outputNum = 2;
            int iterations = 12000;
            long seed = 6;

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation(Activation.TANH)                    		
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(20).build())
                    .layer(1, new DenseLayer.Builder().nIn(20).nOut(4).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(4).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            model.fit(trainingData);
            Map<String, INDArray> map = model.paramTable();
            for(String key:map.keySet()){
            	INDArray array =map.get(key);
            	System.out.println("key"+key);
            	 for(int i=0;i<array.columns();++i){
                 	System.out.println(array.getDouble(i));
                 }
            }
            File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
            boolean saveUpdater = true;  
            //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            /*
             * key0_W
-0.11181222647428513 Amount
0.19276443123817444 NumOfPayment
-0.2559468448162079 age
-0.3116510212421417 homeOwnershipTypeId
0.08618045598268509 workAgeId
0.352331280708313 familyStatusId
-0.8313689231872559 numberOfKids
0.3380129337310791 paymentDayOfMonth
0.1984347552061081 bankCode
-1.894912838935852 bankBranch
0.06417114287614822 academicLevelId
-0.3262596130371094 professionStatusId
0.1954992413520813 familyIncomeRangeId
0.1377146989107132 bankCreditCard
0.1240490972995758 bankAccountAge
-0.42371395230293274 cityId
-0.0916365459561348 gender
-0.2575634717941284 carOwnershipTypeId
0.3315960168838501 chequeReturned
-0.023229477927088737 loanPurpose
key0_b
-0.2465350478887558
-0.3408517837524414
0.30546504259109497
0.04278460890054703
-0.26591384410858154
0.20604930818080902
0.24756143987178802
0.4662345349788666
0.3177718222141266
-0.46440255641937256
0.3180154860019684
-0.23680229485034943
0.11595917493104935
0.25066515803337097
-0.07659965008497238
0.2814006507396698
-0.38805776834487915
0.0248136967420578
-0.13290007412433624
-0.3279922604560852
*/
           
            /*
            //evaluate the model on the test set
            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatureMatrix());

            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());

            setFittedClassifiers(output,hits,dataResults, animals);
            logAnimals(animals);
            for(int i=0;i<2;++i){
            	System.out.println("\r\n");
            	for(int j=0;j<2;++j){
            		System.out.print(hits[i][j]+" , ");
            	}
            }	
            */
/*
202 , 0 , 38 , 

0 , 0 , 0 , 

20 , 0 , 5 ,
*/ 
        } catch (Exception e){
            e.printStackTrace();
        }

    }



    public static void logAnimals(Map<Integer,Map<String,Object>> animals){
        for(Map<String,Object> a:animals.values())
            log.info(a.toString());
    }

    public static void setFittedClassifiers(INDArray output, int[][] hits, List<Integer> dataResults, Map<Integer,Map<String,Object>> animals){
        for (int i = 0; i < output.rows() ; i++) {
        	Integer index=maxIndex(getFloatArrayFromSlice(output.slice(i)));
        	Integer shouldBe=dataResults.get(i);
        	hits[shouldBe][index]++;
            // set the classification from the fitted results
            animals.get(i).put("classifier",
                    classifiers.get(index));

        }

    }


    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     *
     * @param vals
     * @return
     */
    public static int maxIndex(float[] vals){
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++){
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     *
     * @param testData
     * @return
     */
    public static Map<Integer,Map<String,Object>> makeAnimalsForTesting(DataSet testData){
        Map<Integer,Map<String,Object>> animals = new HashMap<>();

        INDArray features = testData.getFeatureMatrix();
        for (int i = 0; i < features.rows() ; i++) {
            INDArray slice = features.slice(i);
            Map<String,Object> animal = new HashMap();

            //set the attributes
            animal.put("Amount", slice.getInt(0));
            animal.put("NumOfPayment", slice.getInt(1));
            animal.put("age", slice.getInt(2));
            animal.put("homeOwnershipTypeId", slice.getInt(3));
            animal.put("workAgeId", slice.getInt(4));
            animal.put("familyStatusId", slice.getInt(5));
            animal.put("numberOfKids", slice.getInt(6));
            animal.put("paymentDayOfMonth", slice.getInt(7));
            animal.put("bankCode", slice.getInt(8));
            animal.put("bankBranch", slice.getInt(9));
            animal.put("academicLevelId", slice.getInt(10));
            animal.put("professionStatusId", slice.getInt(11));
            animal.put("familyIncomeRangeId", slice.getInt(12));
            animal.put("bankCreditCard", slice.getInt(13));
            animal.put("bankAccountAge", slice.getInt(14));
            animal.put("cityId", slice.getInt(15));
            animal.put("gender", slice.getInt(16));
            animal.put("carOwnershipTypeId", slice.getInt(17));
            animal.put("chequeReturned", slice.getInt(18));
            animal.put("loanPurpose", slice.getInt(19));
                      
            animals.put(i,animal);
        }
        return animals;

    }

    public static List<Integer> readKnownResult(String csvFileClasspath){
    	  try{
              List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
              List<Integer> result=new ArrayList<Integer>();
              for(String line:lines){
            	  result.add(Integer.parseInt(line));
              }
              return result;
    	  } catch (Exception e){
              e.printStackTrace();
              return null;
          }	
    }
    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }

    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(
            String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        return iterator.next();
    }



}
