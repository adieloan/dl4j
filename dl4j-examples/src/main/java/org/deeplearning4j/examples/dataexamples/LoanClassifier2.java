package org.deeplearning4j.examples.dataexamples;


import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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
public class LoanClassifier2 {

    private static Logger log = LoggerFactory.getLogger(LoanClassifier2.class);

       private static Map<Integer,String> classifiers = readEnumCSV("/DataExamples/loans/classifiers.csv");

    public static void main(String[] args){

        try {
        	int[][] hits=new int[3][3];
        	for(int i=0;i<3;++i)
        		for(int j=0;j<3;++j)
        			hits[i][j]=0;
        	List<Integer> dataResults=readKnownResult("/DataExamples/loans/result7data.csv");
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int labelIndex = 20;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2

            int batchSizeTraining = 20;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            DataSet trainingData = readCSVDataset(
                    "/DataExamples/loans/result6.csv",
                    batchSizeTraining, labelIndex, numClasses);

            // this is the data we want to classify
            int batchSizeTest = 20;
            DataSet testData = readCSVDataset("/DataExamples/loans/result7.csv",
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
            int iterations = 10000;
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
                    .layer(1, new DenseLayer.Builder().nIn(20).nOut(2).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(2).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            InMemoryStatsStorage inMem=new InMemoryStatsStorage();
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(inMem);
            
            model.setListeners(new ScoreIterationListener(100), new StatsListener(inMem,10));

            model.fit(trainingData);
            List<String> ids=inMem.listSessionIDs();
            for(String sessionID:ids){
            	List<String> typeIds=inMem.listTypeIDsForSession(sessionID);
            	for(String typeID:typeIds){
            		List<String> workerIds=inMem.listWorkerIDsForSessionAndType(sessionID, typeID);
            		for(String workerID:workerIds){
            			Persistable stat=inMem.getLatestUpdate(sessionID, typeID, workerID);
            			System.out.println(stat);
            		}
            	List<Persistable> stat=inMem.getAllStaticInfos(sessionID,typeID);
            	System.out.println(stat.size()+":"+stat);
            	}
            }
            INDArray array = model.params();
            for(int i=0;i<array.columns();++i){
            	System.out.println(array.getDouble(i));
            }
            Map<String, INDArray> map = model.paramTable();
            for(String key:map.keySet()){
            	array =map.get(key);
            	System.out.println("key"+key);
            	 for(int i=0;i<array.columns();++i){
                 	System.out.println(array.getDouble(i));
                 }
            }
            /*
            for(int l=0;l<3;++l){
            	String param=String.valueOf(l)+"_1";
            	System.out.println("param="+param);
                array = model.getParam(param);
                for(int i=0;i<array.columns();++i){
                	System.out.println(array.getDouble(i));
                }
            }
            */
            File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
            boolean saveUpdater = true;  
            //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
/*
            //evaluate the model on the test set
            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatureMatrix());

            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());

            setFittedClassifiers(output,hits,dataResults, animals);
            logAnimals(animals);
            for(int i=0;i<3;++i){
            	System.out.println("\r\n");
            	for(int j=0;j<3;++j){
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
