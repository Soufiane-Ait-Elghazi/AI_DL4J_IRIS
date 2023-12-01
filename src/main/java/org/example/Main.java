package org.example;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;

public class Main {


    public static void main(String[] args) throws Exception {
        double learningRate = 0.001;
        int numInputs = 4;
        int numHidden = 10;
        int numOutputs = 3;
        int batchSize = 1;
        int classIndex = 4;
        int nEpochs = 80;


        System.out.println("Model CREATION ...");

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learningRate))
                .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHidden).activation(Activation.SIGMOID).build())
                    .layer(1, new OutputLayer.Builder().nIn(numHidden).nOut(numOutputs).activation(Activation.SOFTMAX).build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        //System.out.println(configuration.toJson());


        //Monitoring
        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));



        System.out.println("Model TRAINING ...");
        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex,numOutputs);
       /* while(dataSetIteratorTrain.hasNext()){
            DataSet dataSet = dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }*/
        for(int i = 0 ; i< nEpochs ; i++ ){
            model.fit(dataSetIteratorTrain);
        }


        System.out.println("Model EVALUATION ...");

        File fileTest= new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,numOutputs);
        Evaluation evaluation=new Evaluation();
        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labels=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());

        System.out.println("Loading Model");
        ModelSerializer.writeModel(model,"irisModel.zip",true);












    }










}