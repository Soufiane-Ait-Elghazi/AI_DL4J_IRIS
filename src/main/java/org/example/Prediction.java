package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class Prediction {
    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new
                File("C:\\Users\\soufi\\IdeaProjects\\DL4J_TP\\irisModel.zip"));

       System.out.println("Prediction");
        String[] labels={"Iris-setosa","Iris-versicolor","Iris-virginica"};
        INDArray input= Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2}, {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4}, {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8}, {6.9,3.1,5.4,2.1}
        });
        System.out.println("**************");
        INDArray output= model.output(input);
        INDArray classes=output.argMax(1);
        System.out.println(output);
        System.out.println("-----------------");
        System.out.println(classes);
        System.out.println("****************");
        int[] predictions=classes.toIntVector();
        for (int i = 0; i < predictions.length; i++) {
            System.out.println(labels[predictions[i]]);
        }
    }
}
