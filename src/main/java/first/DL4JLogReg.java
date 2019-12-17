package first;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class DL4JLogReg {

    public static void main(String[] args) {

//        int numOfFeatures = (int) Math.pow(2, 29);
        int numOfFeatures = 4;

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();

        OutputLayer outputLayer = new OutputLayer.Builder()
                .nIn(numOfFeatures) //The number of inputs feed from the input layer
                .nOut(1) //The number of output values the output layer is supposed to take
                .weightInit(WeightInit.NORMAL) //The algorithm to use for weights initialization
                .activation(Activation.SIGMOID) //Softmax activate converts the output layer into a probability distribution
                .lossFunction(LossFunctions.LossFunction.XENT)
                .build(); //Building our output layer

        MultiLayerConfiguration conf = builder.updater(new Sgd(0.05)).seed(1).list(outputLayer).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));


            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, 6, 4, 1);
            DataSet allData = iterator.next();
            allData.shuffle(42);


            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            network.fit(trainingData);
            INDArray params = network.params();
            System.out.println(params);
            System.out.println(params.shapeInfoToString());


            // …
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }




    }

}
