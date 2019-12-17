package first;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DL4JLogReg {

    public static void main(String[] args) {

        int numOfFeatures = (int) Math.pow(2, 29);

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

        network.fit();



    }

}
