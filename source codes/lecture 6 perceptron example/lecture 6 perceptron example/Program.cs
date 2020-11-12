using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Statistics.Kernels;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace lecture_6_perceptron_example
{
    class Program
    {
        static private bool saveStatisticsToFiles = false;
        static private volatile bool needToStop = false;

        static string srFileName = "iris.data";


        static void Main(string[] args)
        {
            //this below 3 lines will shuffle the data whenever i run the application
            List<string> lstFile = File.ReadAllLines(srFileName).ToList();

            lstFile = lstFile.OrderBy(a => Guid.NewGuid()).ToList();

            File.WriteAllLines(srFileName, lstFile);

            int irNumberOfExamples = File.ReadAllLines(srFileName).Count();

            //the first array holds all of the instances
            double[][] input = new double[irNumberOfExamples][];
            double[][] output = new double[irNumberOfExamples][];
            double[][] output2 = new double[irNumberOfExamples][];

            List<double> lstOutPutClasses = new List<double>();

            NumberFormatInfo formatProvider = new NumberFormatInfo();
            formatProvider.NumberDecimalSeparator = ".";
            formatProvider.NumberGroupSeparator = ",";

            foreach (var vrPerLine in File.ReadAllLines(srFileName))
            {
                var vrOutPut = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                if (lstOutPutClasses.Contains(vrOutPut) == false)
                {
                    lstOutPutClasses.Add(vrOutPut);
                }
            }

            int irCounter = 0;
            foreach (var vrPerLine in File.ReadAllLines(srFileName))
            {
                input[irCounter] = vrPerLine.Split(',').SkipLast(1).
                    Select(pr => Convert.ToDouble(pr.Replace("I", "0.0").Replace("M", "0.5").Replace("F", "1.0"), formatProvider)).ToArray();

                output2[irCounter] = new double[] { Convert.ToDouble(vrPerLine.Split(',').Last()) };

                output[irCounter] = new double[lstOutPutClasses.Count];

                var vrCurrentOutClass = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                output[irCounter][lstOutPutClasses.IndexOf(vrCurrentOutClass)] = 1;

                irCounter++;
            }



            int irFinalClassCount = lstOutPutClasses.Count;

            double learningRate = 0.1;
            int irNumberOfFeatures = input[0].Length;

            int[] numberOfNeurons = new int[] { 100, 100, irFinalClassCount };

            ThresholdFunction function = new ThresholdFunction();

            ActivationNetwork network3 = new ActivationNetwork(
          new SigmoidFunction(2),
           irNumberOfFeatures,
           12,// two inputs in the network
           irFinalClassCount); // one neuron in the second layer
                               // create teacher
            BackPropagationLearning bpteacher = new BackPropagationLearning(network3);
            // loop

            bpteacher.LearningRate = 0.1;
            bpteacher.Momentum = 0.5;

            for (int i = 0; i < 200000; i++)
            {
                double error = bpteacher.RunEpoch(input, output);

                var vrAcc = calculateAcurracy(network3, input, output);

                Console.WriteLine("BackPropagationLearning -> " + i + ", Error = " + error.ToString("N2") + "\t\t accuracy: " + vrAcc);
            }





            // create activation network
            ActivationNetwork network = new ActivationNetwork(
         function,
           irNumberOfFeatures,
           irFinalClassCount); // one neuron in the second layer

            //double initialStep = 0.125;
            //double sigmoidAlphaValue = 2.0;

            //// create neural network
            //ActivationNetwork neuralNetwork = new ActivationNetwork(new SigmoidFunction(sigmoidAlphaValue), irNumberOfFeatures, 100, irFinalClassCount);

            //// create teacher
            //ParallelResilientBackpropagationLearning neuralTeacher = new ParallelResilientBackpropagationLearning(neuralNetwork);

            //// set learning rate and momentum
            //neuralTeacher.Reset(initialStep);

            //for (int i = 0; i < 5000; i++)
            //{
            //    double error = neuralTeacher.RunEpoch(input, output);

            //        Console.WriteLine("Supervised -> " + i + ", Error = " + error);

            //}

            var Layers = network.Layers.Length;

            //int irCounter_2 = 0;
            //BackPropagationLearning teacher2 = new BackPropagationLearning(network);
            //// loop
            //while (true)
            //{
            //    // run epoch of learning procedure
            //    double error = teacher2.RunEpoch(input, output);
            //    // check error value to see if we need to stop

            //    Console.WriteLine("current iteration: " + irCounter_2.ToString("N0") + "error rate: " + error.ToString("N3"));
            //    // ...
            //    irCounter_2++;
            //}

            ActivationNeuron neuron = network.Layers[0].Neurons[0] as ActivationNeuron;
            // create teacher
            DeltaRuleLearning teacher = new DeltaRuleLearning(network);
            // set learning rate
            teacher.LearningRate = learningRate;

            // iterations
            int iteration = 1;

            // statistic files
            StreamWriter errorsFile = null;
            StreamWriter weightsFile = null;

            // check if we need to save statistics to files
            if (saveStatisticsToFiles)
            {
                // open files
                errorsFile = File.CreateText("errors.csv");
                weightsFile = File.CreateText("weights.csv");
                weightsFile.AutoFlush = true;
            }

            // erros list
            ArrayList errorsList = new ArrayList();

            // loop
            while (!needToStop)
            {
                // save current weights
                if (weightsFile != null)
                {
                    for (int i = 0; i < irNumberOfFeatures; i++)
                    {
                        weightsFile.Write(neuron.Weights[i] + ",");
                    }
                    weightsFile.WriteLine(neuron.Threshold);
                }

                // run epoch of learning procedure
                double error = teacher.RunEpoch(input, output);

                errorsList.Add(error);

                // show current iteration
                Console.WriteLine("current iteration: " + iteration.ToString("N0") + "error rate: " + error.ToString("N3"));


                // save current error
                if (errorsFile != null)
                {
                    errorsFile.WriteLine(error);
                }

                // stop if no error
                if (error == 0)
                    break;

                iteration++;
            }

            Console.ReadLine();
        }


        static string calculateAcurracy(ActivationNetwork network3, double[][] input,
            double[][] output)
        {
            double correct = 0;
       
            for (int i = 0; i < input.Length; i++)
            {
                int irBiggestIndex = -1;
                double dblBiggestValue = double.MinValue;
                double[] outputValues = network3.Compute(input[i]);

                for (int kk = 0; kk < outputValues.Length; kk++)
                {
                    if (outputValues[kk] > dblBiggestValue)
                    {
                        irBiggestIndex = kk;
                        dblBiggestValue = outputValues[kk];
                    }
                }


                if (1 == output[i][irBiggestIndex])
                {
                    correct++;
                }
            }

            return (correct / Convert.ToDouble(input.Length) * 100).ToString("N2") + "%";
        }

    }




}
