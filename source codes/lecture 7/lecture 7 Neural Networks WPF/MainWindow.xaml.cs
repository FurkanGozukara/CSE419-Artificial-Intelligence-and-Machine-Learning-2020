using Accord.Neuro;
using Accord.Neuro.Learning;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace lecture_7_Neural_Networks_WPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        static private bool saveStatisticsToFiles = false;
        static private volatile bool needToStop = false;
        static string srDataSetName = "iris";
        static string srDataSetPath = $"{srDataSetName}.data";
        static string srModelSavePath = $"{srDataSetName}_model.txt";
        static string srModelPropertiesPath = $"{srDataSetName}_model_prop.txt";
        static string srNetWorkDetailsPath = $"{srDataSetName}_model_network.txt";
        static string srTeacherDetailsPath = $"{srDataSetName}_model_teacher.txt";

        private void btnBuildModel_Click(object sender, RoutedEventArgs e)
        {
            Task.Factory.StartNew(() =>
            {
                buildTrainigModel();
            });
        }

        public class csMultiTraining
        {
            public string srFileName = "";
            public double dblAcc = double.MinValue;
            public ActivationNetwork acNet = null;
            public string srPropText = "";
            public int irBestAccIteration = 0;
        }

        private csMultiTraining buildTrainigModel(string srTrainingFileName, string srTestingFile,
            int irMaxIterationCount = 1000000)
        {
            Directory.CreateDirectory("temp_model_savings");

            string srTempModelPath = "temp_model_savings/" + ComputeSha256Hash(srTrainingFileName) + ".txt";

            string srTempModelPropPath = "temp_model_savings/prop_" + ComputeSha256Hash(srTrainingFileName) + ".txt";

            int irNumberOfExamples = File.ReadAllLines(srTrainingFileName).Count();

            double[][] input_training = new double[irNumberOfExamples][];
            double[][] output_training = new double[irNumberOfExamples][];

            double[][] input_testing = new double[irNumberOfExamples][];
            double[][] output_testing = new double[irNumberOfExamples][];

            List<double> lstOutPutClasses = new List<double>();

            NumberFormatInfo formatProvider = new NumberFormatInfo();
            formatProvider.NumberDecimalSeparator = ".";
            formatProvider.NumberGroupSeparator = ",";

            foreach (var vrPerLine in File.ReadAllLines(srTrainingFileName))
            {
                var vrOutPut = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                if (lstOutPutClasses.Contains(vrOutPut) == false)
                {
                    lstOutPutClasses.Add(vrOutPut);
                }
            }

            foreach (var vrPerLine in File.ReadAllLines(srTestingFile))
            {
                var vrOutPut = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                if (lstOutPutClasses.Contains(vrOutPut) == false)
                {
                    lstOutPutClasses.Add(vrOutPut);
                }
            }

            int irFinalClassCount = lstOutPutClasses.Count;

            int irCounter = 0;
            foreach (var vrPerLine in File.ReadAllLines(srTrainingFileName))
            {
                input_training[irCounter] = vrPerLine.Split(',').SkipLast(1).
                    Select(pr => Convert.ToDouble(pr.Replace("I", "0.0").Replace("M", "0.5").Replace("F", "1.0"), formatProvider)).ToArray();

                output_training[irCounter] = new double[lstOutPutClasses.Count];

                var vrCurrentOutClass = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                output_training[irCounter][lstOutPutClasses.IndexOf(vrCurrentOutClass)] = 1;

                irCounter++;
            }


            irCounter = 0;
            foreach (var vrPerLine in File.ReadAllLines(srTestingFile))
            {
                input_testing[irCounter] = vrPerLine.Split(',').SkipLast(1).
                    Select(pr => Convert.ToDouble(pr.Replace("I", "0.0").Replace("M", "0.5").Replace("F", "1.0"), formatProvider)).ToArray();

                output_testing[irCounter] = new double[lstOutPutClasses.Count];

                var vrCurrentOutClass = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                output_testing[irCounter][lstOutPutClasses.IndexOf(vrCurrentOutClass)] = 1;

                irCounter++;
            }

            int irNumberOfFeatures = input_training[0].Length;

            ActivationNetwork network3 = new ActivationNetwork(
            new SigmoidFunction(2),
           irNumberOfFeatures,
           12,// two inputs in the network
           irFinalClassCount); // one neuron in the second layer
                               // create teacher

            string srNetWorkData = JsonConvert.SerializeObject(network3);

            //File.WriteAllText(srNetWorkDetailsPath, srNetWorkData);

            BackPropagationLearning bpteacher = new BackPropagationLearning(network3);

            bpteacher.LearningRate = 0.1;
            bpteacher.Momentum = 0.5;

            string srTeacherData = JsonConvert.SerializeObject(bpteacher);

            //File.WriteAllText(srTeacherDetailsPath, srTeacherData);

            double dblMaxAcc = double.MinValue;
            int irBestIteration = 0;
            for (int i = 0; i < irMaxIterationCount; i++)
            {
                var vrAcc = calculateAcurracy(network3, input_testing, output_testing, lstOutPutClasses);

                if (vrAcc > dblMaxAcc)
                {
                    network3.Save(srTempModelPath);
                    saveModelProperties(vrAcc, i, srTempModelPropPath);
                    dblMaxAcc = vrAcc;
                    irBestIteration = i;
                    if (vrAcc >= 100.0)
                        break;
                }

                string srAccText = vrAcc.ToString("N2") + "%";

                double error = bpteacher.RunEpoch(input_training, output_training);

                if (i % 1000 == 0)
                {
                    var vrMsg = srTestingFile + "\t\t BackPropagationLearning -> " + i.ToString("N0") + ", Error = " + error.ToString("N2") + "\t\t accuracy: " + srAccText;
                    updateLabel(eLabel.lstBox, vrMsg);
                }
            }

            return new csMultiTraining
            { dblAcc = dblMaxAcc, acNet = network3, srPropText = File.ReadAllText(srTempModelPropPath), irBestAccIteration = irBestIteration, srFileName = srTrainingFileName };
        }

        static string ComputeSha256Hash(string rawData)
        {
            // Create a SHA256   
            using (SHA256 sha256Hash = SHA256.Create())
            {
                // ComputeHash - returns byte array  
                byte[] bytes = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(rawData));

                // Convert byte array to a string   
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < bytes.Length; i++)
                {
                    builder.Append(bytes[i].ToString("x2"));
                }
                return builder.ToString();
            }
        }

        private void buildTrainigModel()
        {
            //this below 3 lines will shuffle the data whenever i run the application
            List<string> lstFile = File.ReadAllLines(srDataSetPath).ToList();

            lstFile = lstFile.OrderBy(a => Guid.NewGuid()).ToList();

            //File.WriteAllLines(srFileName, lstFile);

            int irNumberOfExamples = File.ReadAllLines(srDataSetPath).Count();

            //the first array holds all of the instances
            double[][] input = new double[irNumberOfExamples][];
            double[][] output = new double[irNumberOfExamples][];

            List<double> lstOutPutClasses = new List<double>();

            NumberFormatInfo formatProvider = new NumberFormatInfo();
            formatProvider.NumberDecimalSeparator = ".";
            formatProvider.NumberGroupSeparator = ",";

            foreach (var vrPerLine in File.ReadAllLines(srDataSetPath))
            {
                var vrOutPut = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                if (lstOutPutClasses.Contains(vrOutPut) == false)
                {
                    lstOutPutClasses.Add(vrOutPut);
                }
            }

            int irCounter = 0;
            foreach (var vrPerLine in File.ReadAllLines(srDataSetPath))
            {
                input[irCounter] = vrPerLine.Split(',').SkipLast(1).
                    Select(pr => Convert.ToDouble(pr.Replace("I", "0.0").Replace("M", "0.5").Replace("F", "1.0"), formatProvider)).ToArray();

                output[irCounter] = new double[lstOutPutClasses.Count];

                var vrCurrentOutClass = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                output[irCounter][lstOutPutClasses.IndexOf(vrCurrentOutClass)] = 1;

                irCounter++;
            }

            int irFinalClassCount = lstOutPutClasses.Count;

            double learningRate = 0.1;
            int irNumberOfFeatures = input[0].Length;

            int[] numberOfNeurons = new int[] { 100, 100, irFinalClassCount };



            ActivationNetwork network3 = new ActivationNetwork(
            new SigmoidFunction(2),
           irNumberOfFeatures,
           12,// two inputs in the network
           irFinalClassCount); // one neuron in the second layer
                               // create teacher

            string srNetWorkData = JsonConvert.SerializeObject(network3);

            File.WriteAllText(srNetWorkDetailsPath, srNetWorkData);

            BackPropagationLearning bpteacher = new BackPropagationLearning(network3);

            bpteacher.LearningRate = 0.1;
            bpteacher.Momentum = 0.5;

            string srTeacherData = JsonConvert.SerializeObject(bpteacher);

            File.WriteAllText(srTeacherDetailsPath, srTeacherData);

            double dblMaxAcc = double.MinValue;

            for (int i = 0; i < 2000000; i++)
            {
                var vrAcc = calculateAcurracy(network3, input, output, lstOutPutClasses);

                if (vrAcc > dblMaxAcc)
                {
                    network3.Save(srModelSavePath);
                    saveModelProperties(vrAcc, i);
                    dblMaxAcc = vrAcc;
                }

                string srAccText = vrAcc.ToString("N2") + "%";

                double error = bpteacher.RunEpoch(input, output);

                if (i % 100 == 0)
                {
                    var vrMsg = "BackPropagationLearning -> " + i + ", Error = " + error.ToString("N2") + "\t\t accuracy: " + srAccText;
                    updateLabel(eLabel.label1, vrMsg);
                }
            }
        }

        static void saveModelProperties(double dblModelAccuracy, int irIterationIndex, string srCustomPath = null)
        {
            if (string.IsNullOrEmpty(srCustomPath))
                srCustomPath = srModelPropertiesPath;

            StringBuilder srProperties = new StringBuilder();
            srProperties.AppendLine($"Iteration: {irIterationIndex.ToString("N0")}\t\tAcurracy: {dblModelAccuracy.ToString("N2")}%");
            srProperties.AppendLine();
            File.WriteAllText(srCustomPath, srProperties.ToString());
        }

        static double calculateAcurracy(ActivationNetwork network3, double[][] input,
    double[][] output, List<double> lstOutPutClasses, List<double> lstGuessedClasses = null)
        {
            if (lstGuessedClasses == null)
                lstGuessedClasses = new List<double>();

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

                lstGuessedClasses.Add(lstOutPutClasses[irBiggestIndex]);

                if (1 == output[i][irBiggestIndex])
                {
                    correct++;
                }
            }

            return (correct / Convert.ToDouble(input.Length) * 100);
        }

        enum eLabel
        {
            label1,
            label2,
            lstBox
        }

        void updateLabel(eLabel lbl, string srMessage)
        {
            switch (lbl)
            {
                case eLabel.label1:
                    Dispatcher.Invoke(new Action(delegate ()
                    {
                        lblStatus1.Content = srMessage;
                    }));
                    break;
                case eLabel.lstBox:
                    Dispatcher.Invoke(new Action(delegate ()
                    {
                        lstResults.Items.Insert(0, srMessage);
                    }));
                    break;
                default:
                    break;
            }
        }

        private void btnSplitDataTestTrain_Click(object sender, RoutedEventArgs e)
        {
            splitData("iris.data", 4, ",", 10);
        }

        private void splitData(string srFileName, int irClassIndex, string srDataSeperator, int irSeperationPercent)
        {
            Directory.CreateDirectory("training");
            Directory.CreateDirectory("testing");

            Dictionary<string, List<string>> dicAllValues = new Dictionary<string, List<string>>();

            foreach (var vrLine in File.ReadLines("original_data/" + srFileName))
            {
                if (vrLine.Length < 2)
                    continue;

                List<string> lstSplitData = vrLine.Split(srDataSeperator).ToList();

                var vrClassKey = lstSplitData[irClassIndex];

                if (dicAllValues.ContainsKey(vrClassKey) == false)
                {
                    dicAllValues.Add(vrClassKey, new List<string> { vrLine });
                }
                else
                {
                    dicAllValues[vrClassKey].Add(vrLine);
                }
            }

            foreach (var vrKey in dicAllValues.Keys.ToList())
            {
                dicAllValues[vrKey] = dicAllValues[vrKey].OrderBy(a => Guid.NewGuid()).ToList();
            }

            List<string> lstTestVals = new List<string>();
            List<string> lstAllVals = new List<string>();

            foreach (var vrKey in dicAllValues.Keys.ToList())
            {
                int irNumberOfElementsInDic = dicAllValues[vrKey].Count;
                int irTestElementcount = irNumberOfElementsInDic * irSeperationPercent / 100;

                lstTestVals.AddRange(dicAllValues[vrKey].GetRange(0, irTestElementcount));

                lstAllVals.AddRange(dicAllValues[vrKey]);
            }

            //here i am removing the test data from the all dataset
            lstAllVals = lstAllVals.Where(pr => lstTestVals.Contains(pr) == false).ToList();

            File.WriteAllLines("training/" + srFileName, lstAllVals);
            File.WriteAllLines("testing/" + srFileName, lstTestVals);
        }

        private void btnTestUseenData_Click(object sender, RoutedEventArgs e)
        {
            string srDataPath = "testing/iris.data";

            int irNumberOfExamples = File.ReadAllLines(srDataPath).Count();

            //the first array holds all of the instances
            double[][] input = new double[irNumberOfExamples][];
            double[][] output = new double[irNumberOfExamples][];
            List<double> lstCorrectOutPutclass = new List<double>();

            List<double> lstOutPutClasses = new List<double>();

            NumberFormatInfo formatProvider = new NumberFormatInfo();
            formatProvider.NumberDecimalSeparator = ".";
            formatProvider.NumberGroupSeparator = ",";

            foreach (var vrPerLine in File.ReadAllLines(srDataPath))
            {
                var vrOutPut = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);
                lstCorrectOutPutclass.Add(vrOutPut);
                if (lstOutPutClasses.Contains(vrOutPut) == false)
                {
                    lstOutPutClasses.Add(vrOutPut);
                }
            }

            int irCounter = 0;
            foreach (var vrPerLine in File.ReadAllLines(srDataPath))
            {
                input[irCounter] = vrPerLine.Split(',').SkipLast(1).
                    Select(pr => Convert.ToDouble(pr.Replace("I", "0.0").Replace("M", "0.5").Replace("F", "1.0"), formatProvider)).ToArray();

                output[irCounter] = new double[lstOutPutClasses.Count];

                var vrCurrentOutClass = Convert.ToDouble(vrPerLine.Split(',').Last(), formatProvider);

                output[irCounter][lstOutPutClasses.IndexOf(vrCurrentOutClass)] = 1;

                irCounter++;
            }

            ActivationNetwork savedModel = (ActivationNetwork)ActivationNetwork.Load("iris_model.txt");

            List<double> lstGuessedClasses = new List<double>();

            var vrAccuracy = calculateAcurracy(savedModel, input, output, lstOutPutClasses, lstGuessedClasses);

            StringBuilder srBuild = new StringBuilder();
            srBuild.AppendLine("accuracy of unseen data testing: " + vrAccuracy.ToString("N2").ToString());

            int irIndexCounter = 0;
            foreach (var vrGuess in lstGuessedClasses)
            {
                srBuild.AppendLine($"test data index: {irIndexCounter}\t\tReal Class: {lstCorrectOutPutclass[irIndexCounter]}\t\tPredicted Class: {vrGuess}");
                irIndexCounter++;
            }

            File.WriteAllText("unseen_data_test_results.txt", srBuild.ToString());
        }

        private void btnBuildModelNFoldCrossValidation(object sender, RoutedEventArgs e)
        {
            Directory.CreateDirectory("nfold_training");
            Directory.CreateDirectory("nfold_testing");

            int irNNumber = 10;

            List<string> lstAllData = File.ReadAllLines("iris.data").ToList();

            int irChunkSize = lstAllData.Count / irNNumber;

            List<string[]> chunks = new List<string[]>();
            for (int i = 0; i < lstAllData.Count; i += irChunkSize)
            {
                chunks.Add(lstAllData.Skip(i).Take(irChunkSize).ToArray());
            }

            List<Tuple<string, string>> lstTestingDataFilePaths = new List<Tuple<string, string>>();


            for (int i = 0; i < chunks.Count; i++)
            {
                List<string> lstTraining = new List<string>();
                List<string> lstTesting = new List<string>();

                for (int mm = 0; mm < chunks.Count; mm++)
                {
                    if (i == mm)
                    {
                        lstTesting.AddRange(chunks[mm]);
                    }
                    else
                    {
                        lstTraining.AddRange(chunks[mm]);
                    }
                }

                string srTrainingFilePath = "nfold_training/train_" + i + ".txt";
                string srTestFilePath = "nfold_testing/test_" + i + ".txt";

                File.WriteAllLines(srTrainingFilePath, lstTraining);
                File.WriteAllLines(srTestFilePath, lstTesting);

                lstTestingDataFilePaths.Add(new Tuple<string, string>(srTrainingFilePath, srTestFilePath));
            }

            List<Task> lstTasks = new List<Task>();

            List<csMultiTraining> lstNFoldCrossValidationResults = new List<csMultiTraining>();

            Task.Factory.StartNew(() =>
            {

                foreach (var vrKeys in lstTestingDataFilePaths)
                {
                    var vrTask = Task.Factory.StartNew(() =>
                    {
                        var vrTrainingPath = vrKeys.Item1;
                        var vrTestingPath = vrKeys.Item1;

                        var vrReturnedClass = buildTrainigModel(vrTrainingPath, vrTestingPath, 10000);

                        lock (lstNFoldCrossValidationResults)
                        {
                            lstNFoldCrossValidationResults.Add(vrReturnedClass);
                        }

                    });

                    lstTasks.Add(vrTask);
                }

                Task.WaitAll(lstTasks.ToArray());

                lstNFoldCrossValidationResults = lstNFoldCrossValidationResults.OrderByDescending(pr => pr.dblAcc).ToList();

                saveModelProperties(lstNFoldCrossValidationResults.First().dblAcc, lstNFoldCrossValidationResults.First().irBestAccIteration, irNNumber+"FoldModel_best.txt");

                StringBuilder srTempBuild = new StringBuilder();

                srTempBuild.AppendLine($"{irNNumber}-fold cross validation average accuracy is: {(lstNFoldCrossValidationResults.Sum(pr => pr.dblAcc)/lstNFoldCrossValidationResults.Count).ToString("N2")}%");

                foreach (var item in lstNFoldCrossValidationResults)
                {
                    srTempBuild.AppendLine($"File: {item.srFileName}\t\tAcc: {item.dblAcc.ToString("N2")}\t\titeration: {item.irBestAccIteration.ToString("N0")}");
                }

                File.WriteAllText(irNNumber + "FoldModel_all.txt", srTempBuild.ToString());
            });

     

        }
    }
}
