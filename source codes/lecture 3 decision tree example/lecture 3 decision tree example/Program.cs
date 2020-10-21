using Accord;
using Accord.DataSets;
using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Statistics.Filters;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace lecture_3_decision_tree_example
{
    class Program
    {
        static void Main(string[] args)
        {
            simpleDecisionTreeExample();
            realID3Example();

            initDecisionTreeModel();

            //infiniteLoop();

            breastCancerExample();
         //   breastCancerExampleID3();
        }

        private static void breastCancerExampleID3()
        {
            // Ensure we have reproducible results
            Accord.Math.Random.Generator.Seed = 0;

            // Get some data to be learned. We will be using the Wiconsin's
            // (Diagnostic) Breast Cancer dataset, where the goal is to determine
            // whether the characteristics extracted from a breast cancer exam
            // correspond to a malignant or benign type of cancer:
            var data = new WisconsinDiagnosticBreastCancer();
            double[][] input_temp = data.Features; // 569 samples, 30-dimensional features
            int[][] input = new int [input_temp.GetLength(0) ][] ; // 569 samples, 30-dimensional features

            for (int i = 0; i < input_temp.GetLength(0); i++)
            {
                input[i] = new int[input_temp[i].GetLength(0)];
                for (int nn = 0; nn < input_temp[i].GetLength(0); nn++)
                {
                    input_temp[i][nn] = input_temp[i][nn] * 10000.0;
                    input[i][nn] = Convert.ToInt32(input_temp[i][nn]);
                }
            }

            int[] output = data.ClassLabels;  // 569 samples, 2 different class labels

            // Let's say we want to measure the cross-validation performance of
            // a decision tree with a maximum tree height of 5 and where variables
            // are able to join the decision path at most 2 times during evaluation:
            var cv = CrossValidation.Create(

                k: 2, // We will be using 10-fold cross validation

                learner: (p) => new ID3Learning() // here we create the learning algorithm
                {
                    Join = 2,
                    MaxHeight = 5
                },

                // Now we have to specify how the tree performance should be measured:
                loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),

                // This function can be used to perform any special
                // operations before the actual learning is done, but
                // here we will just leave it as simple as it can be:
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),

                // Finally, we have to pass the input and output data
                // that will be used in cross-validation. 
                x: input, y: output
            );

            // After the cross-validation object has been created,
            // we can call its .Learn method with the input and 
            // output data that will be partitioned into the folds:
            var result = cv.Learn(input, output);

            // We can grab some information about the problem:
            int numberOfSamples = result.NumberOfSamples; // should be 569
            int numberOfInputs = result.NumberOfInputs;   // should be 30
            int numberOfOutputs = result.NumberOfOutputs; // should be 2

            double trainingError = result.Training.Mean; // should be 0.017771153143274855
            double validationError = result.Validation.Mean; // should be 0.0755952380952381

            // If desired, compute an aggregate confusion matrix for the validation sets:
            GeneralConfusionMatrix gcm = result.ToConfusionMatrix(input, output);
            double accuracy = gcm.Accuracy; // result should be 0.92442882249560632
            
            Console.WriteLine("id3 learning algorithm accuracy is %" + (accuracy*100).ToString("N2"));
        }

        private static void breastCancerExample()
        {
            // Ensure we have reproducible results
            Accord.Math.Random.Generator.Seed = 0;

            // Get some data to be learned. We will be using the Wiconsin's
            // (Diagnostic) Breast Cancer dataset, where the goal is to determine
            // whether the characteristics extracted from a breast cancer exam
            // correspond to a malignant or benign type of cancer:
            var data = new WisconsinDiagnosticBreastCancer();
            double[][] input = data.Features; // 569 samples, 30-dimensional features
            int[] output = data.ClassLabels;  // 569 samples, 2 different class labels

            // Let's say we want to measure the cross-validation performance of
            // a decision tree with a maximum tree height of 5 and where variables
            // are able to join the decision path at most 2 times during evaluation:
            var cv = CrossValidation.Create(

                k: 10, // We will be using 10-fold cross validation

                learner: (p) => new C45Learning() // here we create the learning algorithm
                {
                    Join = 2,
                    MaxHeight = 5
                },

                // Now we have to specify how the tree performance should be measured:
                loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),

                // This function can be used to perform any special
                // operations before the actual learning is done, but
                // here we will just leave it as simple as it can be:
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),

                // Finally, we have to pass the input and output data
                // that will be used in cross-validation. 
                x: input, y: output
            );

            // After the cross-validation object has been created,
            // we can call its .Learn method with the input and 
            // output data that will be partitioned into the folds:
            var result = cv.Learn(input, output);

            // We can grab some information about the problem:
            int numberOfSamples = result.NumberOfSamples; // should be 569
            int numberOfInputs = result.NumberOfInputs;   // should be 30
            int numberOfOutputs = result.NumberOfOutputs; // should be 2

            double trainingError = result.Training.Mean; // should be 0.017771153143274855
            double validationError = result.Validation.Mean; // should be 0.0755952380952381

            // If desired, compute an aggregate confusion matrix for the validation sets:
            GeneralConfusionMatrix gcm = result.ToConfusionMatrix(input, output);
            double accuracy = gcm.Accuracy; // result should be 0.92442882249560632

            Console.WriteLine("C45Learning learning algorithm accuracy is %" + (accuracy * 100).ToString("N2"));
        }

        private static void infiniteLoop()
        {
            while (true)
            {
                Dictionary<string, string> dicValues = new Dictionary<string, string>();

                foreach (var vrPerFeature in lst_input_features)
                {
                    Console.WriteLine("please enter value for " + vrPerFeature);
                    var vrReadVal = Console.ReadLine();
                    dicValues.Add(vrPerFeature, vrReadVal);
                }

                var vrPredictedResult = predict_the_class(dicValues);
                Console.WriteLine("\r\npredicted result: " + vrPredictedResult + "\r\n");
                Console.WriteLine("press a key to continue");
                Console.ReadKey();
            }
        }

        private static List<string> lst_input_features = new List<string>{"Outlook",
            "Temperature", "Humidity", "Wind" };

        private static string predict_the_class(Dictionary<string, string> dicInput)
        {
            string[,] arrayString = new string[dicInput.Count, 2];
            int irIndex = 0;
            foreach (var vrPerRecord in dicInput)
            {
                arrayString[irIndex, 0] = vrPerRecord.Key;
                arrayString[irIndex, 1] = vrPerRecord.Value;
                irIndex++;
            }

            int[] query = myCodeBook.Transform(arrayString);

            // And then predict the label using
            int predicted = myTreeModel.Decide(query);  // result will be 0

            // We can translate it back to strings using
            string answer = myCodeBook.Revert("PlayTennis", predicted); // Answer will be: "No"
            return answer;
        }

        static DataTable dtStatic = new DataTable("my custom data table");
        static Codification myCodeBook;
        static DecisionTree myTreeModel;

        private static void initDecisionTreeModel()
        {
            dtStatic.Columns.Add("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis");
            dtStatic.Rows.Add("D1", "Sunny", "Hot", "High", "Weak", "No");
            dtStatic.Rows.Add("D2", "Sunny", "Hot", "High", "Strong", "No");
            dtStatic.Rows.Add("D3", "Overcast", "Hot", "High", "Weak", "Yes");
            dtStatic.Rows.Add("D4", "Rain", "Mild", "High", "Weak", "Yes");
            dtStatic.Rows.Add("D5", "Rain", "Cool", "Normal", "Weak", "Yes");
            dtStatic.Rows.Add("D6", "Rain", "Cool", "Normal", "Strong", "No");
            dtStatic.Rows.Add("D7", "Overcast", "Cool", "Normal", "Strong", "Yes");
            dtStatic.Rows.Add("D8", "Sunny", "Mild", "High", "Weak", "No");
            dtStatic.Rows.Add("D9", "Sunny", "Cool", "Normal", "Weak", "Yes");
            dtStatic.Rows.Add("D10", "Rain", "Mild", "Normal", "Weak", "Yes");
            dtStatic.Rows.Add("D11", "Sunny", "Mild", "Normal", "Strong", "Yes");
            dtStatic.Rows.Add("D12", "Overcast", "Mild", "High", "Strong", "Yes");
            dtStatic.Rows.Add("D13", "Overcast", "Hot", "Normal", "Weak", "Yes");
            dtStatic.Rows.Add("D14", "Rain", "Mild", "High", "Strong", "No");
            dtStatic.Rows.Add("D15", "Rain", "Cool", "High", "Strong", "No");
            dtStatic.Rows.Add("D16", "Rain", "Hot", "High", "Strong", "Yes");
            dtStatic.Rows.Add("D17", "Rain", "Hot", "High", "Weak", "Yes");
            dtStatic.Rows.Add("D18", "Rain", "Cool", "High", "Weak", "No");
            dtStatic.Rows.Add("D19", "Rain", "Cool", "High", "Weak", "Yes");
            dtStatic.Rows.Add("D20", "Rain", "Mild", "High", "Strong", "Yes");

            myCodeBook = new Codification(dtStatic);

            DataTable symbols = myCodeBook.Apply(dtStatic);
            int[][] inputs = symbols.ToJagged<int>("Outlook", "Temperature", "Humidity", "Wind");
            int[] outputs = symbols.ToArray<int>("PlayTennis");
            var id3learning = new ID3Learning(){
    new DecisionVariable("Outlook",     3), // 3 possible values (Sunny, overcast, rain)
    new DecisionVariable("Temperature", 3), // 3 possible values (Hot, mild, cool)  
    new DecisionVariable("Humidity",    2), // 2 possible values (High, normal)    
    new DecisionVariable("Wind",        2)  // 2 possible values (Weak, strong) 
};
            myTreeModel = id3learning.Learn(inputs, outputs);

            double error = new ZeroOneLoss(outputs).Loss(myTreeModel.Decide(inputs));

            Console.WriteLine("learnt model training accuracy is: " + (100 - error).ToString("N2"));

        }

        private static void realID3Example()
        {
            // In this example, we will be using the famous Play Tennis example by Tom Mitchell (1998).
            // In Mitchell's example, one would like to infer if a person would play tennis or not
            // based solely on four input variables. Those variables are all categorical, meaning that
            // there is no order between the possible values for the variable (i.e. there is no order
            // relationship between Sunny and Rain, one is not bigger nor smaller than the other, but are 
            // just distinct). Moreover, the rows, or instances presented above represent days on which the
            // behavior of the person has been registered and annotated, pretty much building our set of 
            // observation instances for learning:

            // Note: this example uses DataTables to represent the input data , but this is not required.
            DataTable data = new DataTable("Mitchell's Tennis Example");

            data.Columns.Add("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis");
            data.Rows.Add("D1", "Sunny", "Hot", "High", "Weak", "No");
            data.Rows.Add("D2", "Sunny", "Hot", "High", "Strong", "No");
            data.Rows.Add("D3", "Overcast", "Hot", "High", "Weak", "Yes");
            data.Rows.Add("D4", "Rain", "Mild", "High", "Weak", "Yes");
            data.Rows.Add("D5", "Rain", "Cool", "Normal", "Weak", "Yes");
            data.Rows.Add("D6", "Rain", "Cool", "Normal", "Strong", "No");
            data.Rows.Add("D7", "Overcast", "Cool", "Normal", "Strong", "Yes");
            data.Rows.Add("D8", "Sunny", "Mild", "High", "Weak", "No");
            data.Rows.Add("D9", "Sunny", "Cool", "Normal", "Weak", "Yes");
            data.Rows.Add("D10", "Rain", "Mild", "Normal", "Weak", "Yes");
            data.Rows.Add("D11", "Sunny", "Mild", "Normal", "Strong", "Yes");
            data.Rows.Add("D12", "Overcast", "Mild", "High", "Strong", "Yes");
            data.Rows.Add("D13", "Overcast", "Hot", "Normal", "Weak", "Yes");
            data.Rows.Add("D14", "Rain", "Mild", "High", "Strong", "No");

            // In order to try to learn a decision tree, we will first convert this problem to a more simpler
            // representation. Since all variables are categories, it does not matter if they are represented
            // as strings, or numbers, since both are just symbols for the event they represent. Since numbers
            // are more easily representable than text string, we will convert the problem to use a discrete 
            // alphabet through the use of a Accord.Statistics.Filters.Codification codebook.</para>

            // A codebook effectively transforms any distinct possible value for a variable into an integer 
            // symbol. For example, “Sunny” could as well be represented by the integer label 0, “Overcast” 
            // by “1”, Rain by “2”, and the same goes by for the other variables. So:</para>

            // Create a new codification codebook to 
            // convert strings into integer symbols
            var codebook = new Codification(data);

            // Translate our training data into integer symbols using our codebook:
            DataTable symbols = codebook.Apply(data);
            int[][] inputs = symbols.ToJagged<int>("Outlook", "Temperature", "Humidity", "Wind");
            int[] outputs = symbols.ToArray<int>("PlayTennis");

            // For this task, in which we have only categorical variables, the simplest choice 
            // to induce a decision tree is to use the ID3 algorithm by Quinlan. Let’s do it:

            // Create a teacher ID3 algorithm
            var id3learning = new ID3Learning()
{
    // Now that we already have our learning input/ouput pairs, we should specify our
    // decision tree. We will be trying to build a tree to predict the last column, entitled
    // “PlayTennis”. For this, we will be using the “Outlook”, “Temperature”, “Humidity” and
    // “Wind” as predictors (variables which will we will use for our decision). Since those
    // are categorical, we must specify, at the moment of creation of our tree, the
    // characteristics of each of those variables. So:

    new DecisionVariable("Outlook",     3), // 3 possible values (Sunny, overcast, rain)
    new DecisionVariable("Temperature", 3), // 3 possible values (Hot, mild, cool)  
    new DecisionVariable("Humidity",    2), // 2 possible values (High, normal)    
    new DecisionVariable("Wind",        2)  // 2 possible values (Weak, strong) 

    // Note: It is also possible to create a DecisionVariable[] from a codebook:
    // DecisionVariable[] attributes = DecisionVariable.FromCodebook(codebook);
};

            // Learn the training instances!
            DecisionTree tree = id3learning.Learn(inputs, outputs);

            // Compute the training error when predicting training instances
            double error = new ZeroOneLoss(outputs).Loss(tree.Decide(inputs));

            // The tree can now be queried for new examples through 
            // its decide method. For example, we can create a query

            int[] query = codebook.Transform(new[,]
            {
                { "Outlook",     "Sunny"  },
                { "Temperature", "Hot"    },
                { "Humidity",    "High"   },
                { "Wind",        "Strong" }
            });

            // And then predict the label using
            int predicted = tree.Decide(query);  // result will be 0

            // We can translate it back to strings using
            string answer = codebook.Revert("PlayTennis", predicted); // Answer will be: "No"
        }

        private static void simpleDecisionTreeExample()
        {
            // In this example, we will learn a decision tree directly from integer
            // matrices that define the inputs and outputs of our learning problem.
            // XOR = https://en.wikipedia.org/wiki/Exclusive_or
            int[][] inputs =
                {
                new int[] { 0, 0 },
                new int[] { 0, 1 },
                new int[] { 1, 0 },
                new int[] { 1, 1 },
                };

            int[] outputs = // xor between inputs[0] and inputs[1]
            {
                0, 1, 1, 0
            };

            // Create an ID3 learning algorithm
            ID3Learning teacher = new ID3Learning();

            // Learn a decision tree for the XOR problem
            var tree = teacher.Learn(inputs, outputs);

            var vrTestResult = tree.Decide(inputs);

            // Compute the error in the learning
            double error = new ZeroOneLoss(outputs).Loss(tree.Decide(inputs));

            // The tree can now be queried for new examples:
            int[] predicted = tree.Decide(inputs); // should be { 0, 1, 1, 0 }
        }
    }
}
