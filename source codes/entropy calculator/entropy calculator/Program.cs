using System;
using System.Collections.Generic;
using System.Linq;

namespace entropy_calculator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("please enter inputs with pairs : and  differents ; seperated");
            var vrInput = Console.ReadLine();
            List<Tuple<double, double>> lstValues = new List<Tuple<double, double>>();

            foreach (var vrPerCouple in vrInput.Split(';'))
            {
                string srFirstParam = vrPerCouple.Split(':').First();
                string srSecondParam = vrPerCouple.Split(':').Last();
                double dblFirstParam = Convert.ToDouble(srFirstParam);
                double dblSecondParam = Convert.ToDouble(srSecondParam);
                lstValues.Add(new Tuple<double, double>(dblFirstParam, dblSecondParam));
            }

            double dblFinalEntropyOfAll = 0;

            foreach (var vrPerValPair in lstValues)
            {
                double dblCurrentPairProbability = vrPerValPair.Item1 + vrPerValPair.Item2;
                dblCurrentPairProbability = dblCurrentPairProbability / (lstValues.Select(pr => pr.Item1 + pr.Item2).Sum());
                dblFinalEntropyOfAll += dblCurrentPairProbability * calculateEntropy(vrPerValPair);
                Console.WriteLine("current final entropy of all value pairs: " + dblFinalEntropyOfAll.ToString("N3"));
            }

        }

        private static double calculateEntropy(Tuple<double, double> tupVars)
        {

            double dblFinalEntropy = 0;

            List<double> lstNumbers = new List<double> { tupVars.Item1, tupVars.Item2 };

            foreach (var vrCurrentNumber in lstNumbers)
            {
                double dblCurrentNumberProbability = vrCurrentNumber / lstNumbers.Sum();
                double dblCurrentNumberEntropy = -1 *
                    dblCurrentNumberProbability *
                    Math.Log2(dblCurrentNumberProbability);

                if (dblCurrentNumberProbability==0)
                    dblCurrentNumberEntropy = 0;

                Console.WriteLine($"current number = {vrCurrentNumber}");
                Console.WriteLine($"all numbers in this list = {string.Join(" , ", lstNumbers)}");
                Console.WriteLine($"entropy calculation = -1 * probability : {dblCurrentNumberProbability.ToString("N2")} *  Math.Log2(proability) = {Math.Log2(dblCurrentNumberProbability).ToString("N2")} = {dblCurrentNumberEntropy.ToString("N2")}" );
                dblFinalEntropy += dblCurrentNumberEntropy;
                Console.WriteLine("current final entropy = " + dblFinalEntropy.ToString("N2"));
            }


            return dblFinalEntropy;


        }
    }
}
