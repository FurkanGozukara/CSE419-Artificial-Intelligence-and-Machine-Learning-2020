using System;

namespace lecture_4_knn_example_program
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Begin weighted k-NN demo ");
            Console.WriteLine("Normalized age-expenses data: ");
            Console.WriteLine("[id =  0, 0.25, 0.43, class = 0]");
            Console.WriteLine(" . . . ");
            Console.WriteLine("[id = 32, 0.46, 0.22, class = 2]");
            double[][] data = new double[33][];
            data[0] = new double[] { 0, 0.25, 0.43, 0 };
            data[1] = new double[] { 1, 0.26, 0.54, 0 };
            data[2] = new double[] { 2, 0.27, 0.60, 0 };
            data[3] = new double[] { 3, 0.37, 0.31, 0 };
            data[4] = new double[] { 4, 0.38, 0.70, 0 };
            data[5] = new double[] { 5, 0.49, 0.32, 0 };
            data[6] = new double[] { 6, 0.46, 0.70, 0 };
            data[7] = new double[] { 7, 0.55, 0.32, 0 };
            data[8] = new double[] { 8, 0.58, 0.74, 0 };
            data[9] = new double[] { 9, 0.67, 0.42, 0 };
            data[10] = new double[] { 10, 0.69, 0.51, 0 };
            data[11] = new double[] { 11, 0.66, 0.63, 0 };
            data[12] = new double[] { 12, 0.29, 0.43, 1 };
            data[13] = new double[] { 13, 0.35, 0.51, 1 };
            data[14] = new double[] { 14, 0.39, 0.63, 1 };
            data[15] = new double[] { 15, 0.47, 0.40, 1 };
            data[16] = new double[] { 16, 0.48, 0.50, 1 };
            data[17] = new double[] { 17, 0.45, 0.61, 1 };
            data[18] = new double[] { 18, 0.55, 0.41, 1 };
            data[19] = new double[] { 19, 0.57, 0.53, 1 };
            data[20] = new double[] { 20, 0.56, 0.62, 1 };
            data[21] = new double[] { 21, 0.68, 0.12, 1 };
            data[22] = new double[] { 22, 0.78, 0.24, 1 };
            data[23] = new double[] { 23, 0.86, 0.30, 1 };
            data[24] = new double[] { 24, 0.86, 0.22, 1 };
            data[25] = new double[] { 25, 0.86, 0.12, 1 };
            data[26] = new double[] { 26, 0.78, 0.14, 1 };
            data[27] = new double[] { 27, 0.28, 0.13, 2 };
            data[28] = new double[] { 28, 0.25, 0.21, 2 };
            data[29] = new double[] { 29, 0.39, 0.14, 2 };
            data[30] = new double[] { 30, 0.37, 0.24, 2 };
            data[31] = new double[] { 31, 0.45, 0.13, 2 };
            data[32] = new double[] { 32, 0.46, 0.22, 2 };
            double[] item = new double[] { 0.35, 0.38 };
            Console.WriteLine("Nearest (k=6) to (0.35, 0.38):");
            Analyze(item, data, 6, 3);  // 3 classes
            Console.WriteLine("End weighted k-NN demo ");
            Console.ReadLine();
        }

        static void Analyze(double[] item, double[][] data,
      int k, int c)
        {
            // 1. Compute all distances
            int N = data.Length;
            double[] distances = new double[N];
            for (int i = 0; i < N; ++i)
                distances[i] = DistFunc(item, data[i]);
            // 2. Get ordering
            int[] ordering = new int[N];
            for (int i = 0; i < N; ++i)
                ordering[i] = i;
            double[] distancesCopy = new double[N];
            Array.Copy(distances, distancesCopy, distances.Length);
            Array.Sort(distancesCopy, ordering);
            // 3. Show info for k nearest
            double[] kNearestDists = new double[k];
            for (int i = 0; i < k; ++i)
            {
                int idx = ordering[i];
                Show(data[idx]);
                Console.WriteLine("  distance = " +
                  distances[idx].ToString("F4"));
                kNearestDists[i] = distances[idx];
            }
            // 4. Vote
            double[] votes = new double[c];  // one per class
            double[] wts = MakeWeights(k, kNearestDists);
            Console.WriteLine("Weights (inverse technique): ");
            for (int i = 0; i < wts.Length; ++i)
                Console.Write(wts[i].ToString("F4") + "  ");
            Console.WriteLine("\n\nPredicted class: ");
            for (int i = 0; i < k; ++i)
            {
                int idx = ordering[i];
                int predClass = (int)data[idx][3];
                votes[predClass] += wts[i] * 1.0;
            }
            for (int i = 0; i < c; ++i)
                Console.WriteLine("[" + i + "]  " +
                votes[i].ToString("F4"));
        } // Analyze

        static double[] MakeWeights(int k, double[] distances)
        {
            // Inverse technique
            double[] result = new double[k];  // one perneighbor
            double sum = 0.0;
            for (int i = 0; i < k; ++i)
            {
                result[i] = 1.0 / distances[i];
                sum += result[i];
            }
            for (int i = 0; i < k; ++i)
                result[i] /= sum;
            return result;
        }
        static double DistFunc(double[] item, double[] dataPoint)
        {
            double sum = 0.0;
            for (int i = 0; i < 2; ++i)
            {
                double diff = item[i] - dataPoint[i + 1];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
        static void Show(double[] v)
        {
            Console.Write("idx = " + v[0].ToString().PadLeft(3) +
              "  (" + v[1].ToString("F2") + " " +
              v[2].ToString("F2") + ") class = " + v[3]);
        }
    }
}
