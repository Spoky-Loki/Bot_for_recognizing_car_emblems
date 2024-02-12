using Accord.MachineLearning.Boosting;
using Accord.Math;
using Accord.Neuro;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private double LearningRate = 0.3;

        public Stopwatch stopWatch = new Stopwatch();

        /// <summary>
        /// Значение ошибки при обучении единичному образцу. При обучении на наборе образов не используется
        /// </summary>
        public double desiredErrorValue = 0.0005;

        private double[][] WieghtLayer1;
        private double[][] WieghtLayer2;
        private double[][] WieghtOutputLayer;

        private double[] OutputLayer1;
        private double[] OutputLayer2;
        private double[] NetOutput;

        private double[] Computelayer1;
        private double[] Computelayer2;
        private double[] ComputeOutput;

        int countLayers;

        public StudentNetwork(int[] structure)
        {
            ReInit(structure);
        }

        /// <summary>
        /// Реиницализация сети - создаём заново сеть с указанной структурой
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        /// <param name="initialLearningRate">Начальная скорость обучения</param>
        public override void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            LearningRate = initialLearningRate;
            countLayers = structure.Length;
            if (structure.Length == 4)
            {
                OutputLayer1 = new double[structure[1]];
                OutputLayer2 = new double[structure[2]];
                NetOutput = new double[structure[3]];

                Computelayer1 = new double[structure[1]];
                Computelayer2 = new double[structure[2]];
                ComputeOutput = new double[structure[3]];

                WieghtLayer1 = new double[structure[0]][];
                WieghtLayer2 = new double[structure[1]][];
                WieghtOutputLayer = new double[structure[2]][];
                for (int i = 0; i < structure[0]; i++)
                    WieghtLayer1[i] = new double[structure[1]];
                for (int i = 0; i < structure[1]; i++)
                    WieghtLayer2[i] = new double[structure[2]];
                for (int i = 0; i < structure[2]; i++)
                    WieghtOutputLayer[i] = new double[structure[3]];
            }
            else if (structure.Length == 3)
            {
                OutputLayer1 = new double[structure[1]];
                NetOutput = new double[structure[2]];

                Computelayer1 = new double[structure[1]];
                ComputeOutput = new double[structure[2]];

                WieghtLayer1 = new double[structure[0]][];
                WieghtOutputLayer = new double[structure[1]][];
                for (int i = 0; i < structure[0]; i++)
                    WieghtLayer1[i] = new double[structure[1]];
                for (int i = 0; i < structure[1]; i++)
                    WieghtOutputLayer[i] = new double[structure[2]];
            }
            RandomizeWieght();
        }

        public override int Train(Sample sample, bool parallel)
        {
            int iters = 1;
            while (Run(sample.input, sample.output) > desiredErrorValue
                && iters <= 100)
            {
                ++iters;
            }

            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].output;
            }

            double error = double.PositiveInfinity;
            int epoch_to_run = 0;

            stopWatch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                epoch_to_run++;

                error = RunEpoch(inputs, outputs);

                updateDelegate((epoch_to_run * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

            updateDelegate(1.0, error, stopWatch.Elapsed);
            stopWatch.Stop();

            return error;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            double correct = 0.0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                testSet[i].output = Compute(testSet[i].input);
                testSet[i].processOutput();
                if (testSet[i].actualClass == testSet[i].recognizedClass) correct += 1;
            }
            return correct / testSet.Count;
        }

        public override FigureType Predict(Sample sample)
        {
            sample.output = Compute(sample.input);
            sample.processOutput();
            return sample.recognizedClass;
        }

        public override double[] getOutput()
        {
            return NetOutput;
        }

        private double[] Compute(double[] input)
        {
            if (countLayers == 4)
            {
                Computelayer1 = ComputeLayer(input, WieghtLayer1);
                for (int i = 0; i < Computelayer1.Length; i++)
                    OutputLayer1[i] = SigmoidFunction(Computelayer1[i]);

                Computelayer2 = ComputeLayer(OutputLayer1, WieghtLayer2);
                for (int i = 0; i < Computelayer2.Length; i++)
                    OutputLayer2[i] = SigmoidFunction(Computelayer2[i]);

                ComputeOutput = ComputeLayer(OutputLayer2, WieghtOutputLayer);
                for (int i = 0; i < ComputeOutput.Length; i++)
                    NetOutput[i] = SigmoidFunction(ComputeOutput[i]);
            }
            else if (countLayers == 3)
            {
                Computelayer1 = ComputeLayer(input, WieghtLayer1);
                for (int i = 0; i < Computelayer1.Length; i++)
                    OutputLayer1[i] = SigmoidFunction(Computelayer1[i]);

                ComputeOutput = ComputeLayer(OutputLayer1, WieghtOutputLayer);
                for (int i = 0; i < ComputeOutput.Length; i++)
                    NetOutput[i] = SigmoidFunction(ComputeOutput[i]);
            }

            return NetOutput;
        }

        private double RunEpoch(double[][] inputs, double[][] outputs)
        {
            double result = 0.0;

            for (int i = 0; i < inputs.Rows(); i++)
                result += Run(inputs[i], outputs[i]);

            return result;
        }

        private double Run(double[] inputs, double[] outputs)
        {
            double error = 0.0;
            Compute(inputs);

            if (countLayers == 4)
            {
                var ouputError = OutputError(ref outputs, ref error);
                var layer2Error = LayerError(ouputError, Computelayer2, WieghtOutputLayer);
                var layer1Error = LayerError(layer2Error, Computelayer1, WieghtLayer2);

                UpdateWiehgts(ref ouputError, ref OutputLayer2, ref WieghtOutputLayer);
                UpdateWiehgts(ref layer2Error, ref OutputLayer1, ref WieghtLayer2);
                UpdateWiehgts(ref layer1Error, ref inputs, ref WieghtLayer1);
            }
            else if (countLayers == 3)
            {
                var ouputError = OutputError(ref outputs, ref error);
                var layer1Error = LayerError(ouputError, Computelayer1, WieghtOutputLayer);

                UpdateWiehgts(ref ouputError, ref OutputLayer1, ref WieghtOutputLayer);
                UpdateWiehgts(ref layer1Error, ref inputs, ref WieghtLayer1);
            }

            return error;
        }

        private double[] OutputError(ref double[] correctOutput, ref double error)
        {
            double[] result = new double[NetOutput.Length];

            for (int i = 0; i < NetOutput.Length; i++)
            {
                var er = (correctOutput[i] - NetOutput[i]);
                result[i] = er * DerivativeSigmoidFunction(ComputeOutput[i]);
                error += er * er;
            }

            error *= 0.5;
            return result;
        }

        private double[] LayerError(double[] prevLayerError, 
            double[] computeLayer, double[][] wieght)
        {
            double[] result = new double[computeLayer.Length];

            Parallel.For(0, result.Length, i =>
            {
                for (int j = 0; j < wieght.Columns(); j++)
                    result[i] += prevLayerError[j] * wieght[i][j];
                result[i] *= DerivativeSigmoidFunction(computeLayer[i]);
            });

            return result;
        }

        private void UpdateWiehgts(ref double[] Error, ref double[] OutputPrevLayer, ref double[][] Wieghts)
        {
            for (int i = 0; i < Wieghts.Rows(); i++)
            {
                for (int j = 0; j < Wieghts.Columns(); j++)
                {
                    Wieghts[i][j] += LearningRate * Error[j] * OutputPrevLayer[i];
                }
            }
        }

        private double[] ComputeLayer(double[] input, double[][] layerWieght)
        {
            double[] result = new double[layerWieght.Columns()];

            for (int i = 0; i < layerWieght.Columns(); i++)
                for (int j = 0; j < layerWieght.Rows(); j++)
                    result[i] += input[j] * layerWieght[j][i];

            return result;
        }

        private double SigmoidFunction(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double DerivativeSigmoidFunction(double x) 
        {
            var f = SigmoidFunction(x);

            return f * (1 - f);
        }

        private void RandomizeWieght()
        {
            if (countLayers == 4)
            {
                Random rand = new Random();
                for (int i = 0; i < WieghtLayer1.Rows(); i++)
                    for (int j = 0; j < WieghtLayer1.Columns(); j++)
                        WieghtLayer1[i][j] = Math.Round(rand.NextDouble() - 0.5, 3);

                for (int i = 0; i < WieghtLayer2.Rows(); i++)
                    for (int j = 0; j < WieghtLayer2.Columns(); j++)
                        WieghtLayer2[i][j] = Math.Round(rand.NextDouble() - 0.5, 3);

                for (int i = 0; i < WieghtOutputLayer.Rows(); i++)
                    for (int j = 0; j < WieghtOutputLayer.Columns(); j++)
                        WieghtOutputLayer[i][j] = Math.Round(rand.NextDouble() - 0.5, 3);
            }
            else if (countLayers == 3)
            {
                Random rand = new Random();
                for (int i = 0; i < WieghtLayer1.Rows(); i++)
                    for (int j = 0; j < WieghtLayer1.Columns(); j++)
                        WieghtLayer1[i][j] = Math.Round(rand.NextDouble() - 0.5, 3);

                for (int i = 0; i < WieghtOutputLayer.Rows(); i++)
                    for (int j = 0; j < WieghtOutputLayer.Columns(); j++)
                        WieghtOutputLayer[i][j] = Math.Round(rand.NextDouble() - 0.5, 3);
            }
        }
    }
}