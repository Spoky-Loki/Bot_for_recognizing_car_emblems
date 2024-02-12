using System;
using System.Drawing;
using Telegram.Bot.Requests;
using Telegram.Bot.Types;

namespace NeuralNetwork1
{
    /// <summary>
    /// Тип фигуры
    /// </summary>
    public enum FigureType : byte { Suzuki = 0, Mercedes, Infinity, Citroen, Opel, Hyundai, Audi, Mitsubishi, Undef };

    public class GenerateImage
    {       
        /// <summary>
        /// Текущая сгенерированная фигура
        /// </summary>
        public FigureType current_figure = FigureType.Undef;

        /// <summary>
        /// Количество классов генерируемых фигур (8 - максимум)
        /// </summary>
        public int FigureCount { get; set; } = 8;

        /// <summary>
        /// Класс, реализующий логику обработки картинок
        /// </summary>
        private static Controller controller = null;

        private SamplesSet learnImages;
        private SamplesSet testImages;

        public GenerateImage()
        {
            learnImages = new SamplesSet();
            testImages = new SamplesSet();
            LoadImages();
            controller = new Controller();
            controller.settings.border = 80;
            controller.settings.threshold = 70;
            controller.settings.differenceLim = (float)controller.settings.threshold / 255;
        }

        public SamplesSet LearnImages
        {
            get { return learnImages; }
        }

        public SamplesSet TestImages
        {
            get { return testImages; }
        }

        public static Sample GenerateFigure(Bitmap image)
        {
            controller.ProcessImage(image);

            var img = controller.GetProcessedImage();
            img.Save("1.png");

            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(28, 28);
            AForge.Imaging.UnmanagedImage unmanaged = scaleFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(img));
            img = unmanaged.ToManagedImage();

            double[] input = getBitMap2(img);

            return new Sample(input, 8, FigureType.Undef);
        }

        private void LoadImages()
        {
            loadLearnFig(FigureType.Suzuki);
            loadTestFig(FigureType.Suzuki);

            loadLearnFig(FigureType.Mercedes);
            loadTestFig(FigureType.Mercedes);

            loadLearnFig(FigureType.Infinity);
            loadTestFig(FigureType.Infinity);

            loadLearnFig(FigureType.Citroen);
            loadTestFig(FigureType.Citroen);

            loadLearnFig(FigureType.Opel);
            loadTestFig(FigureType.Opel);

            loadLearnFig(FigureType.Hyundai);
            loadTestFig(FigureType.Hyundai);

            loadLearnFig(FigureType.Audi);
            loadTestFig(FigureType.Audi);

            loadLearnFig(FigureType.Mitsubishi);
            loadTestFig(FigureType.Mitsubishi);
        }

        private void loadLearnFig(FigureType figure)
        {
            for (int k = 21; k < 180; k++)
            {
                Image image;
                try
                { image = Image.FromFile("../../Images/" + figure.ToString() + "/image_" + k + ".png"); }
                catch
                { continue; }
                var img = new Bitmap(image);

                AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(28, 28);
                AForge.Imaging.UnmanagedImage unmanaged = scaleFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(img));
                img = unmanaged.ToManagedImage();

                var input = getBitMap2(img);
                Sample fig = new Sample(input, FigureCount, figure);
                learnImages.AddSample(fig);
            }
        }

        private void loadTestFig(FigureType figure)
        {
            for (int k = 0; k < 21; k++)
            {
                Image image;
                try
                { image = Image.FromFile("../../Images/" + figure.ToString() + "/image_" + k + ".png"); }
                catch
                { continue; }
                var img = new Bitmap(image);

                AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(28, 28);
                AForge.Imaging.UnmanagedImage unmanaged = scaleFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(img));
                img = unmanaged.ToManagedImage();

                var input = getBitMap2(img);
                Sample fig = new Sample(input, FigureCount, figure);
                testImages.AddSample(fig);
            }
        }

        private static double[] getBitMap(Bitmap img)
        {
            double[] input = new double[784];
            for (int i = 0; i < 784; i++)
                input[i] = 0;

            int k = 0;
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                {
                    if (img.GetPixel(i, j).R == 0 && img.GetPixel(i, j).G == 0 && img.GetPixel(i, j).B == 0)
                    {
                        input[k] = 1;
                    }
                    k++;
                }
            return input;
        }

        private static double[] getBitMap2(Bitmap img)
        {
            double[] input = new double[56];
            for (int i = 0; i < 56; i++)
                input[i] = 0;

            Color prev = Color.White;
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                {
                    if (img.GetPixel(i, j).R == 0 && img.GetPixel(i, j).G == 0 && img.GetPixel(i, j).B == 0)
                    {
                        if (prev.R == 255 && prev.G == 255 && prev.B == 255)
                        {
                            prev = Color.Black;
                            input[i] += 1;
                            input[28 + j] += 1;
                        }
                    }
                    else if (img.GetPixel(i, j).R == 255 && img.GetPixel(i, j).G == 255 && img.GetPixel(i, j).B == 255)
                    {
                        if (prev.R == 0 && prev.G == 0 && prev.B == 0)
                        {
                            prev = Color.White;
                            input[i] += 1;
                            input[28 + j] += 1;
                        }
                    }
                }
            return input;
        }
    }
}
