﻿using System;
using System.Drawing;

namespace NeuralNetwork1
{
    internal class Settings
    {
        private int _border = 20;
        public int border
        {
            get
            {
                return _border;
            }
            set
            {
                if ((value > 0) && (value < height / 3))
                {
                    _border = value;
                    if (top > 2 * _border) top = 2 * _border;
                    if (left > 2 * _border) left = 2 * _border;
                }
            }
        }

        public int width = 640;
        public int height = 640;
        
        /// <summary>
        /// Размер сетки для сенсоров по горизонтали
        /// </summary>
        public int blocksCount = 10;

        /// <summary>
        /// Желаемый размер изображения до обработки
        /// </summary>
        public Size orignalDesiredSize = new Size(500, 500);
        /// <summary>
        /// Желаемый размер изображения после обработки
        /// </summary>
        public Size processedDesiredSize = new Size(200, 200);

        public int margin = 10;
        public int top = 40;
        public int left = 40;

        /// <summary>
        /// Второй этап обработки
        /// </summary>
        public bool processImg = false;

        /// <summary>
        /// Порог при отсечении по цвету 
        /// </summary>
        public byte threshold = 120;
        public float differenceLim = 0.15f;

        public void incTop() { if (top < 2 * _border) ++top; }
        public void decTop() { if (top > 0) --top; }
        public void incLeft() { if (left < 2 * _border) ++left; }
        public void decLeft() { if (left > 0) --left; }
    }

    internal class MagicEye
    {
        /// <summary>
        /// Обработанное изображение
        /// </summary>
        public Bitmap processed;
        /// <summary>
        /// Оригинальное изображение после обработки
        /// </summary>
        public Bitmap original;

        /// <summary>
        /// Класс настроек
        /// </summary>
        public Settings settings = new Settings();

        public MagicEye()
        {
        }

        public bool ProcessImage(Bitmap bitmap)
        {
            // На вход поступает необработанное изображение с веб-камеры

            //  Минимальная сторона изображения (обычно это высота)
            if (bitmap.Height > bitmap.Width)
            {
                int side1 = bitmap.Width;
                //  Мы сейчас занимаемся тем, что красиво оформляем входной кадр, чтобы вывести его на форму
                Rectangle cropRect1 = new Rectangle(0, settings.top + settings.border, side1, side1);

                //  Тут создаём новый битмапчик, который будет исходным изображением
                original = new Bitmap(cropRect1.Width, cropRect1.Height);

                //  Объект для рисования создаём
                Graphics g1 = Graphics.FromImage(original);

                g1.DrawImage(bitmap, new Rectangle(0, 0, original.Width, original.Height), cropRect1, GraphicsUnit.Pixel);
                Pen p1 = new Pen(Color.Red);
                p1.Width = 1;

                //  Теперь всю эту муть пилим в обработанное изображение
                AForge.Imaging.Filters.Grayscale grayFilter1 = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
                var uProcessed1 = grayFilter1.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));


                int blockWidth1 = original.Width / settings.blocksCount;
                int blockHeight1 = original.Height / settings.blocksCount;
                for (int r = 0; r < settings.blocksCount; ++r)
                    for (int c = 0; c < settings.blocksCount; ++c)
                    {
                        //  Тут ещё обработку сделать
                        g1.DrawRectangle(p1, new Rectangle(c * blockWidth1, r * blockHeight1, blockWidth1, blockHeight1));
                    }


                //  Масштабируем изображение до 500x500 - этого достаточно
                AForge.Imaging.Filters.ResizeBilinear scaleFilter1 = new AForge.Imaging.Filters.ResizeBilinear(settings.orignalDesiredSize.Width, settings.orignalDesiredSize.Height);
                uProcessed1 = scaleFilter1.Apply(uProcessed1);
                original = scaleFilter1.Apply(original);
                g1 = Graphics.FromImage(original);

                //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
                AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter1 = new AForge.Imaging.Filters.BradleyLocalThresholding();
                threshldFilter1.PixelBrightnessDifferenceLimit = settings.differenceLim;
                threshldFilter1.ApplyInPlace(uProcessed1);


                original.Save("213.png");

                processSample(ref uProcessed1);

                processed = uProcessed1.ToManagedImage();

                return true;
            }
            else
            {
                //  Можно было, конечено, и не кидаться эксепшенами в истерике, но идите и купите себе нормальную камеру!
                int side = bitmap.Height;

                //  Отпиливаем границы, но не более половины изображения
                if (side < 4 * settings.border) settings.border = side / 4;
                side -= 2 * settings.border;

                //  Мы сейчас занимаемся тем, что красиво оформляем входной кадр, чтобы вывести его на форму
                Rectangle cropRect = new Rectangle((bitmap.Width - bitmap.Height) / 2 + settings.left + settings.border, settings.top + settings.border, side, side);

                //  Тут создаём новый битмапчик, который будет исходным изображением
                original = new Bitmap(cropRect.Width, cropRect.Height);

                //  Объект для рисования создаём
                Graphics g = Graphics.FromImage(original);

                g.DrawImage(bitmap, new Rectangle(0, 0, original.Width, original.Height), cropRect, GraphicsUnit.Pixel);
                Pen p = new Pen(Color.Red);
                p.Width = 1;

                //  Теперь всю эту муть пилим в обработанное изображение
                AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
                var uProcessed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));


                int blockWidth = original.Width / settings.blocksCount;
                int blockHeight = original.Height / settings.blocksCount;
                for (int r = 0; r < settings.blocksCount; ++r)
                    for (int c = 0; c < settings.blocksCount; ++c)
                    {
                        //  Тут ещё обработку сделать
                        g.DrawRectangle(p, new Rectangle(c * blockWidth, r * blockHeight, blockWidth, blockHeight));
                    }


                //  Масштабируем изображение до 500x500 - этого достаточно
                AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(settings.orignalDesiredSize.Width, settings.orignalDesiredSize.Height);
                uProcessed = scaleFilter.Apply(uProcessed);
                original = scaleFilter.Apply(original);
                g = Graphics.FromImage(original);
                //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
                AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
                threshldFilter.PixelBrightnessDifferenceLimit = settings.differenceLim;
                threshldFilter.ApplyInPlace(uProcessed);

                original.Save("213.png");

                processSample(ref uProcessed);

                processed = uProcessed.ToManagedImage();

                return true;
            }
        }

        /// <summary>
        /// Обработка одного сэмпла
        /// </summary>
        /// <param name="index"></param>
        private string processSample(ref AForge.Imaging.UnmanagedImage unmanaged)
        {
            string rez = "Обработка";

            ///  Инвертируем изображение
            AForge.Imaging.Filters.Invert InvertFilter = new AForge.Imaging.Filters.Invert();
            InvertFilter.ApplyInPlace(unmanaged);

            ///    Создаём BlobCounter, выдёргиваем самый большой кусок, масштабируем, пересечение и сохраняем
            ///    изображение в эксклюзивном использовании
            AForge.Imaging.BlobCounterBase bc = new AForge.Imaging.BlobCounter();

            bc.FilterBlobs = true;
            bc.MinWidth = 3;
            bc.MinHeight = 3;
            // Упорядочиваем по размеру
            bc.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;
            // Обрабатываем картинку
            
            bc.ProcessImage(unmanaged);

            Rectangle[] rects = bc.GetObjectsRectangles();
            int lx = unmanaged.Width;
            int ly = unmanaged.Height;
            int rx = 0;
            int ry = 0;
            for (int i = 0; i < rects.Length; ++i)
            {
                if (lx > rects[i].X) lx = rects[i].X;
                if (ly > rects[i].Y) ly = rects[i].Y;
                if (rx < rects[i].X + rects[i].Width) rx = rects[i].X + rects[i].Width;
                if (ry < rects[i].Y + rects[i].Height) ry = rects[i].Y + rects[i].Height;
            }

            // Обрезаем края, оставляя только центральные блобчики
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(new Rectangle(lx, ly, rx - lx, ry - ly));
            unmanaged = cropFilter.Apply(unmanaged);

            ///  Инвертируем изображение
            AForge.Imaging.Filters.Invert InvertFilter1 = new AForge.Imaging.Filters.Invert();
            InvertFilter1.ApplyInPlace(unmanaged);

            //  Масштабируем до 200x200
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(200, 200);
            unmanaged = scaleFilter.Apply(unmanaged);

            return rez;
        }
    }
}

