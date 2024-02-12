using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public delegate void FormUpdater(double progress, double error, TimeSpan time);

    public delegate void UpdateTLGMessages(string msg);

    public partial class Form1 : Form
    {
        /// <summary>
        /// Чат-бот AIML
        /// </summary>
        AIMLBotik botik = new AIMLBotik();

        TLGBotik tlgBot;

        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        GenerateImage generator = new GenerateImage();
        
        /// <summary>
        /// Обёртка для ActivationNetwork из Accord.Net
        /// </summary>
        AccordNet AccordNet = null;

        private readonly Dictionary<string, Func<int[], BaseNetwork>> networksFabric;
        private Dictionary<string, BaseNetwork> networksCache = new Dictionary<string, BaseNetwork>();

        /// <summary>
        /// Текущая выбранная через селектор нейросеть
        /// </summary>
        public BaseNetwork net
        {
            get
            {
                var selectedItem = (string)netTypeBox.SelectedItem;
                if (!networksCache.ContainsKey(selectedItem))
                    networksCache.Add(selectedItem, CreateNetwork(selectedItem));

                return networksCache[selectedItem];
            }
        }

        public Form1(Dictionary<string, Func<int[], BaseNetwork>> networksFabric)
        {
            InitializeComponent();
            this.networksFabric = networksFabric;
            netTypeBox.Items.AddRange(this.networksFabric.Keys.Select(s => (object)s).ToArray());
            netTypeBox.SelectedIndex = 0;
            generator.FigureCount = (int)classCounter.Value;
            tlgBot = new TLGBotik(net, botik, new UpdateTLGMessages(UpdateTLGInfo));
            button3_Click(this, null);
        }

		public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
		{
			if (progressBar1.InvokeRequired)
			{
				progressBar1.Invoke(new FormUpdater(UpdateLearningInfo),new Object[] {progress, error, elapsedTime});
				return;
			}
            StatusLabel.Text = "Accuracy: " + error.ToString();
            int prgs = (int)Math.Round(progress*100);
			prgs = Math.Min(100, Math.Max(0,prgs));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = prgs;
		}

        private BaseNetwork CreateNetwork(string networkName)
        {
            string n = networkName;
            var network = networksFabric[networkName](CurrentNetworkStructure());
            network.updateDelegate += UpdateLearningInfo;
            return network;
        }

        private int[] CurrentNetworkStructure()
        {
            return netStructureBox.Text.Split(';').Select(int.Parse).ToArray();
        }

        public void UpdateTLGInfo(string message)
        {
            if (TLGUsersMessages.InvokeRequired)
            {
                TLGUsersMessages.Invoke(new UpdateTLGMessages(UpdateTLGInfo), new Object[] { message });
                return;
            }
            TLGUsersMessages.Text += message + Environment.NewLine;
        }

        private void set_result(Sample figure)
        {
            label1.Text = figure.ToString();

            if (figure.Correct())
                label1.ForeColor = Color.Green;
            else
                label1.ForeColor = Color.Red;

            label1.Text = "Распознано : " + figure.recognizedClass.ToString();

            label8.Text = String.Join("\n", net.getOutput().Select(d => d.ToString()));
        }

        private async Task<double> train_networkAsync(int epoches, double acceptable_error, bool parallel = true)
        {
            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;

            //  Обучение запускаем асинхронно, чтобы не блокировать форму
            var curNet = net;
            double f = await Task.Run(() => curNet.TrainOnDataSet(generator.LearnImages, epoches, acceptable_error, parallel));

            label1.Text = "Щелкните на картинку для теста нового образа";
            label1.ForeColor = Color.Green;
            groupBox1.Enabled = true;
            StatusLabel.Text = "Accuracy: " + f.ToString();
            StatusLabel.ForeColor = Color.Green;
            return f;

        }

        private void button1_Click(object sender, EventArgs e)
        {
            #pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync((int)EpochesCounter.Value, (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
            #pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        }

        private void button2_Click(object sender, EventArgs e)
        {
            this.Enabled = false;
            //  Тут просто тестирование новой выборки

            double accuracy = net.TestOnDataSet(generator.TestImages);
            
            StatusLabel.Text = string.Format("Точность на тестовой выборке : {0,5:F2}%", accuracy*100);
            if (accuracy*100 >= AccuracyCounter.Value)
                StatusLabel.ForeColor = Color.Green;
            else
                StatusLabel.ForeColor = Color.Red;

            this.Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //  Проверяем корректность задания структуры сети
            int[] structure = netStructureBox.Text.Split(';').Select((c) => int.Parse(c)).ToArray();
            if (structure.Length < 2 || structure[structure.Length - 1] != generator.FigureCount)
            {
                MessageBox.Show("А давайте вы структуру сети нормально запишите, ОК?", "Ошибка", MessageBoxButtons.OK);
                return;
            };
            
            //AccordNet = new AccordNet(structure);

            // Чистим старые подписки сетей
            foreach (var network in networksCache.Values)
                network.updateDelegate -= UpdateLearningInfo;
            // Пересоздаём все сети с новой структурой
            networksCache = networksCache.ToDictionary(oldNet => oldNet.Key, oldNet => CreateNetwork(oldNet.Key));

            tlgBot.SetNet(net);
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.FigureCount = (int)classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            int outputNeurons;
            if (int.TryParse(vals.Last(),out outputNeurons))
            {
                vals[vals.Length - 1] = classCounter.Value.ToString();
                netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
            }
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке такого же размера";
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            var phrase = AIMLInput.Text;
            if (phrase.Length > 0)
                AIMLOutput.Text += botik.Talk(666, "Default", phrase) + Environment.NewLine;
        }

        private void TLGBotOnButton_Click(object sender, EventArgs e)
        {
            tlgBot.Act();
            TLGBotOnButton.Enabled = false;
        }
    }
}
