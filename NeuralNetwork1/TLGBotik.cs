using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using Telegram.Bot.Types.InputFiles;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.StartPanel;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Window;


namespace NeuralNetwork1
{
    class TLGBotik
    {
        public Telegram.Bot.TelegramBotClient botik = null;

        private UpdateTLGMessages formUpdater;

        private readonly AIMLBotik aiml;
        public string Username { get; }

        private BaseNetwork perseptron = null;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public TLGBotik(BaseNetwork net, AIMLBotik aiml, UpdateTLGMessages updater)
        {
            var botKey = System.IO.File.ReadAllText("botkey.txt");
            botik = new Telegram.Bot.TelegramBotClient(botKey);
            formUpdater = updater;
            perseptron = net;
            this.aiml = aiml;
            Username = botik.GetMeAsync().Result.Username;
        }

        public void SetNet(BaseNetwork net)
        {
            perseptron = net;
            formUpdater("Net updated!");
        }

        private async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            //  Тут очень простое дело - банально отправляем назад сообщения
            var message = update.Message;
            formUpdater("Тип сообщения : " + message.Type.ToString());
            var username = message.Chat.FirstName;
            var chatId = message.Chat.Id;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, messageText),
                    cancellationToken: cancellationToken);
                return;
            }
            //  Получение файла (картинки)
            if (message.Type == Telegram.Bot.Types.Enums.MessageType.Photo)
            {
                formUpdater("Picture loadining started");
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = botik.GetFileAsync(photoId).Result;
                var imageStream = new MemoryStream();
                await botik.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                var img = System.Drawing.Image.FromStream(imageStream);

                System.Drawing.Bitmap bm = new System.Drawing.Bitmap(img);

                Sample sample = GenerateImage.GenerateFigure(bm);
                string recogn;
                switch (perseptron.Predict(sample))
                {
                    case FigureType.Suzuki: recogn = "сузуки"; break;
                    case FigureType.Mercedes: recogn = "мерседес"; break;
                    case FigureType.Infinity: recogn = "инфинити"; break;
                    case FigureType.Citroen: recogn = "ситроен"; break;
                    case FigureType.Opel: recogn = "опель"; break;
                    case FigureType.Hyundai: recogn = "хендай"; break;
                    case FigureType.Audi: recogn = "ауди"; break;
                    case FigureType.Mitsubishi: recogn = "митсубиси"; break;
                    default: recogn = "XXX"; break;
                }

                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, "распознал " + recogn),
                    cancellationToken: cancellationToken);
                formUpdater("Picture recognized!");
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await botik.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await botik.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
            await botik.SendTextMessageAsync(message.Chat.Id, "XXX", cancellationToken: cancellationToken);
        }
        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public bool Act()
        {
            try
            {
                botik.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
                {   // Подписываемся только на сообщения
                    AllowedUpdates = new[] { UpdateType.Message }
                },
                cancellationToken: cts.Token);
                // Пробуем получить логин бота - тестируем соединение и токен
                Console.WriteLine($"Connected as {botik.GetMeAsync().Result}");
            }
            catch (Exception e)
            {
                return false;
            }
            return true;
        }

        public void Stop()
        {
            cts.Cancel();
        }

    }
}
