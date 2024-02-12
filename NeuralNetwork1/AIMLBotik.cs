using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AIMLbot;

namespace NeuralNetwork1
{
    class AIMLBotik
    {
        readonly Bot bot;
        readonly Dictionary<long, User> users = new Dictionary<long, User>();

        public AIMLBotik()
        {
            bot = new Bot();
            bot.loadSettings();
            bot.isAcceptingUserInput = false;
            bot.loadAIMLFromFiles();
            bot.isAcceptingUserInput = true;
        }

        public string Talk(long userId, string userName, string phrase)
        {
            var result = "";
            User user;
            if (!users.ContainsKey(userId))
            {
                user = new User(userId.ToString(), bot);
                users.Add(userId, user);
            }
            else
            {
                user = users[userId];
            }
            result += bot.Chat(new Request(phrase, user, bot)).Output;
            return result;
        }
    }
}
