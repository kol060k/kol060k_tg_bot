import telebot
import os
import re

import strings.py # Немного полезных строк, используемых в коде

PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1686208190:AAFhWf0SMuGXHTTOP9C90CIeml9cKtPeEWo'

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f'Я бот. Приятно познакомиться, {message.from_user.first_name}')
    bot.send_message(message.from_user.id, 'Я обучен преобразовывать картинки с применением к ним определённого стиля (NST), а также парочке мемов, на которые у моего создателя хватило фантазии и испорченности. Наслаждайтесь :)')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.from_user.id, 'Привет!')
    elif re.findall(forbidden_strings, message.text.lower()):
        bot.send_message(message.from_user.id, 'Мат в боте запрещён!')
    else:
        bot.send_message(message.from_user.id, 'Не понимаю, что это значит.')


bot.polling(none_stop=True)
#bot.start_webhook(listen="0.0.0.0", port=int(PORT), url_path=TOKEN)
#bot.bot.setWebhook('https://yourherokuappname.herokuapp.com/' + TOKEN)
