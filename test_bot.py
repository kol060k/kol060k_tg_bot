import telebot
import os
import re
from PIL import Image

import strings  # Немного полезных строк, используемых в коде

PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1686208190:AAFhWf0SMuGXHTTOP9C90CIeml9cKtPeEWo'

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f'Я бот. Приятно познакомиться, {message.from_user.first_name}')
    bot.send_message(message.from_user.id, strings.welcome_string)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.from_user.id, 'Привет!')
    elif re.findall(strings.forbidden_words, message.text.lower()):
        bot.send_message(message.from_user.id, 'Мат в боте запрещён!')
    else:
        bot.send_message(message.from_user.id, 'Не понимаю, что это значит.')

@bot.message_handler(content_types=['photo'])
def get_photo_message(message):
    #Получение
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    image_data = bot.download_file(file_info.file_path)
    img = Image.open(Image.io.BytesIO(image_data))

    #Отправка
    img_data = img
    bio = Image.io.BytesIO()
    bio.name = 'image.jpeg'
    img_data.save(bio, 'JPEG')
    bio.seek(0)
    bot.send_photo(message.from_user.id, photo=bio)


bot.polling(none_stop=True)
#bot.start_webhook(listen="0.0.0.0", port=int(PORT), url_path=TOKEN)
#bot.bot.setWebhook('https://yourherokuappname.herokuapp.com/' + TOKEN)
