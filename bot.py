import telebot
import os
import re
from PIL import Image
import urllib

import strings  # Several long strings that are sent by bot

### Functions of image receiving and sending, which are often used in code
def get_img(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    image_data = bot.download_file(file_info.file_path)
    img = Image.open(Image.io.BytesIO(image_data))
    return img

# Looks like it's OK to send just PIL Image with bot.send_photo
#def send_img(img, user_id, text=None):
#    img_data = img
#    bio = Image.io.BytesIO()
#    bio.name = 'image.jpeg'
#    img_data.save(bio, 'JPEG')
#    bio.seek(0)
#    if text:
#        bot.send_photo(user_id, bio, text)
#    else:
#        bot.send_photo(user_id, bio)



PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1686208190:AAFhWf0SMuGXHTTOP9C90CIeml9cKtPeEWo'

style_img = None

bot = telebot.TeleBot(TOKEN)


########## User greeting ##########

@bot.message_handler(commands=['start'])
def send_welcome(message):
    if message.from_user.id == 498505917:
        bot.reply_to(message, strings.welcome_me)
        bot.send_photo(message.from_user.id, urllib.request.urlopen(strings.url_Arkady).read(), 'Когда смотришь 50-го бота подряд')
    elif message.from_user.id == 327361684:
        bot.reply_to(message, strings.welcome_Arkady)
        bot.send_photo(message.from_user.id, urllib.request.urlopen(strings.url_Arkady).read(), 'Когда смотришь 50-го бота подряд')
    else:
        bot.reply_to(message, strings.welcome_string_1 + message.from_user.first_name)
    bot.send_message(message.from_user.id, strings.welcome_string_2)

@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.send_message(message.from_user.id, strings.welcome_string_2)


########## Set and show the Style Image ##########

@bot.message_handler(commands=['show_style_img'])
def show_style_img(message):
    if style_img:
        bot.send_photo(message.from_user.id, photo=style_img)
    else:
        bot.reply_to(message, 'Style Image не определено')

@bot.message_handler(commands=['set_style_img'])
def set_style_img(message):
    msg = bot.send_message(message.from_user.id, 'Прикрепите, пожалуйста, style image')
    bot.register_next_step_handler(msg, get_style_img)

def get_style_img(message):
    global style_img
    try:
        style_img = get_img(message)
        bot.send_message(message.from_user.id, 'Изображение стиля установлено!')
    except Exception as e:
        bot.reply_to(message, strings.error_message)


########## Get, transform and send back Content Image ##########

@bot.message_handler(content_types=['photo'])
def get_photo_message(message):
	if style_img:
		img = get_img(message)
		bot.send_photo(message.from_user.id, img, 'Возвращаю Вашу картинку')
	else:
		bot.send_message(message.from_user.id, 'Изображение стиля не установлено! Используйте /set_style_img')


########## Pointless communication with user :) ##########
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.from_user.id, 'Привет!')
    elif re.findall(strings.forbidden_words, message.text.lower()):
        bot.send_message(message.from_user.id, 'Мат в боте запрещён!')
    else:
        bot.send_message(message.from_user.id, 'Не понимаю, что это значит.')


bot.polling(none_stop=True)
#bot.start_webhook(listen="0.0.0.0", port=int(PORT), url_path=TOKEN)
#bot.bot.setWebhook('https://yourherokuappname.herokuapp.com/' + TOKEN)
