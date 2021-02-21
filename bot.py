import telebot
from telebot import types
import os
import re
from PIL import Image
import urllib
import torchvision.transforms as transforms

import strings  # Several long strings that are sent by bot

import NST_class # File with NST model functions and necessary imports
print('Model creation started')
model = NST_class.NST()
print('Model successfully created')

### Functions of image receiving and sending, which are often used in code
def get_img(message):
	file_id = message.photo[-1].file_id
	file_info = bot.get_file(file_id)
	image_data = bot.download_file(file_info.file_path)
	img = Image.open(Image.io.BytesIO(image_data))
	return img

# Looks like it's OK to send just PIL Image with bot.send_photo
#def send_img(img, user_id, text=None):
#	img_data = img
#	bio = Image.io.BytesIO()
#	bio.name = 'image.jpeg'
#	img_data.save(bio, 'JPEG')
#	bio.seek(0)
#	if text:
#		bot.send_photo(user_id, bio, text)
#	else:
#		bot.send_photo(user_id, bio)


### Image transformers: PIL <=> Tensor
imsize = 256 
loader = transforms.Compose([
	transforms.Resize(imsize),  # нормируем размер изображения
	transforms.CenterCrop(imsize),
	transforms.ToTensor()])  # превращаем в удобный формат

unloader = transforms.ToPILImage() # тензор в кратинку


PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1686208190:AAFhWf0SMuGXHTTOP9C90CIeml9cKtPeEWo'


style_img_1 = {}
style_img_1_tensor = {}
style_img_2 = {}
style_img_2_tensor = {}
mask = None
mask_tensor = None

bot = telebot.TeleBot(TOKEN)


def init_keyboard():
	markup = types.ReplyKeyboardMarkup(row_width=2)
	itembtn_style = types.KeyboardButton(text='Изображения стиля')
	itembtn21 = types.KeyboardButton(text='Перенос 1 стиля')
	itembtn22 = types.KeyboardButton(text='Перенос 2 стилей')
	itembtn23 = types.KeyboardButton(text='Перенос 1 стиля + маска')
	itembtn24 = types.KeyboardButton(text='Перенос 2 стилей + маска')
	itembtn_help = types.KeyboardButton(text='Помощь')
	itembtn_commands = types.KeyboardButton(text='Показать команды')
	markup.row(itembtn_style)
	markup.row(itembtn21, itembtn22)
	markup.row(itembtn23, itembtn24)
	markup.row(itembtn_help, itembtn_commands)
	return markup

def style_keyboard():
	markup = types.ReplyKeyboardMarkup(row_width=2)
	itembtn_show = types.KeyboardButton(text='Показать стили')
	itembtn_1_style = types.KeyboardButton(text='Установить 1-й стиль')
	itembtn_2_style = types.KeyboardButton(text='Установить 2-й стиль')
	itembtn_swap_styles = types.KeyboardButton(text='Поменять стили местами')
	itembtn_del_style = types.KeyboardButton(text='Удалить 2-й стиль')
	itembtn_del_both = types.KeyboardButton(text='Удалить все стили')
	itembtn_menu = types.KeyboardButton(text='В меню')
	markup.row(itembtn_show, itembtn_1_style, itembtn_2_style)
	markup.row(itembtn_swap_styles, itembtn_del_style, itembtn_del_both)
	markup.row(itembtn_menu)
	return markup
	
init_markup = init_keyboard()
style_markup = style_keyboard()



#######################################################
#################### User greeting ####################
#######################################################

@bot.message_handler(commands=['start'])
def send_welcome(message):
	if message.from_user.id == 498505917:
		bot.reply_to(message, strings.welcome_me)
#		bot.send_photo(message.from_user.id, urllib.request.urlopen(strings.url_Arkady).read(), 'Когда смотришь 50-го бота подряд')
	elif message.from_user.id == 327361684:
		bot.reply_to(message, strings.welcome_Arkady)
		bot.send_photo(message.from_user.id, urllib.request.urlopen(strings.url_Arkady).read(), 'Когда смотришь 50-го бота подряд')
	else:
		bot.reply_to(message, strings.welcome_string_1 + message.from_user.first_name)
	bot.send_message(message.from_user.id, strings.welcome_string_2, reply_markup=init_markup)

@bot.message_handler(func=lambda message: message.text == 'Помощь')
@bot.message_handler(commands=['help'])
def send_welcome(message):
	bot.send_message(message.from_user.id, strings.welcome_string_2)
	
@bot.message_handler(func=lambda message: message.text == 'Показать команды')
@bot.message_handler(commands=['show_commands'])
def send_welcome(message):
	bot.send_message(message.from_user.id, strings.commands)
	
@bot.message_handler(func=lambda message: message.text == 'В меню')
@bot.message_handler(commands=['main_keyboard'])
def send_welcome(message):
	bot.send_message(message.from_user.id, strings.return_message, reply_markup=init_markup)



####################################################################
#################### Work with the Style Images ####################
####################################################################

@bot.message_handler(func=lambda message: message.text == 'Изображения стиля')
@bot.message_handler(commands=['style_keyboard'])
def show_style_keyboard(message):
	if style_img_1.get(message.from_user.id) == None:
		bot.send_message(message.from_user.id, strings.style_handler_0, reply_markup=style_markup)
	elif style_img_2.get(message.from_user.id) == None:
		bot.send_message(message.from_user.id, strings.style_handler_1, reply_markup=style_markup)
	else:
		bot.send_message(message.from_user.id, strings.style_handler_2, reply_markup=style_markup)

@bot.message_handler(func=lambda message: message.text == 'Показать стили')
@bot.message_handler(commands=['show_style_img'])
def show_style_img(message):
	if style_img_1.get(message.from_user.id) != None:
		bot.send_photo(message.from_user.id, style_img_1[message.from_user.id], strings.style_1_message)
		if style_img_2.get(message.from_user.id) != None:
			bot.send_photo(message.from_user.id, style_img_2[message.from_user.id], strings.style_2_message)
	else:
		bot.reply_to(message, strings.first_style_missing)

@bot.message_handler(func=lambda message: message.text == 'Установить 1-й стиль')
@bot.message_handler(commands=['set_style_img'])
def set_style_img(message):
	msg = bot.send_message(message.from_user.id, strings.style_request)
	bot.register_next_step_handler(msg, get_style_img)

def get_style_img(message):
	try:
		img = get_img(message)
		style_img_1[message.from_user.id] = img
		style_img_1_tensor[message.from_user.id] = loader(img).unsqueeze(0)
		bot.send_message(message.from_user.id, strings.style_1_success)
	except Exception as e:
		bot.reply_to(message, strings.error_message)

@bot.message_handler(func=lambda message: message.text == 'Установить 2-й стиль')
@bot.message_handler(commands=['set_2nd_style_img'])
def set_2nd_style_img(message):
	if style_img_1.get(message.from_user.id) != None:
		msg = bot.send_message(message.from_user.id, strings.style_request)
		bot.register_next_step_handler(msg, get_2nd_style_img)
	else:
		msg = bot.send_message(message.from_user.id, strings.style_1_request)
		bot.register_next_step_handler(msg, get_style_img)

def get_2nd_style_img(message):
	try:
		img = get_img(message)
		style_img_2[message.from_user.id] = img
		style_img_2_tensor[message.from_user.id] = loader(img).unsqueeze(0)
		bot.send_message(message.from_user.id, strings.style_2_success)
	except Exception as e:
		bot.reply_to(message, strings.error_message)

@bot.message_handler(func=lambda message: message.text == 'Удалить 2-й стиль')	
@bot.message_handler(commands=['del_2nd_style_img'])
def del_2nd_style_img(message):
	style_img_2.pop(message.from_user.id, None)
	style_img_2_tensor.pop(message.from_user.id, None)
	bot.send_message(message.from_user.id, strings.delete_2_success)
	
@bot.message_handler(func=lambda message: message.text == 'Удалить все стили')	
@bot.message_handler(commands=['del_style_imgs'])
def del_2nd_style_img(message):
	style_img_2.pop(message.from_user.id, None)
	style_img_2_tensor.pop(message.from_user.id, None)
	style_img_1.pop(message.from_user.id, None)
	style_img_1_tensor.pop(message.from_user.id, None)
	bot.send_message(message.from_user.id, strings.delete_success)

@bot.message_handler(func=lambda message: message.text == 'Поменять стили местами')
@bot.message_handler(commands=['swap_style_imgs'])
def del_2nd_style_img(message):
	if style_img_2.get(message.from_user.id) != None:
		img1 = style_img_1.pop(message.from_user.id, None)
		tens1 = style_img_1_tensor.pop(message.from_user.id, None)
		img2 = style_img_2.pop(message.from_user.id, None)
		tens2 = style_img_2_tensor.pop(message.from_user.id, None)
		style_img_1[message.from_user.id] = img2
		style_img_2[message.from_user.id] = img1
		style_img_1_tensor[message.from_user.id] = tens2
		style_img_2_tensor[message.from_user.id] = tens1
		bot.send_message(message.from_user.id, strings.swap_success)
	elif style_img_1.get(message.from_user.id) != None:
		bot.send_message(message.from_user.id, strings.secong_style_missing)
	else:
		bot.send_message(message.from_user.id, strings.both_styles_missing)



####################################################################################
#################### Get, transform and send back Content Image ####################
####################################################################################

# Handler of NST commands that initializes the appropriate checks and then the corresponding NST
@bot.message_handler(func=lambda message: message.text in ['Перенос 1 стиля', 'Перенос 2 стилей', 'Перенос 1 стиля + маска', 'Перенос 2 стилей + маска'])
@bot.message_handler(commands=['NST_1_style', 'NST_2_styles', 'NST_1_style_with_mask', 'NST_2_styles_with_mask'])
def NST_handler(message):
	if message.text in ['Перенос 1 стиля', '/NST_1_style']:
		if style_img_1.get(message.from_user.id) != None:
			msg = bot.send_message(message.from_user.id, strings.content_request)
			bot.register_next_step_handler(msg, NST_1_style)
		else:
			bot.send_message(message.from_user.id, strings.first_style_missing)
	if message.text in ['Перенос 2 стилей', '/NST_2_styles']:
		if style_img_2.get(message.from_user.id) != None:
			msg = bot.send_message(message.from_user.id, strings.content_request)
			bot.register_next_step_handler(msg, NST_2_styles)
		elif style_img_1.get(message.from_user.id) != None:
			bot.send_message(message.from_user.id, strings.secong_style_missing)
		else:
			bot.send_message(message.from_user.id, strings.both_styles_missing)
	if message.text in ['Перенос 1 стиля + маска', '/NST_1_style_with_mask']:
		bot.send_message(message.from_user.id, 'Кажется, это ещё не готово. Зайдите позже :)')
	if message.text in ['Перенос 2 стилей + маска', '/NST_2_styles_with_mask']:
		bot.send_message(message.from_user.id, 'Кажется, это ещё не готово. Зайдите позже :)')

# NST with 1 style	
def NST_1_style(message):
	try:
		content_img = get_img(message)
		content_img = loader(content_img).unsqueeze(0)
		input_img = content_img.clone()
		bot.send_message(message.from_user.id, strings.NST_wait)
		output = model.run_style_transfer(content_img, style_img_1_tensor[message.from_user.id], input_img)
		output = output.cpu().clone()
		output = output.squeeze(0)
		output = unloader(output)
		bot.send_photo(message.from_user.id, output, strings.NST_success)
	except Exception as e:
		bot.reply_to(message, strings.error_message)

# NST with 2 styles	
def NST_2_styles(message):
	try:
		content_img = get_img(message)
		content_img = loader(content_img).unsqueeze(0)
		input_img = content_img.clone()
		bot.send_message(message.from_user.id, strings.NST_wait)
		output = model.run_style_transfer(content_img, style_img_1_tensor[message.from_user.id], 
										  input_img, style_img_2=style_img_2_tensor[message.from_user.id])
		output = output.cpu().clone()
		output = output.squeeze(0)
		output = unloader(output)
		bot.send_photo(message.from_user.id, output, strings.NST_success)
	except Exception as e:
		bot.reply_to(message, strings.error_message)



##############################################################################
#################### Pointless communication with user :) ####################
##############################################################################

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
	if message.text.lower() == 'привет':
		bot.send_message(message.from_user.id, 'Привет!')
	elif re.findall(strings.forbidden_words, message.text.lower()):
		bot.send_message(message.from_user.id, 'Мат в боте запрещён!')
	else:
		bot.send_message(message.from_user.id, 'Не понимаю, что это значит.')
		
@bot.message_handler(content_types=['document'])
def nice_pic(message):
	bot.send_message(message.from_user.id, strings.no_documents)

@bot.message_handler(content_types=['photo'])
def nice_pic(message):
	nice_pic_photo = open('nice_pic.jpg', 'rb')
	bot.send_sticker(message.from_user.id, nice_pic_photo)
	
@bot.message_handler(content_types=['voice'])
def use_words(message):
	use_words_photo = open('use_words.jpg', 'rb')
	bot.send_sticker(message.from_user.id, use_words_photo)
	
@bot.message_handler(content_types=['audio'])
def volume_up(message):
	volume_up_photo = open('volume_up.jpg', 'rb')
	bot.send_sticker(message.from_user.id, volume_up_photo)
	
@bot.message_handler(content_types=['sticker'])
def sticker(message):
	sticker_photo = open('sticker.jpg', 'rb')
	bot.send_sticker(message.from_user.id, sticker_photo)
	
@bot.message_handler(content_types=['video'])
def deleted_video(message):
	deleted_video_photo = open('deleted_video.png', 'rb')
	bot.send_sticker(message.from_user.id, deleted_video_photo)
	
@bot.message_handler(content_types=['location'])
def big_brother(message):
	big_brother_photo = open('big_brother.png', 'rb')
	bot.send_sticker(message.from_user.id, big_brother_photo)


bot.polling(none_stop=True)
#bot.start_webhook(listen="0.0.0.0", port=int(PORT), url_path=TOKEN)
#bot.bot.setWebhook('https://yourherokuappname.herokuapp.com/' + TOKEN)
