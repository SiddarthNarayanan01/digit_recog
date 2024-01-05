import pygame
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


model = load_model('DigitRecognition.h5')
model.compile(loss='categorical_crossentropy')
grid = []
for a in range(28):
	grid.append([])
	for b in range(28):
		grid[a].append(0)


WIDTH = 25
HEIGHT = 25

pygame.init()
display = pygame.display.set_mode((900, 800))
display.fill((0,0,0))
white = (255,255,255)
black = (0,0,0)
grey = (200, 200, 200)
clock = pygame.time.Clock()


def text_objects(text, font):
	surf = pygame.Surface(font.size(text))
	surf.fill(white)
	textSurface = font.render(text, True, black, surf)
	return textSurface, textSurface.get_rect()


def clear():
	global grid
	pygame.draw.rect(display, black, (0, 0, 750, 750))
	grid = []
	for a in range(28):
		grid.append([])
		for b in range(28):
			grid[a].append(0)


def guess():
	global grid
	num = int(input('Actual Number? '))

	true = to_categorical(num, 10)
	true = np.array([true]*100)
	grid = np.array(grid)
	grid = grid/255
	prediction = np.argmax(model.predict(grid.reshape(1, 28, 28, 1)), axis=-1)
	if prediction[0] == num:
		print(prediction[-1])
		sns.heatmap(grid)
		plt.show()
		clear()
	else:
		print(f"Sorry, I did't guess the right one, I guessed {prediction[0]}, "
				f"I will be training on how you write {num}!")
		grid = [grid]*100
		grid = np.array(grid)
		# Definitely overfitting to singular example
		# But maybe not so bad in this use case of recognizing handwriting 🤔
		model.fit(grid.reshape(100, 28, 28, 1), true, epochs=2)
		clear()


def button(msg, x, y, w, h, ic, ac, action=None):

	(x1, y1) = pygame.mouse.get_pos()
	if x + w > x1 > x and y + h > y1 > y:
		if action:
			action()
		click = pygame.mouse.get_pressed()
		# print(click)
		pygame.draw.rect(display, ic, (x, y, w, h))

	else:
		pygame.draw.rect(display, ac, (x, y, w, h))

	text = pygame.font.Font('../../Andale Mono.ttf', 20)
	textSurf, textRect = text_objects(msg, text)
	textRect.center = (x+ (w/2), y+(h/2))
	display.blit(textSurf, textRect)




def update():
	global grid

	keys = pygame.key.get_pressed()
	if keys[pygame.K_SPACE]:
		(x, y) = pygame.mouse.get_pos()
		if not(x < 40 or y < 40 or x > 700 or y > 700):
			column = x//WIDTH
			row = y//HEIGHT
			grid[row][column] = 255
			pygame.draw.rect(display, white, (WIDTH*column, HEIGHT*row, WIDTH, HEIGHT))
			# pygame.draw.rect(display, (240,240,240), (WIDTH*(column+1), HEIGHT*(row+1), WIDTH, HEIGHT))
			grid[row+1][column+1] = 160

	button('Guess', 375, 760, 80, 30, white, grey, guess)
	button('Clear', 525, 760, 80, 30, white, grey, clear)

run = True
while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	update()
	pygame.display.update()
	clock.tick(100)

pygame.quit()
