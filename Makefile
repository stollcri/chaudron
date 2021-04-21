gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	
	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/a

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg \
	./wip-img/b

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/Charles_Turzak_-_Chicago.jpeg \
	./wip-img/c
