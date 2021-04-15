gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	
	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/a -v --learning-rate 5.0
	
	# ./transfer_texture.py \
	# ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# ./images/texture/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg \
	# ./wip-img/b
	# 
	# ./transfer_texture.py \
	# ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# ./images/texture/ROLL_36_IMG_0297_POS.jpeg \
	# ./wip-img/c
	# # ./transfer_texture.py \
	# # ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# # ./images/texture/MCEscher-54087.jpeg \
	# # ./wip-img/c
