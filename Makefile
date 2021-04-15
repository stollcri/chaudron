gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	
	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/a \
	--learning-rate 2.0 \
	--style-layer-weights 1.0 0.0 0.0 0.0 0.0

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/b \
	--learning-rate 2.0 \
	--style-layer-weights 0.0 1.0 0.0 0.0 0.0

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/c \
	--learning-rate 2.0 \
	--style-layer-weights 0.0 0.0 1.0 0.0 0.0

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/d \
	--learning-rate 2.0 \
	--style-layer-weights 0.0 0.0 0.0 1.0 0.0

	./transfer_texture.py \
	./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	./wip-img/e \
	--learning-rate 2.0 \
	--style-layer-weights 0.0 0.0 0.0 0.0 1.0


