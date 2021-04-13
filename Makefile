gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	rm -f wip-img/d/*
	rm -f wip-img/e/*
	
	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg \
	# --target-dir wip-img/a
	./generate.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/doctor-who-exploding-tardis-blue-box-exploding-56_28747.jpg \
	--target-dir wip-img/a
	
	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/IMG_2077.jpeg \
	# --target-dir wip-img/b
	./generate.py \
	--content-image ./images/texture/IMG_2077.jpeg \
	--style-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--target-dir wip-img/b

	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/The_Great_Wave_off_Kanagawa.jpg \
	# --target-dir wip-img/c
	./generate.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/tardis-doctor-who-900x506.jpg \
	--target-dir wip-img/c

	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/Monet_Cliff_walk.jpg \
	# --target-dir wip-img/d
	./generate.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/tardis-pattern-vector.jpg \
	--target-dir wip-img/d

	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/10jackson-pollock1-superJumbo.jpg \
	# --target-dir wip-img/e

	# ./generate.py \
	# --content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	# --style-image ./images/texture/Albert_Gleizes_l_Homme_au_Balcon.jpg \
	# --target-dir wip-img/f
