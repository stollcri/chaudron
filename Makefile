gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	rm -f wip-img/d/*
	rm -f wip-img/e/*
	
	./generate.py \
	--content-image ./images/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg \
	--target-dir wip-img/a
	
	./generate.py \
	--content-image ./images/Green_Sea_Turtle_grazing_seagrass.jpg \
	--style-image ./images/The_Great_Wave_off_Kanagawa.jpg \
	--target-dir wip-img/b
	
	./generate.py \
	--content-image ./images/Tuebingen_Neckarfront.jpg \
	--style-image ./images/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg \
	--target-dir wip-img/c
	
	./generate.py \
	--content-image ./images/Tuebingen_Neckarfront.jpg \
	--style-image ./images/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg \
	--target-dir wip-img/d
	
	./generate.py \
	--content-image ./images/Tuebingen_Neckarfront.jpg \
	--style-image ./images/Vassily_Kandinsky_1913_-_Composition_7.jpg \
	--target-dir wip-img/e
	
	./generate.py \
	--content-image ./images/Green_Sea_Turtle_grazing_seagrass.jpg \
	--style-image ./images/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg \
	--target-dir wip-img/f
