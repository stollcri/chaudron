gen:
	rm -f wip-img/a/*
	rm -f wip-img/b/*
	rm -f wip-img/c/*
	rm -f wip-img/d/*
	rm -f wip-img/e/*
	
	./transfer_texture.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/63313834792__57FB485E-76F1-40D4-B860-D4CBF3AAA7E5.jpeg \
	--target-dir wip-img/a

	./transfer_texture.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/Charles_Turzak_-_Chicago.jpeg \
	--target-dir wip-img/b

	./transfer_texture.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/IMG_0829.jpeg \
	--target-dir wip-img/c
	
	./transfer_texture.py \
	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
	--style-image ./images/texture/JoseGuadalupePosada-8139470273_43b3b2f7ab_o.jpeg \
	--target-dir wip-img/d
# 
# 	./transfer_texture.py \
# 	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
# 	--style-image ./images/texture/10jackson-pollock1-superJumbo.jpg \
# 	--target-dir wip-img/e
# 
# 	./transfer_texture.py \
# 	--content-image ./images/canvas/ROLL_16_IMG_9387_POS.jpg \
# 	--style-image ./images/texture/Albert_Gleizes_l_Homme_au_Balcon.jpg \
# 	--target-dir wip-img/f
