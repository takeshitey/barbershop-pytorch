python seg_alignment.py \
--exp_dir barbershop_reproduce \
--stylegan_weights stylegan2-ffhq-config-f.pt \
--psp_encoder_weights psp_celebs_seg_to_face.pt \
--bisenet_weights 79999_iter.pth \
--size 1024 \
--lr 1e-2 \
--step 20 \
--noise 0.03 \
--noise_ramp 0.07 \
--noise_regularize 1e3 \
--refer_images data/images/01012.jpg data/images/02602.jpg