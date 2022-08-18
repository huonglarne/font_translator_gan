set -ex
python train.py --dataroot ./datasets/font --model font_translator_gan --name test_new_dataset --no_dropout --gpu_ids 0 --batch_size 56