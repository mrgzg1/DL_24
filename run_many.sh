# example base we run with:
# python main.py --experiment_name text_exp --data_path /data/DL_24/data --gpu_id 0 -b 512


EXP_NAME=b64_cnn_vr_l
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 6 -b 64  >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b128_cnn_vr_l
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 5 -b 128 >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b256_cnn_vr_l
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 4 -b 256 >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b256_vit_vr_l
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 3 -b 1024 --encoder_type vit --loss_type vicreg >> resources/$EXP_NAME.log 2>&1 &


