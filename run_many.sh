# example base we run with:
# python main.py --experiment_name text_exp --data_path /data/DL_24/data --gpu_id 0 -b 512


EXP_NAME=b1024_cnn_vr
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 0 -b 1024 --encoder_type cnn --loss_type vicreg >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b1024_cnn_byol
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 1 -b 1024 --encoder_type cnn --loss_type byol >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b1024_vit_vr
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 2 -b 1024 --encoder_type vit --loss_type vicreg >> resources/$EXP_NAME.log 2>&1 &

EXP_NAME=b1024_vit_byol
python main.py --experiment_name $EXP_NAME --data_path /data/DL_24/data --gpu_id 3 -b 1024 --encoder_type vit --loss_type byol >> resources/$EXP_NAME.log 2>&1 &


