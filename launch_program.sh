DEVICE=$1
DATASET=$2
MODEL=$3
FUSION='ensemble'

echo ">>>> Training teacher network for distillation"
CUDA_VISIBLE_DEVICES=$DEVICE python3 -W ignore train_teacher_network.py \
--dataset $DATASET \
--device cuda:0 \
--batch_size 128 \
--lr 0.001 \
--epochs 100 \
--model $3 \
--fusion $FUSION \
--with_cross_attn

echo ">>>> Training teacher network for soft labels"
CUDA_VISIBLE_DEVICES=$DEVICE python3 -W ignore train_teacher_network.py \
--dataset $DATASET \
--device cuda:0 \
--batch_size 128 \
--lr 0.001 \
--epochs 100 \
--model $3 \
--fusion $FUSION 

echo ">>>> Obtaining embeddings...."
CUDA_VISIBLE_DEVICES=$DEVICE python3 -W ignore obtain_embeddings.py \
--batch_size 128 \
--arch_path pretrained/$DATASET/$MODEL/teacher_model_bixattn.pt \
 --model $MODEL \
 --dataset $DATASET

echo ">>>> Obtaining initialization data...."
CUDA_VISIBLE_DEVICES=$DEVICE python3 -W ignore obtain_init_data.py --dataset $DATASET

echo ">>>> Generate distilled dataset...."
 ./launch_synthesis_vgg10k.sh $DEVICE

echo ">>>> Evaluate on the distilled dataset..."
./evaluate_syn_data_vgg10k.sh 0 pretrained/$DATASET/$MODEL/teacher_model_wo_attn.pt $MODEL 0.0 10 1
./evaluate_syn_data_vgg10k.sh 0 pretrained/$DATASET/$MODEL/teacher_model_wo_attn.pt $MODEL 0.0 10 10
./evaluate_syn_data_vgg10k.sh 0 pretrained/$DATASET/$MODEL/teacher_model_wo_attn.pt $MODEL 0.0 10 20
./evaluate_syn_data_vgg10k.sh 0 pretrained/$DATASET/$MODEL/teacher_model_wo_attn.pt $MODEL 0.0 10 50
