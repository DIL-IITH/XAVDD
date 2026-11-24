alpha=$4
temperature=$5 
ipc=$6 
CUDA_VISIBLE_DEVICES=$1 python3 -W ignore evaluate_syn_data.py \
--device cuda:0 \
--arch_path $2 \
--dataset vgg10k \
--model $3 \
--batch_size 32 \
--lr 0.001 \
--syn_data_path syn_data/vgg10k/$3/syn_av_init_attn/ \
--ipc $ipc \
--soft_label 1 \
--iterations 3 \
--alpha $alpha \
--temperature $temperature \
--type ours 
