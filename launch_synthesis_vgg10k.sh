
for cpc_start in {0..40..10}
do 
./recover.sh $1 vgg10k convnet pretrained/vgg10k/convnet/teacher_model_bixattn.pt 10 10 0 $cpc_start 1.0
done 


