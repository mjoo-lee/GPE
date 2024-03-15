GPU=0
SHOT=16

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars #oxford_pets food101 ucf101 caltech101 sun397 #imagenet
#for dataset in caltech101 imagenet
do
    for seed in 1 2 3
    do
    sh scripts/rpo_prime/base2new_train.sh ${dataset} ${seed} ${GPU} main_final990 ${SHOT}
    #sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_9_9 ${SHOT} base
    sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_final990 ${SHOT} 30 new
    done
done

