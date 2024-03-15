GPU=$1
SHOT=16
EPOCH=15

##Train on food101##
for seed in 1 2 3
do
    for cfg in main_K24
    do
        # training
        sh scripts/rpo/base2new_train.sh food101 ${seed} ${GPU} ${cfg} ${SHOT}
    done
done
####################

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101
do
    for seed in 1 2 3
    do
        for cfg in main_K24
        do
            # # training
            # bash scripts/rpo/base2new_train.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT}
            
            # evaluation for each epoch
                sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K24 ${SHOT} 10 base
                sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} 10 new
        done
    done
done

# sh scripts/rpo/base2new_test.sh food101 1 ${GPU} main_K24 ${SHOT} 1 base

# for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101
# do
#     for seed in 1 2 3
#     do
#         for cfg in main_K24
#         do
#                 # for EPOCH in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#                 # do
#                 #sh scripts/rpo/base2new_test.sh food101 ${seed} ${GPU} main_K24 ${SHOT} ${EPOCH} base
#                 sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K24 ${SHOT} 10 base
#                 #sh scripts/rpo/base2new_test.sh food101 ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} new
#                 sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} 10 new
#                 #done
#         done
#     done
# done