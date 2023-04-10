#!/bin/bash
# I'm using 333 images of the test set for validation here
subject="Recloser gci"

echo Round 0 of 5
echo Prediction train: round $((round))

python send_mail.py --event "Prediction train: round $round" --subject "$subject"

time python predict.py --model deeplabv3plus_mobilenet --val_batch_size 16 --input /home/jovyan/work/deeplab/data/train/ --num_classes 16 --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16_recloser_gci.pth --save_dir recloser_gci_r0_prediction_train --separable_conv

echo Postprocess train: round 0
python send_mail.py --event "Postprocess train: round 0" --subject "$subject"
exit_status=1
fail=0
while [ "${exit_status}" -ne 0 ]
do
    time python /home/jovyan/work/deeplab/utils/post_process_crf_filter_list.py --image_path /home/jovyan/work/deeplab/data/train/ --seg_path /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r0_prediction_train/ --save_path /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r$((round))_postprocess_train/
    exit_status=$?
    if [ $exit_status -ne 0 ]
    then
        fail=$((fail+1))
    fi
done
echo "Postprocess train failures: ${fail}"
python send_mail.py --event "Round $round, postprocess train failures: ${fail}" --subject "$subject"

---
time python train_mail.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.01 --crop_size 640 --batch_size 16 --output_stride 16 --train_dir /home/jovyan/work/deeplab/data/train/ --train_seg_dir /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r0_postprocess_train/ --val_dir /home/jovyan/work/deeplab/data/test/valid/ --val_seg_dir /home/jovyan/work/deeplab/data/test/valid_r_seg/ --num_classes 16 --model_name recloser_gci_r1 --mail_subject "$subject" --training_round 1 --total_epochs 200 --total_itrs 40000 --separable_conv
exit_status=$?
while [ "${exit_status}" -ne 0 ]
do
    if [ $exit_status -ne 0 ]
    then
        fail=$((fail+1))
    fi
    time python train_mail.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.01 --crop_size 640 --batch_size 16 --output_stride 16 --train_dir /home/jovyan/work/deeplab/data/train/ --train_seg_dir /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r0_postprocess_train/ --val_dir /home/jovyan/work/deeplab/data/test/valid/ --val_seg_dir /home/jovyan/work/deeplab/data/test/valid_r_seg/ --num_classes 16 --model_name recloser_gci_r1 --mail_subject "$subject" --training_round 1 --total_epochs 200 --continue_training --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16_recloser_gci_r1.pth --total_itrs 40000 --separable_conv
    exit_status=$?
    done
    echo "Training failures: ${fail}"
    temp=$round+1
    python send_mail.py --event "Round ${temp}, training failures: ${fail}" --subject "$subject"
fi
sleep 30m

# Don't forget to user --separable_conv on predict, if train was made using it!
#________________ AFTER FIRST ROUND ________________#
# echo Round $round, next is $((round+1))
for (( round=1; round<=5; round++))
do
    echo Round $round of 5
    echo Prediction train: round $((round))

    python send_mail.py --event "Prediction train: round $round" --subject "$subject"

    time python predict.py --model deeplabv3plus_mobilenet --val_batch_size 16 --input /home/jovyan/work/deeplab/data/train/ --num_classes 16 --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16_recloser_gci_r$((round)).pth --save_dir recloser_gci_r$((round))_prediction_train --separable_conv

    echo Postprocess train: round $((round))
    
    python send_mail.py --event "Postprocess train: round $round" --subject "$subject"
    
    echo Postprocess train: round $((round))
    python send_mail.py --event "Postprocess train: round $round" --subject "$subject"
    exit_status=1
    fail=0
    while [ "${exit_status}" -ne 0 ]
    do
        time python /home/jovyan/work/deeplab/utils/post_process_crf_filter_list.py --image_path /home/jovyan/work/deeplab/data/train/ --seg_path /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r$((round))_prediction_train/ --save_path /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r$((round))_postprocess_train/
        exit_status=$?
        if [ $exit_status -ne 0 ]
        then
            fail=$((fail+1))
        fi
    done
    echo "Postprocess train failures: ${fail}"
    python send_mail.py --event "Round $round, postprocess train failures: ${fail}" --subject "$subject"
      
    if [ $round -lt 6 ];
    then
        echo Training: round $((round+1))
        
        python send_mail.py --event "Training: round "$((round+1)) --subject "$subject"
        
        time python train_mail.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.01 --crop_size 640 --batch_size 16 --output_stride 16 --train_dir /home/jovyan/work/deeplab/data/train/ --train_seg_dir /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r$((round))_postprocess_train/ --val_dir /home/jovyan/work/deeplab/data/test/valid/ --val_seg_dir /home/jovyan/work/deeplab/data/test/valid_r_seg/ --num_classes 16 --model_name recloser_gci_r$((round+1)) --mail_subject "$subject" --training_round $((round+1)) --total_epochs 200 --total_itrs 40000 --separable_conv
    exit_status=$?
    while [ "${exit_status}" -ne 0 ]
    do
        if [ $exit_status -ne 0 ]
        then
            fail=$((fail+1))
        fi
        time python train_mail.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.01 --crop_size 640 --batch_size 16 --output_stride 16 --train_dir /home/jovyan/work/deeplab/data/train/ --train_seg_dir /home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/recloser_gci_r$((round))_postprocess_train/ --val_dir /home/jovyan/work/deeplab/data/test/valid/ --val_seg_dir /home/jovyan/work/deeplab/data/test/valid_r_seg/ --num_classes 16 --model_name recloser_gci_r$((round+1)) --mail_subject "$subject" --training_round $((round+1)) --total_epochs 200 --continue_training --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16_recloser_gci_r$((round+1)).pth --total_itrs 40000 --separable_conv
        exit_status=$?
    done
    echo "Training failures: ${fail}"
    temp=$round+1
    python send_mail.py --event "Round ${temp}, training failures: ${fail}" --subject "$subject"
        
    fi
    sleep 30m
done