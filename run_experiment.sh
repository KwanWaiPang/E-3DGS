#!/bin/bash
#SBATCH -p gpu22
#SBATCH --mem=64G
#SBATCH --signal=B:SIGTERM@120
#SBATCH -o ./output/.exp_main.Company.4.out
#SBATCH -t 15:00:00
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:a40:1
 
# Define the processing function
process_training() {
    local TRAIN_DIR=$1
    local DATA_DIR=$2
    shift 2

	rm -r ${TRAIN_DIR}/
    python train.py -s $DATA_DIR -m ${TRAIN_DIR} "$@"
	rm -r ${TRAIN_DIR}/test/ours_60000/renders
    python render.py -s $DATA_DIR -m ${TRAIN_DIR} --skip_train

	if [[ "$TRAIN_DIR" == *"_perturbed"* ]]; then
		rm -r ${TRAIN_DIR}/test/ours_60000/renders_optim
		python test_cam_optim.py -s $DATA_DIR -m ${TRAIN_DIR} --sh_degree 1
		python psnr.py $DATA_DIR/test/rgb/ ${TRAIN_DIR}/test/ours_60000/renders_optim
	else
		python psnr.py $DATA_DIR/validation/rgb/ ${TRAIN_DIR}/test/ours_60000/renders
	fi
}


# E3DGS-SYNTHETIC + E3DGS-SYNTHETIC-HARD DATA WITH PERTURBED POSES
for TRAIN_NAME in Company Subway ScienceLab
do
    TRAIN_NAME=${TRAIN_NAME}
    
	DATA_DIR=data/synthetic/${TRAIN_NAME}
    TRAIN_DIR=trainings/synthetic/${TRAIN_NAME}_${POSE_FOLDER}

	POSE_FOLDER=pose

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
		--max_events 1000000 --sh_degree 3 --pose_folder ${POSE_FOLDER}
    # process_training ${TRAIN_DIR}_no_isotropic_reg ${DATA_DIR} \
	# 	--max_events 1000000 --sh_degree 3 --pose_folder ${POSE_FOLDER} --lambda_isotropic_reg 0.0 
    # process_training ${TRAIN_DIR}_one_event_loss ${DATA_DIR} \
	# 	--max_events 1000000 --sh_degree 3 --pose_folder ${POSE_FOLDER} --n_event_losses 1
    # process_training ${TRAIN_DIR}_old_event_windows ${DATA_DIR} \
	# 	--max_events 1000000 --sh_degree 3 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0

   DATA_DIR=data/synthetic-hard/${TRAIN_NAME}
   TRAIN_DIR=trainings/synthetic-hard/${TRAIN_NAME}_${POSE_FOLDER}

   POSE_FOLDER=pose_perturbed

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER}
    # process_training ${TRAIN_DIR}_no_pose_reg ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_pose_reg 0.0 
    process_training ${TRAIN_DIR}_no_isotropic_reg ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_isotropic_reg 0.0 
    process_training ${TRAIN_DIR}_eventnerf_windows ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0 --n_event_losses 1
    # process_training ${TRAIN_DIR}_one_event_loss ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --n_event_losses 1
    # process_training ${TRAIN_DIR}_old_event_windows ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0
    process_training ${TRAIN_DIR}_no_pose_optim ${DATA_DIR} \
		                --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER}
    process_training ${TRAIN_DIR}_no_pose_reg_no_isotropic_reg ${DATA_DIR} \
		                --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_pose_reg 0.0 --lambda_isotropic_reg 0.0
    process_training ${TRAIN_DIR}_eventnerf_windows_no_iso ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0 --n_event_losses 1 --lambda_isotropic_reg 0.0 
    # process_training ${TRAIN_DIR}_eventnerf_windows_no_pose_reg ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0 --n_event_losses 1 --lambda_pose_reg 0.0 
done

# E3DGS-REAL
for TRAIN_NAME in shot_009 
# shot_003 shot_004 shot_005 shot_006 shot_008 shot_009 shot_011 shot_012 shot_013
do
    DATA_DIR=data/real/${TRAIN_NAME}    
	POSE_FOLDER=pose

	TRAIN_DIR=trainings/real/${TRAIN_NAME}_${POSE_FOLDER}

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
	 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER}		
    process_training ${TRAIN_DIR}_no_pose_reg_no_isotropic_reg ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_pose_reg 0.0 --lambda_isotropic_reg 0.0 
    # process_training ${TRAIN_DIR}_no_pose_reg ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_pose_reg 0.0 
    process_training ${TRAIN_DIR}_no_isotropic_reg ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --lambda_isotropic_reg 0.0 
    process_training ${TRAIN_DIR}_event_nerf_windows ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --n_event_losses 1 --adaptive_event_window 0
    # process_training ${TRAIN_DIR}_one_event_loss ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --n_event_losses 1
    # process_training ${TRAIN_DIR}_old_event_windows ${DATA_DIR} \
	# 	--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --adaptive_event_window 0
    process_training ${TRAIN_DIR}_no_pose_optim ${DATA_DIR} \
		                --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER}
    process_training ${TRAIN_DIR}_event_nerf_windows_no_iso ${DATA_DIR} \
		--pose_lr 0.001 --max_events 1000000 --sh_degree 1 --pose_folder ${POSE_FOLDER} --n_event_losses 1 --adaptive_event_window 0 --lambda_isotropic_reg 0.0 
done


# TUM_VIE
for TRAIN_NAME in mocap-1d-trans mocap-desk2
# mocap-1d-trans mocap-desk2
do
    DATA_DIR=data/tum-vie-e3dgs/${TRAIN_NAME}    
    TRAIN_DIR=trainings/tum-vie-e3dgs/${TRAIN_NAME}

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
		--max_events 1000000 --sh_degree 3 --event_threshold 0.2 --densify_grad_threshold_final 0.000008
done

# EventNeRF Synthetic
for TRAIN_NAME in chair drums ficus hotdog lego materials mic
do
    DATA_DIR=data/eventnerf_synthetic/${TRAIN_NAME}    
    TRAIN_DIR=trainings/eventnerf_synthetic/${TRAIN_NAME}

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
		--max_events 100000 --sh_degree 3 --event_threshold 0.25 --bg_color 0.62352941176470588235294117647059 --lambda_color_range 0
done

# EventNeRF Real
for TRAIN_NAME in controller bottle chicken dragon microphone multimeter plant
do
    DATA_DIR=data/eventnerf_real/${TRAIN_NAME}    
    TRAIN_DIR=trainings/eventnerf_real/${TRAIN_NAME}

    # Call the processing function with the appropriate arguments
    process_training ${TRAIN_DIR} ${DATA_DIR} \
		--max_events 100000 --sh_degree 3 --event_threshold 0.125 --bg_color 0.62352941176470588235294117647059 --lambda_color_range 0
done