#!/bin/bash
# Run best config (ent=0.05, act=1.0, smooth=0.001) on all 6 remaining objects
# Best config from sweep: phase_a_engineers_hammer_vray_20260228_031715_807134_seed42_pid16696

DATA_ROOT="/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /phase_a_yaw_only_10obj_yaw2deg_512_nodof_dist0.20_yaw60_cam0.5_look0.1"

OBJECTS=(
    "ball_peen_hammer"
    "coffeemug"
    "mug"
    "pepper"
    "scissors"
    "wooden_pepper_mill"
)

TOTAL=${#OBJECTS[@]}
COUNT=0

for OBJ in "${OBJECTS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=========================================="
    echo "  Training $OBJ ($COUNT/$TOTAL)"
    echo "=========================================="
    echo ""
    
    python /Users/kirubeso.r/Documents/PhD/keypoint_net/train.py \
        --data_root "$DATA_ROOT" \
        --object "$OBJ" \
        --epochs 1000 \
        --num_keypoints 10 \
        --frame_skip 3 \
        --lambda_disp 0.1 \
        --lambda_act 1.0 \
        --lambda_ent 0.05 \
        --lambda_smooth 0.001 \
        --num_action_classes 2 \
        --sigma 0.1 \
        --seed 42 \
        --auto_eval
    
    echo ""
    echo "  Finished $OBJ ($COUNT/$TOTAL)"
    echo ""
done

echo ""
echo "=========================================="
echo "  ALL DONE: $TOTAL objects completed"
echo "=========================================="
