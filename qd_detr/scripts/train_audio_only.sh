# train.sh에서 video 자리를 audio로 바꿔서 audio-only로 학습

dset_name=hl
ctx_mode=video_tef
v_feat_types=pann
t_feat_type=clip 
results_root=results
exp_id=audio_only

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"pann"* ]]; then
  v_feat_dirs+=(${feat_root}/pann_features)
  (( v_feat_dim += 2048 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
