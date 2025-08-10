dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
a_feat_type=pann
results_root=results
gating_type=global # 'global' or 'clipwise' or 'elementwise'
input_dropout=0.5
seed=2018
fusion_layers=4
fusion_n_heads=8
exp_id=avigate_custom_${gating_type}_${input_dropout}_${seed}_l${fusion_layers}_h${fusion_n_heads}

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# audio features
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/pann_features/
  a_feat_dim=2050
else
  echo "Wrong arg for a_feat_type."
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
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--use_avigate_custom \
--gating_type ${gating_type} \
--input_dropout ${input_dropout} \
--seed ${seed} \
--fusion_layers ${fusion_layers} \
--fusion_n_heads ${fusion_n_heads} \
"${@:1}"