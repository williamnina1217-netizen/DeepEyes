set -x

PROJECT_NAME="xhs-deepeyes"
EXPERIMENT_NAME="agent_merged_v3"

export SAVE_CHECKPOINT_DIR=/diancpfs/user/fengyuan/verl_checkpoints
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

BASEDIR=/cpfs/user/fengyuan/verl_data/minghao_data
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/minghao_data_vnew/data_v0.8_visual_toolbox_v2.parquet
VISUAL_DATASET_TEST=${BASEDIR}/seekworld_test.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_function_call.parquet
XINCE_DATASET_TRAIN=${BASEDIR}/train_xince_acc.parquet
SEEKWORLD_DATASET_TRAIN=${BASEDIR}/seekworld_train_acc.parquet

DATA_V1_THINKLITE=/cpfs/user/fengyuan/verl_data/minghao_data/thinklite_modified_250703.parquet

DATA_V2_TRAIN_0_1_2=/cpfs/user/fengyuan/verl_data/minghao_data/data_0.1.2_visual_toolbox_v2_acc_v2.parquet
DATA_V2_TRAIN_0_8_SPLIT1=/cpfs/user/fengyuan/verl_data/minghao_data/minghao_data_vnew/data_v0.8_visual_toolbox_v2_acc_split1_v2.parquet
DATA_V2_TRAIN_0_8_SPLIT2=/cpfs/user/fengyuan/verl_data/minghao_data/minghao_data_vnew/data_v0.8_visual_toolbox_v2_acc_split2_v2.parquet
DATA_V2_TRAIN_THINKLITE=/cpfs/user/fengyuan/verl_data/minghao_data/data_thinklite_reasoning_function_call_acc_v2.parquet
DATA_V2_TRAIN_XINCE=/cpfs/user/fengyuan/verl_data/minghao_data/train_xince_acc_acc_v2.parquet

DATA_TRAIN_SEEKWORLD=/cpfs/user/fengyuan/verl_data/minghao_data/seekworld_train_acc_acc_v2.parquet
DATA_TRAIN_GEOGUESSR_1=/cpfs/user/fengyuan/code/github/zero-rl-data/geoguessr/kaggle-geoguessr/kaggle-geoguessr.parquet
DATA_TRAIN_GEOGUESSR_2=/cpfs/user/fengyuan/code/github/zero-rl-data/geoguessr/deboradum-geogeussr/deboradum-geogeussr-test.parquet

DATA_TRAIN_SEEKWORLD_WITH_SEARCH=/cpfs/user/fengyuan/verl_data/minghao_data/seekworld_train_acc_acc_v2_with_search.parquet
DATA_TRAIN_BROWSECOMP=/cpfs/user/fengyuan/verl_data/minghao_data/browse_comp_xhs.parquet

# train data
DATA_TRAIN_V012=/cpfs/user/fengyuan/benchmarks/train_parquet/data_0.1.2_visual_toolbox_v2_acc_v2.parquet
DATA_TRAIN_V08_SPLIT1=/cpfs/user/fengyuan/benchmarks/train_parquet/data_v0.8_visual_toolbox_v2_acc_split1_v2.parquet
DATA_TRAIN_V08_SPLIT2=/cpfs/user/fengyuan/benchmarks/train_parquet/data_v0.8_visual_toolbox_v2_acc_split1_v2.parquet
DATA_TRAIN_THINKLITE=/cpfs/user/fengyuan/benchmarks/train_parquet/data_thinklite_reasoning_function_call_acc_v2.parquet
DATA_TRAIN_THINKLITE_NO_TOOL=/cpfs/user/fengyuan/benchmarks/train_parquet/data_thinklite_v0.parquet
DATA_TRAIN_SEEKWORLD=/cpfs/user/fengyuan/benchmarks/train_parquet/seekworld_train_acc_acc_v2.parquet
DATA_TRAIN_BROWSECOMP=/cpfs/user/fengyuan/benchmarks/train_parquet/browse_comp_xhs.parquet
DATA_TRAIN_PHYSICS=/cpfs/user/fengyuan/benchmarks/train_parquet/physreason-train.parquet
DATA_TRAIN_REVISUAL_MMRL=/cpfs/user/fengyuan/benchmarks/train_parquet/revisual_mmrl-train.parquet
DATA_TRAIN_REVISUAL_TEXTRL=/cpfs/user/fengyuan/benchmarks/train_parquet/revisual_textrl-train.parquet
DATA_TRAIN_SKYWORK_MATH=/cpfs/user/fengyuan/benchmarks/train_parquet/skywork-math-train.parquet
DATA_TRAIN_XINCE=/cpfs/user/fengyuan/benchmarks/train_parquet/xince-train.parquet

# benchmark data
DATA_TEST_VSTAR=/cpfs/user/fengyuan/benchmarks/eval_parquet/vstar-test.parquet
DATA_TEST_SEEKWORLD=/cpfs/user/fengyuan/benchmarks/eval_parquet/seekworld-test.parquet
DATA_TEST_MMSEARCH=/cpfs/user/fengyuan/benchmarks/eval_parquet/mmsearch-test.parquet
DATA_TEST_BROWSECOMP_OPENAI=/cpfs/user/fengyuan/benchmarks/eval_parquet/openai-browsecomp-test.parquet
DATA_TEST_BROWSECOMP_ZH=/cpfs/user/fengyuan/benchmarks/eval_parquet/browsecomp-zh-test.parquet
DATA_TEST_CHINESE_SIMPLEQA=/cpfs/user/fengyuan/benchmarks/eval_parquet/chinese_simpleqa-test.parquet
DATA_TEST_SIMPLEQA_OPENAI=/cpfs/user/fengyuan/benchmarks/eval_parquet/openai-simpleqa-test.parquet
DATA_TEST_SIMPLE_VQA=/cpfs/user/fengyuan/benchmarks/eval_parquet/simple-vqa-test.parquet
DATA_TEST_ZERO_BENCH=/cpfs/user/fengyuan/benchmarks/eval_parquet/zero-bench-test.parquet
DATA_TEST_OCR_REASONING=/cpfs/user/fengyuan/benchmarks/eval_parquet/ocr-reasoning-test.parquet
DATA_TEST_VISULOGIC=/cpfs/user/fengyuan/benchmarks/eval_parquet/visulogic-test.parquet
DATA_TEST_AIME=/cpfs/user/fengyuan/benchmarks/eval_parquet/aime-test.parquet

# split large eval dataset to avoid oom error
DATA_TEST_SIMPLEQA_OPENAI_SPLIT1=/cpfs/user/fengyuan/benchmarks/eval_parquet/openai-simpleqa-test_split1.parquet
DATA_TEST_SIMPLEQA_OPENAI_SPLIT2=/cpfs/user/fengyuan/benchmarks/eval_parquet/openai-simpleqa-test_split2.parquet

CUSTOM_STOP='["</tool_call>"]'
LOSS_AGG_MODE="token-mean"
export WORKING_DIR=${WORKING_DIR:-"${PWD}"}
export RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

REF_MODEL_PATH=/cpfs/user/fengyuan/backbone/qwen25/Qwen2.5-VL-7B-Instruct
PYTHONUNBUFFERED=1 python3 -m recipe.deepeyes_v2.main_dapo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${DATA_TRAIN_V012},${DATA_TRAIN_SEEKWORLD},${DATA_TRAIN_THINKLITE_NO_TOOL},${DATA_TRAIN_V08_SPLIT1},${DATA_TRAIN_V08_SPLIT2},${DATA_TRAIN_BROWSECOMP},${DATA_TRAIN_PHYSICS},${DATA_TRAIN_REVISUAL_MMRL},${DATA_TRAIN_REVISUAL_TEXTRL},${DATA_TRAIN_SKYWORK_MATH},${DATA_TRAIN_XINCE}] \
    data.val_files=[${DATA_TEST_VSTAR},${DATA_TEST_ZERO_BENCH},${DATA_TEST_MMSEARCH},${DATA_TEST_AIME}] \
    data.train_batch_size=256 \
    data.gen_batch_size=128 \
    data.max_prompt_length=20480 \
    data.max_response_length=12288 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=32 \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.threshold=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=10 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.custom_stop=${CUSTOM_STOP} \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    reward_model.reward_manager=dapo \
    reward_model.num_workers=16 \
    critic.cliprange_value=50 \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
