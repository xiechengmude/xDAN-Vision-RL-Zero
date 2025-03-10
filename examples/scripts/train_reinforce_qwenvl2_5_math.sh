# wget https://raw.githubusercontent.com/TideDra/lmm-r1/refs/heads/main/examples/data/mathlv345_8k_chatml.json
DATASET="/openrlhf/examples/test_scripts/mathlv345_8k_chatml.json"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url /openrlhf/examples/scripts/math_verifier.py \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.8 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --deepspeed_enable_sleep \
   --pretrain Qwen/Qwen2.5-VL-3B-Instruct \
   --save_path /openrlhf/examples/test_scripts/final_ckpt/qwen2_5vl_3b \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0 \
   --prompt_data $DATASET \
   --input_key prompt \
   --label_key answer \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/qwen2_5vl_3b \
   --save_hf_ckpt

   # for visual dataset
   # --train_vlm
