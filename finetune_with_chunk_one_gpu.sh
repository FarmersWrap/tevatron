rm -rf retriever-one-gpu-chunk-128
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.train \
  --output_dir retriever-one-gpu-chunk-128 \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --do_train \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --dataset_name Tevatron/scifact \
  --dataset_split train \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --padding_side right \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 4 \
  --train_group_size 4 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --passage_chunk_size 128 \
  --num_train_epochs 1 \
  --logging_steps 20 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --dataloader_drop_last true \
  --attn_implementation sdpa


output_dir=/root/autodl-tmp/tevatron/retriever-one-gpu-chunk-128
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --lora_name_or_path /root/autodl-tmp/tevatron/retriever-one-gpu-chunk-128\
  --bf16 \
  --per_device_eval_batch_size 32 \
  --normalize \
  --pooling last \
  --padding_side right \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --query_max_len 32 \
  --dataset_name Tevatron/beir \
  --dataset_config scifact \
  --dataset_split test \
  --encode_output_path ${output_dir}/queries_scifact.pkl \
  --encode_is_query \
  --attn_implementation sdpa


# Encode corpus
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --lora_name_or_path /root/autodl-tmp/tevatron/retriever-one-gpu-chunk-128\
  --bf16 \
  --per_device_eval_batch_size 32 \
  --normalize \
  --pooling last \
  --padding_side right \
  --passage_prefix "" \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config scifact \
  --dataset_split train \
  --encode_output_path ${output_dir}/corpus_scifact.pkl \
  --passage_chunk_size 128 \
  --attn_implementation sdpa

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_scifact.pkl \
    --passage_reps ${output_dir}/corpus_scifact.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.scifact.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.scifact.txt \
                                                       --output ${output_dir}/rank.scifact.trec \
                                                       --remove_query

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_scifact.pkl \
    --passage_reps ${output_dir}/corpus_scifact.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.scifact.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.scifact.txt \
                                                       --output ${output_dir}/rank.scifact.trec \
                                                       --remove_query
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test ${output_dir}/rank.scifact.trec

# recall_100              all     0.9400
# ndcg_cut_10             all     0.6554

