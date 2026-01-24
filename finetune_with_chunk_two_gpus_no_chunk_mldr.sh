rm -rf retriever-two-gpu-no-chunk-mldr
TEVATRON_DEBUG_DATALOAD=1 deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-two-gpu-no-chunk-mldr \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --do_train \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 5000 \
  --dataset_name Shitao/MLDR \
  --dataset_config en \
  --dataset_split train \
  --query_prefix "Instruct: Given a question, retrieve documents that answer the question.\nQuery:" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --padding_side right \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 4 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 4096 \
  --num_train_epochs 1 \
  --logging_steps 20 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --dataloader_drop_last true \
  --seed 42 \
  --attn_implementation sdpa


output_dir=retriever-two-gpu-no-chunk-mldr
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --lora_name_or_path retriever-two-gpu-no-chunk-mldr \
  --bf16 \
  --per_device_eval_batch_size 32 \
  --normalize \
  --pooling last \
  --padding_side right \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --query_max_len 32 \
  --dataset_name json \
  --dataset_path data/queries.jsonl \
  --dataset_split train \
  --encode_output_path ${output_dir}/queries_mldr.pkl \
  --encode_is_query \
  --attn_implementation sdpa


# # Encode corpus
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --lora_name_or_path retriever-two-gpu-no-chunk-mldr \
  --bf16 \
  --per_device_eval_batch_size 32 \
  --normalize \
  --pooling last \
  --padding_side right \
  --passage_prefix "" \
  --passage_max_len 4096 \
  --dataset_name json \
  --dataset_path data/corpus.jsonl \
  --dataset_split train \
  --encode_output_path ${output_dir}/corpus_mldr.pkl \
  --attn_implementation sdpa

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_mldr.pkl \
    --passage_reps ${output_dir}/corpus_mldr.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.mldr.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.mldr.txt \
                                                       --output ${output_dir}/rank.mldr.trec \
                                                       --remove_query

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_mldr.pkl \
    --passage_reps ${output_dir}/corpus_mldr.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.mldr.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.mldr.txt \
                                                       --output ${output_dir}/rank.mldr.trec \
                                                       --remove_query
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 data/qrels.txt ${output_dir}/rank.mldr.trec


# recall_100              all     0.8883
# ndcg_cut_10             all     0.5899
