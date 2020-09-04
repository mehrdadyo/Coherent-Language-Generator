# modified_GPT2
The original implementation and pretrained weights come from Huggingface Transformers

GPT-2 Language Modeling Steps:
1. Install huggingface transformers (https://huggingface.co/transformers/installation.html)
2. run the following command in the current directory:
        python ./run_generation.py --model_type=gpt2 --model_name_or_path=gpt2 --length 1600 --seed 100
   if there is a line in the console displaying "************ THIS MODEL COMES FROM CS224N PROJECT ************", it means this model works on your computer
   You can try entering a prompt followed by <Enter> to generate a related story
      
      
      
Fine-training a combined NSP and Casual Language Modeling objective:
1. Download Wikipedia dataset: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
2. Set environment variable: 
        export TRAIN_FILE=/path/to/dataset/wiki.train.raw 
        export TEST_FILE=/path/to/dataset/wiki.test.raw
3. If using sentence embedding approach: Run the following command: python ./run_language_modeling_sentence_embed.py --output_dir=output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=$TRAIN_FILE --do_eval --eval_data_file=$TEST_FILE --per_gpu_train_batch_size=2 --save_total_limit=1 --overwrite_output_dir --num_train_epochs=2 --warmup_steps=300 --per_gpu_eval_batch_size=2 --evaluate_during_training --fp16 --fp16_opt_level='O2'

   If using cross attention approach: Run the following command: python ./run_language_modeling_cross_attention.py --output_dir=output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=$TRAIN_FILE --do_eval --eval_data_file=$TEST_FILE --fp16 --fp16_opt_level=O2 --per_gpu_train_batch_size=1 --save_total_limit=1 --overwrite_output_dir --evaluate_during_training --num_train_epochs=20 --warmup_steps=100
   
Run NSP and LM evaluation using saved parameters:
python ./run_language_modeling_sentence_embed.py --output_dir=output --model_type=gpt2 --model_name_or_path=./output --train_data_file=$TRAIN_FILE --do_eval --eval_data_file=$TEST_FILE --per_gpu_train_batch_size=2 --save_total_limit=1 --overwrite_output_dir --num_train_epochs=4 --warmup_steps=300 --per_gpu_eval_batch_size=2 --evaluate_during_training --fp16 --fp16_opt_level='O2' --DP_classifier_dir=./output/DP_classifier.bin


Train forced-topic:
python ./forced_topic.py --output_dir=output_ft --model_type=gpt2 --model_name_or_path=./output --do_train --train_data_file=$TRAIN_FILE --do_eval --eval_data_file=$TEST_FILE --per_gpu_train_batch_size=4 --save_total_limit=1 --overwrite_output_dir --num_train_epochs=4 --warmup_steps=300 --per_gpu_eval_batch_size=4 --evaluate_during_training --fp16 --fp16_opt_level='O2' --DP_classifier_dir=./output/DP_classifier.bin

Run forced-topic conditional language model:
python ./run_generation_forced_topic.py --model_type=gpt2 --model_name_or_path=./output_ft/ --length 1600 --seed 100
