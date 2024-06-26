# A GUIDE TO USE FINETUNE VIDEOLLAMA ON CUSTOMIZED DATASETS

This document walks you through 3 steps to create a vision language model that is capable of generating suitable low-level modes (in controlling robots), taking into account the visual context. For example, being able to distinguish the movement of one's hand to-and-fro as in greeting or as in cleaning a glass door:
- preparing the instruction dataset (for the task of human action recognition)
- configuring the finetuning experiments
- launching the experiments

Explanations on model details (the components, the steps in creating the model, the setting up of the environment) are adaptated from the main [README.md](README.md) file.

## Preface - the model
- Video-LLaMA is an Audio-Visual Language Model for video understanding.
- The authors published pretrained models and instruction-finetuned models:
  - Pretrained models (`Video-LLaMA-2-7B-Pretrained` and `Video-LLaMA-2-13B-Pretrained`): Pretrained on Webvid-2M video caption and LLaVA's image caption datasets for video-to-text generation task and static visual understanding capabilities.
  - Instruction-finetuned models (`Video-LLaMA-2-7B-Finetuned` and `Video-LLaMA-2-13B-Finetuned`): instruction-finetuned on specific datasets, be it image/video caption datasets.
- Video-LLaMA comprises of two components: Vision-Language (VL) branch and Audio-Language (AL) branch.
  - Each branch is pretrained/finetuned separately on video/image datasets (note: no audio data is needed even for the AL branch).
  - Trainable layers are: Video Q-Former and Audio Q-Former (for computing video/audio representations), positional embedding layers, and linear layers.

## Setting up
- Install `ffmpeg` as instructed in [README.md](README.md):
  ```
  apt update
  apt install ffmpeg
  ```
- Create a virtual environment and conda install all packages in `environment.yml`:
  ```
  conda env create -f environment.yml
  conda activate videollama
  ```
  - might need to reinstall `torchaudio`: `pip install -U torch torchaudio --no-cache-dir` if encounter `OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory` ([ref](https://github.com/pytorch/pytorch/issues/91186#issuecomment-1791766605))

## Finetune Video-LLaMA on instruction video dataset (action recognition)

### Data preparation
We construct a simple instruction dataset from the [video dataset about human actions](https://www.kaggle.com/datasets/ngoduy/dataset-video-for-human-action-recognition) (this dataset divides videos into different folders, each corresponds to a certain action), which comprises of 2 main parts:
  - The videos: video inputs to VideoLLaMA.
  - The instructions: instruction text sequences (based on which folder a video belongs to).

Therefore, the framework requires an annotation json file that stores mappings of the videos' respective question and answer pairs. An example of an entry in the annotation file is as below:
  ```json
  {
    "video": "train_fall_down_video_127.avi",
    "QA": [
        {
            "q": "What is the action shown in the displayed video?",
            "a": "Fall down"
        }
    ]
  }
  ```
  , where:
  - `"video"`: path to the input video,
  - `"QA"`: list of question-answer pairs about the video (when there's more than 1 pair, this will be considered a conversation about the video, which we don't necessarily need in our case):
    - `"q"`: the question used in the instruction finetuning paradigm, simply a question asking what the action seem to be in the video.
    - `"a"`: simply the corresponding label of the action shown in the video.

### Configure the experiment
- We finetune Video-LLaMA visual branch on the prepared dataset, which can be done by simply updating the configuration file [visionbranch_stage2_finetune.yaml](train_configs/visionbranch_stage2_finetune.yaml). Here we configure the experiments in terms of the model, the datasets to use, and the running environment:
  ```yml
  model:
    arch: video_llama
    model_type: pretrain_llama_v2

    # freeze the visual encoder (ViT and Q-Former)
    freeze_vit: True
    freeze_qformer: True

    # Q-Former
    num_query_token: 32

    # If you want train models based on LLaMA-2-chat,
    # some ckpts could be download from our provided huggingface repo
    # i.e.  https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned
    llama_model: "meta-llama/Llama-2-7b-chat-hf" #"ckpt/vicuna-13b/" or "ckpt/vicuna-7b/" or "ckpt/llama-2-7b-chat-hf"  or "ckpt/llama-2-13b-chat-hf"

    # The ckpt of vision branch after stage1 pretrained, 
    ckpt: 'ckpts/VL_LLaMA_2_7B_Pretrained.pth'   # you can use our pretrained ckpt from https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained/

    # only train vision branch
    equip_audio_branch: False  # whether equips the audio branch
    frozen_llama_proj: False
    frozen_video_Qformer: False
    frozen_audio_Qformer: True

    fusion_head_layers: 2
    max_frame_pos: 32
    fusion_header_type: "seqTransf"

    max_txt_len: 320

    # vicuna and llama_2_chat use different template !!!

    # for llama_2_chat:
    end_sym: "</s>"
    prompt_path: "prompts/alignment_image.txt"
    prompt_template: '[INST] <<SYS>>\n \n<</SYS>>\n\n{} [/INST] '

    # # for vicuna:
    # end_sym: "###"
    # prompt_path: "prompts/alignment_image.txt"
    # prompt_template: '###Human: {} ###Assistant: '

  datasets:
    webvid_instruct:
      data_type: video
      build_info:
        anno_dir: dataset_action_split/videoaction_instruct_11k.json
        videos_dir: dataset_action_split/data/
      vis_processor:
        train:
          name: "alpro_video_train"
          n_frms: 8
          image_size: 224
      text_processor:
        train:
          name: "blip_caption"
      num_video_query_token: 32
      tokenizer_name: "meta-llama/Llama-2-7b-chat-hf" # "ckpt/vicuna-13b/" or "ckpt/vicuna-7b/" or "ckpt/llama-2-7b-chat-hf"  or "ckpt/llama-2-13b-chat-hf"
      model_type: "llama_v2" # or "vicuna"  # need to set, as vicuna and llama_2_chat use different template

  run:
    task: video_text_pretrain
    # optimizer
    lr_sched: "linear_warmup_cosine_lr"
    init_lr: 3e-5
    min_lr: 1e-5
    warmup_lr: 1e-6

    weight_decay: 0.05
    max_epoch: 3
    iters_per_epoch: 1000
    batch_size_train: 4
    batch_size_eval: 4
    num_workers: 4
    warmup_steps: 1000

    seed: 42
    output_dir: "output/videollama_stage2_finetune"

    amp: True
    resume_ckpt_path: null

    evaluate: False 
    train_splits: ["train"]

    device: "cuda"
    world_size: 4
    dist_url: "env://"
    distributed: True
  ```
  - for `model` configuration, important attributes are:
    - `arch: video_llama`: the model used is `VideoLLAMA` (registered as `video_llama`),
    - `llama_model`: the huggingface path to the language decoder model to use, either `llama` or `vicuna`,
    - `model_type`: either `pretrain_llama_v2` or `pretrain_vicuna`,
    - `ckpt`: path to the checkpoint of the pretrained vision branch model (download the pretrained **visual branch** of Video-LLaMA model ([7B](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained) or [13B](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained)), and pass the path to `ckpt`, e.g. `./ckpts/VL_LLaMA_2_13B_Pretrained.pth`.),
    - `end_sym`, `prompt_path`, `prompt_template`: prompt template for either llama 2 or vicuna model,
  - for `datasets` configuration, we need to specify the type of dataset builders to use, which in our case, we just need to build an instruction dataset for video, aka `webvid_instruct`. Important attributes to pass to a `webvid_instruct` builder are:
    - `build_info.anno_dir`: the path to the video annotation json file prepared in the Data Preparation step
    - `build_info.video_dif`: the path to the directory that stores all videos prepared in the Data Preparation step
    - `vis_processor.train.n_frms`: the number of frames to take from each video, note that this number should be smaller than the number of frames of the shortest video
    - `tokenizer_name`: the text tokenizer corresponding to the language decoder model
    - `model_type`: the language decoder model type, (`llama` or `vicuna`)
  - for `run` configuration, we specify the task and other related experiment related parameters (). important attributes are:
    - `task`: the task to train VideoLLaMA on, either `image_text_pretrain` or `video_text_pretrain`
    - `iters_per_epoch`: number of iteration per an epoch, which can be set to `num_data / (batch_size * num_workers)`
    - other training hyperparameters: lr scheduler, lr, seed, weight decay, etc.
    - other parameters for distributed experiments: `world_size` aka the number of parallel processes to run, `dist_url` specifies the rendezvous environment (`"env://"` refers to the virtual environment, so env variables related to the distributed experiments are stored in the virtual environment), `distributed` to enable/disable this.

### Launch the experiment
- When distributed enabled, we can use torchrun elastic to run `train.py` as a parallel experiment with `DistributedDataParallel`:
  - single node experiment with 2 GPUs:
    ```
    torchrun --standalone --nproc_per_node=2 train.py --cfg-path  ./train_configs/visionbranch_stage2_finetune.yaml
    ```
  - 2 node experiment with 2 GPUs, launched with SLURM job scheduler:
    ```
    srun torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29603 train.py --cfg-path  ./train_configs/visionbranch_stage2_finetune.yaml
    ```
## Inference

### Configure the inference pipeline
After finetuning Video-LLaMA visual branch on the prepared dataset, configure the inference pipeline for use with `demo_video.py` that launches a `gradio` server for hosting the finetuned model. The inference configuration [video_llama_eval_only_vl.yaml](eval_configs/video_llama_eval_only_vl.yaml) is as follows:
  ```yml
  model:
    arch: video_llama
    model_type: pretrain_llama_v2
    freeze_vit: True
    freeze_qformer: True
    max_txt_len: 512
    end_sym: "###"
    low_resource: False

    frozen_llama_proj: False

    # If you want use LLaMA-2-chat,
    # some ckpts could be download from our provided huggingface repo
    # i.e.  https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned
    llama_model: "meta-llama/Llama-2-7b-chat-hf"
    ckpt: 'video_llama/output/videollama_stage2_finetune/20240531153/checkpoint_0.pth'   # you can use our pretrained ckpt from https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained/
    equip_audio_branch: False

    fusion_head_layers: 2
    max_frame_pos: 32
    fusion_header_type: "seqTransf"

  datasets:
    webvid:
      vis_processor:
        train:
          name: "alpro_video_eval"
          n_frms: 8
          image_size: 224
      text_processor:
        train:
          name: "blip_caption"

  run:
    task: video_text_pretrain

  ```
  , which largely resembles the finetuning configuration above:
  - `model` configuration: use the path to the newly finetuned model, `"video_llama/output/videollama_stage2_finetune/20240531153/checkpoint_0.pth"`, for `ckpt`
  - `datasets` configuration: use `webvid` as the dataset builder for inference

Launch the inference pipeline:
  ```
  python demo_video.py --cfg-path eval_configs/video_llama_eval_only_vl.yaml --model_type llama_v2 --gpu-id 0
  ```
  , the model will be running on GPU device 0. `Gradio` will host the chat service and return local URL for the service (to get public URLs, set `share=True` at `demo.launch()` in the [demo_video.py](demo_video.py))

Below are some demo videos of the finetuned VideoLLaMA on test data, note how the input text from the user is `"What is the action shown in the displayed video?"`, which is the instruction used for finetuning the model:
<p align="center" width="100%">
<img src="figs/demo-falldown.gif" controls preload></video> </br> </br>
<img src="figs/demo-walking.gif" controls preload></video>
</p>

## Next steps
- This guide focuses on a sample task of action recognition, using the public dataset for human action detection. However, these videos are very short. Further augmentation to extend the length of these videos or using a better dataset can be considered.
- We can prompt engineer `"q"` and `"a"` a way that allows the optimal training of the model, e.g.:
  - `"q"`: use varied questions instead of a simple question `"What is the action shown in the displayed video?"`,
  - `"a"`: use better explained answers (e.g. explain the movement in human language and conclude the label, which requires more effort in augmenting the answer set whether with crowdsourcing or with AI).