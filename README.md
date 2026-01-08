CUDA_VISIBLE_DEVICES=0 python generate.py
    --ref_path /root/float/camila.png \
    --aud_path path/to/audio \
    --seed 15 \
    --a_cfg_scale 2 \
    --e_cfg_scale 1 \
    --ckpt_path ./checkpoints/float.pth
    --no_crop                  
这是杨在10/09/2025年更新的最新训练camila 的方式
首先 train一个stylegan
nohup python trainstylegan916.py \
    --dataset /home/ubuntu/float/results/data2 \
    --epochs 3000 \
    --log_resolution 7 \
    > train.log 2>&1 &


然后在/home/ubuntu/float/ 文件夹里放进1.mp4 2.mp4 。。。。。最好来个1000个的

大概1天左右 然后 使用feature extraction 脚本process_mp4s.sh 直接运行（注意要安装ffmpeg）
注意用这个extract的audiofeature 只是最后一层所以要用

python extract_all_layers.py \
  --root /home/ubuntu/float \
  --start 1 --end 1 \
  --ckpt /home/ubuntu/float/wav2vec2_fairseq_base_ls960.pth \
  --outdir_name audiofeatures_alllayers \
  --overwrite
来得到所有的隐藏层的feature 这里audiofeatures_alllayers的npy 的shape 应该是 50 12（12层隐藏层） 764

然后最费时的过程来了 
python batch_invert_wplus_sbs_video.py --ckpt /home/ubuntu/float/results/checkpoints/epoch_10001.pt（你stygan的模型） --input_dir /home/ubuntu/float/48/images --max_images 6000 --out_mp4 sidebyside2000.mp4 --fps 25 --steps 500 --w_dir /home/ubuntu/float/48/wplus --frames_dir /home/ubuntu/float/48/w_frames
这里面是48 实际上要把所有mp4 都要来一遍
然后smooth wplus：
python -u smooth_wplus_win3.py   --input_dir /home/ubuntu/float/1/wplus   --output_dir /home/ubuntu/float/1/wplus_smooth

或者想批量：
/home/ubuntu/float# for i in {7..19}; do
  echo "Processing vid=$i ..."
  python -u smooth_wplus_win3.py \
    --input_dir /home/ubuntu/float/$i/wplus \
    --output_dir /home/ubuntu/float/$i/wplus_smooth
done
########################################################################################################################
以上是训练的准备工作 下面是训练部分 之后可以修改框架：
python -u train_wplus_v_multivid.py --root /home/ubuntu/float --vid 1 2 3 4 5 --epochs 1200 --batch_size 8 --amp

检查模型和real value作比较：
python infer_and_plot_direct_ar2.py   --root /home/ubuntu/float   --vid 50   --ckpt /home/ubuntu/float/awt_alllayers_w.pt   --start_audio_idx 2   --end_audio_idx 2   --amp
50 是视频号 start_audio_idx 2   --end_audio_idx 2可以改 当这两个号一样的时候（即只生成25frame的时候）可以检测模型的输入是否正确

#########################################################################################################################
python train_wplus_direct_v4.py   --root /home/ubuntu/float --vids 1,2,3,4,5   --wplus_dirname wplus_smooth --audio_dirname audiofeatures_alllayers   --epochs 4000 --batch_size 32 --lr 2e-4 --amp   --prev_pool firstn --prev_first_n 2 --prev_use_backbone_hidden   --audio_halve_to_25   --lambda_pred 1.0 --lambda_cons 1.0   --lambda_bound56 1.0 --delta56_max 0.05 --delta
56_metric l1mean   --save /home/ubuntu/float/50/ckpts/train_wplus_direct_v4.pt

python infer_and_plot_direct_ar2_v4.py   --root /home/ubuntu/float --vid 5   --ckpt /home/ubuntu/float/50/ckpts/train_wplus_direct_v4.pt   --wplus_dirname wplus_smooth   --audio_dirname audiofeatures_alllayers   --out_dirname wplus_generate_direct_v4   --start_audio_idx 2 --end_audio_idx 2   --audio_halve_to_25   --amp   --d_model 640 --nhead 8 --layers 6 --ff 2048 --dropout 0.1   --prev_head_hidden 512 --prev_pool firstn --prev_first_n 2
%%%%%%%%%%
/home/ubuntu/float# python train_wplus_direct_v4.py --root /home/ubuntu/float --vids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50 --wplus_dirname wplus_smooth --audio_dirname audiofeatures_alllayers --epochs 400 --batch_size 8 --lr 2e-4 --amp --prev_pool firstn --prev_first_n 2 --prev_use_backbone_hidden --audio_halve_to_25 --lambda_pred 1.0 --lambda_cons 1.0 --lambda_bound56 1.0 --delta56_max 0.15 --delta56_metric l1mean --save /home/ubuntu/float/50/ckpts/train_wplus_direct_v4.pt --val_holdout 2


/home/ubuntu/float# python infer_and_plot_direct_ar2_v4.py   --root /home/ubuntu/float --vid 31   --ckpt /home/ubuntu/float/50/ckpts/train_wplus_direct_v4.pt   --wplus_dirname wplus_smooth   --audio_dirname audiofeatures_alllayers   --out_dirname wplus_generate_direct_v4   --start_audio_idx 42 --end_audio_idx 60   --audio_halve_to_25   --amp   --d_model 640 --nhead 8 --layers 6 --ff 2048 --dropout 0.1   --prev_head_hidden 512 --prev_pool firstn --prev_first_n 2

##############
python extract_all_layers.py   --root /home/ubuntu/float   --start 335 --end 350   --ckpt /home/ubuntu/float/wav2vec2_fairseq_base_ls960.pth   --outdir_name audiofeatures_alllayers   --overwrite

tar -xzvf 435_450.tar.gz

for i in {435..450}; do   echo "Processing vid=$i ...";   python -u smooth_wplus_win3.py     --input_dir /home/ubuntu/float/$i/wplus     --output_dir /home/ubuntu/float/$i/wplus_smooth; done

 python train_wplus_direct_v4.py --root /home/ubuntu/float --vids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450 --wplus_dirname wplus_smooth --audio_dirname audiofeatures_alllayers --epochs 4000 --batch_size 8 --lr 2e-4 --amp --prev_pool firstn --prev_first_n 2 --prev_use_backbone_hidden --audio_halve_to_25 --lambda_pred 1.0 --lambda_cons 1.0 --lambda_bound56 1.0 --delta56_max 0.15 --delta56_metric l1mean --save /home/ubuntu/float/50/ckpts/train_wplus_direct_v4.pt --val_holdout 2

 python train_wplus_direct_v44.py   --root /home/ubuntu/float --vids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,334,333,332,331,330,329,328,327,326,325,324,323,322,321,320,319,434,433,432,431,430,429,428,427,426,425,424,423,422,421,420,419  --wplus_dirname wplus_smooth --audio_dirname audiofeatures_alllayers   --audio_halve_to_25   --epochs 400 --batch_size 64 --lr 2e-4 --amp   --prev_pool firstn --prev_first_n 2 --prev_use_backbone_hidden   --lambda_pred 1.0 --lambda_cons 1.0   --lambda_bound56 1.0 --delta56_max 0.15 --delta56_metric l1mean   --norm_vids 7-45   --save /home/ubuntu/float/50/ckpts/train_wplus_direct_v4_norm.pt --val_holdout 2











find /home/ubuntu/float -type d -name "__pycache__" -exec rm -rf {} +




python3 mp4_to_all12_1s_feats.py   --mp4 /home/ubuntu/float/50.mp4   --out_dir /home/float/50/audiofeatures_alllayers_1s   --model facebook/wav2vec2-base-960h   --sr 16000   --frames_per_sec 50   --device cuda   --skip_last_partial


 prep_audio_align_25.py   --root /home/ubuntu/float   --vids 50   --src_dirname audiofeatures_alllayers_1s   --dst_dirname audio_aligned

python3 /home/ubuntu/float/train_wplus_fmt.py   --root /home/ubuntu/float   --vids 50   --wplus_dirname wplus_smooth   --audio_dirname audio_aligned   --fps 25 --K 5 --L 25   --w_dim 256 --dim_a 9216   --bs 16 --lr 2e-4 --epochs 400   --device cuda   --save /home/ubuntu/float/50/ckpts/wplus_fmt_all12_256.pt   --debug_print


python stylegan2_from_pdf.py   --dataset /home/ubuntu/float/results/data2   --epochs 300   --batch_size 128   --log_resolution 7   --z_dim 256   --w_dim 256   --lr 1e-4

#############################################################################################
train判别器网络11.5：python /home/ubuntu/float/train_syncnet_wplus_audio_v2.py   --vid $(seq -s, 5 30)   --epochs 200 --batch_size 16 --emb_M 32 --lr 1e-3   --sr 16000 --win_length 1024 --hop_length 640   --standardize_wplus   --loss_type margin --margin 0.2   --near_neg_prob 0.7 --near_neg_max_offset 2
