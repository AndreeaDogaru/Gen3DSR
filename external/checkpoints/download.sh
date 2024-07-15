# OneFormer 
wget https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large/resolve/main/250_16_dinat_l_oneformer_ade20k_160k.pth
# Recommended depth estimator: Metric3D 
wget https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth
# Alternative depth estimation: DepthAnything 
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
# wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt
# Alternative depth estimation: Marigold 
wget https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/marigold-lcm-v1-0.tar
tar -xvf marigold-lcm-v1-0.tar
rm marigold-lcm-v1-0.tar
# Entity
wget --header="Authorization: Bearer $1" https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth
# OVSAM
wget https://huggingface.co/spaces/HarborYuan/ovsam/resolve/main/models/sam2clip_vith_rn50.pth
wget https://huggingface.co/spaces/HarborYuan/ovsam/resolve/main/models/ovsam_R50x16_lvisnorare.pth
wget https://huggingface.co/spaces/HarborYuan/ovsam/resolve/main/models/R50x16_fpn_lvis_norare_v3det.pth
wget https://huggingface.co/spaces/HarborYuan/ovsam/resolve/main/models/RN50x16_LVISV1Dataset.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Our amodal completion
wget https://huggingface.co/andreead-a/amodal-completion/resolve/main/unet/diffusion_pytorch_model.safetensors -P amodal_completion 