from huggingface_hub import hf_hub_download, snapshot_download

# 1. you should login in huggingface-cli first to access FLUX.1-dev model
# huggingface-cli login
# and Enter your token
# hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="flux1-dev.safetensors", local_dir="/cache/hanmo/models/FLUX.1-dev")

# snapshot_download(
#     repo_id="black-forest-labs/FLUX.1-dev",
#     local_dir="/cache/hanmo/models/FLUX.1-dev",
#     local_dir_use_symlinks=False,  # 不用符号链接，方便移动或压缩
#     revision="main",               # 可选，指定分支或版本号
# )

# 2. download OmniConsistency model
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="OmniConsistency.safetensors", local_dir="/cache/hanmo/models/omniconsistency/models")

# 3. download loras
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/3D_Chibi_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/American_Cartoon_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Chinese_Ink_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Clay_Toy_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Fabric_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Ghibli_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Irasutoya_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Jojo_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/LEGO_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Line_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Macaron_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Oil_Painting_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Origami_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Paper_Cutting_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Picasso_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pixel_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Poly_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pop_Art_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Rick_Morty_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Snoopy_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Van_Gogh_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")
# hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Vector_rank128_bf16.safetensors", local_dir="/cache/hanmo/models/omniconsistency/LoRAs")