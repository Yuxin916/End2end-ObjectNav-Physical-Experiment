import torch
from transformers import AutoProcessor
# 具体类名看你本地 import 路径：可能是 transformers 里的，也可能是你 repo 里的
from transformers import Qwen3VLForConditionalGeneration  # 或者 from your_repo import ...

MODEL_DIR = "/home/all/muyi/CL_CoTNav/Qwen3-VL/output/RGB_HisKFSingleColor_100K"
# MODEL_DIR = "/home/all/muyi/CL_CoTNav/Qwen3-VL/pretrained/Qwen3-VL-2B-Instruct" 

device = "cuda"
dtype = torch.bfloat16  # 没有 bf16 就用 torch.float16

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=False)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto",          # 单卡也可以 device_map=None 然后 .to(device)
    trust_remote_code=False
).eval()

# 你要评测的图片和文本
image = ["/home/all/muyi/CL_CoTNav/training_data/test_file/016.jpg", 
         "/home/all/muyi/CL_CoTNav/training_data/test_file/000.jpg",
         "/home/all/muyi/CL_CoTNav/training_data/test_file/black.jpg",
         "/home/all/muyi/CL_CoTNav/training_data/test_file/black.jpg",
         "/home/all/muyi/CL_CoTNav/training_data/test_file/black.jpg",
         "/home/all/muyi/CL_CoTNav/training_data/test_file/black.jpg",
         ]  
# prompt = "do you think this image is very hard to see?"
instruction = "Walk into the closet and turn left, then stop and wait by the poster on the wall."
# instruction = "you should stop now! STOP! TURN LEFT"
prompt = f"Imagine you are an autonomous robot in an indoor environment for instruction following task. You should follow the instruction and then predict a navigable goal pixel ratio in the image. You should output a X and Y with the range from 0 to 999, where [15, 471], X > 970 represents right turns, exactly [985, 471], and Y > 940 represents STOP, exactly (500, 999).\nInstruction: {instruction} \n\nInput: - Current Step egocentric RGB image <image> , where the left/right gray padding indicates left/right turns, and the bottom padding indicates stop. \n- History Images: Previous 5 egocentric RGB keyframes from the past trajectory, with blue trajectory showing movement trajectory: <image>(latest), <image>, <image>, <image>, <image>(oldest)\n\nOutput: Output: a pixel coordinates (X, Y) - Example: 123, 456."


# Qwen 系列一般推荐用 chat template 组织多模态消息
# 按 <image> 占位符拆分 prompt，把每张图插到对应位置
parts = prompt.split("<image>")
content = []
for i, part in enumerate(parts):
    if part:  # 非空文本段
        content.append({"type": "text", "text": part})
    if i < len(parts) - 1:  # 每个 split 间隙对应一张图
        content.append({"type": "image", "image": image[i]})

messages = [
    {
        "role": "user",
        "content": content,
    }
]


# messages = [
#     {
#         "role": "user",
#         "content": [{"type": "text", "text": prompt},
#                     {type: "image", "image": image[0]},
#                     ],
#     }
# ]

# 1) 把 messages 变成模型可读的 text
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("TEXT HERE: ", text)

# 2) processor 同时把 text + image 变成 input_ids/pixel_values/image_grid_thw/attention_mask 等
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    do_resize=True
).to(model.device)


# 3) generate
with torch.inference_mode():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

# 4) 解码（把 prompt 部分截掉更干净）
gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
pred = processor.decode(gen_ids, skip_special_tokens=True)
print(pred)



