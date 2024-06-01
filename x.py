from transformers import AutoTokenizer, AutoModel
from PIL import Image

# 모델과 토크나이저를 다운로드합니다.
model = AutoModel.from_pretrained("mustafaaljadery/llama3v").cuda()
tokenizer = AutoTokenizer.from_pretrained("mustafaaljadery/llama3v")

# 이미지를 엽니다.
image = Image.open("test_image.png")

# 모델을 사용하여 이미지에 대한 질문을 생성합니다.
answer = model.generate(image=image, message="What is this image?", temperature=0.1, tokenizer=tokenizer)

print(answer)