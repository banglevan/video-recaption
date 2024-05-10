from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (500, 500), "white")
font = ImageFont.truetype("data/arial.ttf", 40)
draw = ImageDraw.Draw(image)
position = (10, 10)
text = "Trương Huệ Vân, 36 tuổi, Tổng giám đốc Tập đoàn Vạn Thịnh Phát, sẽ bị TAND TP"

bbox = draw.textbbox(position, text, font=font)
draw.rectangle(bbox, fill="red")
draw.text(position, text, font=font, fill="black")

image.show()