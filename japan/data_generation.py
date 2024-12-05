import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import tensorflow as tf

# Dataset dimensions and variables
height, width = 32, 512
min_len, max_len = 4, 16
katakana = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
kanji = "阿伊宇江於加幾久介己散之須世曽多千川天止奈仁奴祢乃八比不部保末三牟女毛也由與良利流礼呂和乎尓"
katakana_small = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ"
chars = sorted([c for c in katakana + katakana_small + kanji])

# Character to index mapping
char_to_ind = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)
ind_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_ind.get_vocabulary(), mask_token=None, invert=True)

# Load font
font_path = "msgothic.ttc"  # Make sure to provide correct path to the font file
font = ImageFont.truetype(font_path, 32)

# Create synthetic dataset
def make_data(N_samples, min_len=4, max_len=10, add_noise=False):
    X, y = [], []
    for i in range(N_samples):
        N = np.random.randint(min_len, max_len + 1)
        text = "".join(np.random.choice(chars, N))
        n_blanks = max_len - len(text)
        x_shift = np.random.randint(0, n_blanks + 1)
        x_pos = 0 + x_shift * font.size
        y_pos = np.round((height - font.size) / 2)
        pos = (x_pos, y_pos)
        image = Image.new("RGB", [width, height], (0, 0, 0))
        d = ImageDraw.Draw(image)
        d.text(pos, text, font=font, fill=(255, 255, 255), anchor="mm")
        image = ImageOps.grayscale(image)
        image = np.array(image)
        if add_noise:
            gn = np.random.normal(loc=0, scale=np.log(2), size=image.shape)
            image = image + gn
        X.append(image.astype(np.uint8).reshape((height, width, 1)))
        text = text.strip()
        pad = (max_len - len(text)) * " "
        y.append(text + pad)
    return np.array(X), np.array(y)

# Function to encode data
def encode_data(image, label):
    image = tf.keras.layers.Rescaling(1.0 / 255)(image)
    image = tf.transpose(image, perm=[1, 0, 2])
    label = char_to_ind(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": image, "label": label}
