import glob
import PIL.Image
import PIL.ImageDraw
import numpy as np
import tqdm
import os


############
# Divide image into BxB blocks.
B = 4
input_dir = 'sample_input_images'
output_dir = 'sample_images_highlighted_blocks'
############
print ("Input directory: {}".format(input_dir))
print ("Saving output to: {}".format(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#based_on: https://stackoverflow.com/questions/34255938/is-there-a-way-to-specify-the-width-of-a-rectangle-in-pil
def drawrect(drawcontext, xy, outline=None, width=0, draw_buf=None, shape=(480, 640)):
    (x1, y1), (x2, y2) = xy
    if draw_buf is None:
        draw_buf = (width // 2) + 1
    x1 = max(0, x1 - draw_buf)
    y1 = max(0, y1 - draw_buf)
    x2 = min(shape[0], x2 + draw_buf)
    y2 = min(shape[1], y2 + draw_buf)
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

files = glob.glob(os.path.join(input_dir, "*"))

for file in tqdm.tqdm(files):
    print ("  Working on: {}...".format(file))
    img = PIL.Image.open(file)
    block_x_step = int(img.size[1] / float(B))
    block_y_step = int(img.size[0] / float(B))
    for i in range(0, img.size[1]  - block_x_step+1, block_x_step):
        for j in range(0, img.size[0]  - block_y_step+1, block_y_step):
            orig_img = PIL.Image.open(file).convert("RGB")
            orig_img = np.array(orig_img, dtype=np.float32)
            orig_img[...,:] /= 1.5
            orig_img[max(0, i-3):i+block_x_step+3, max(0, j-3):j+block_y_step+3, :] *= 1.5
            orig_img = orig_img.astype(np.uint8)
            #orig_img = PIL.Image.fromarray(orig_img).convert("RGB")
            orig_img = PIL.Image.fromarray(orig_img)

            draw = PIL.ImageDraw.Draw(orig_img)
            drawrect(draw, [(j,i),(j+block_y_step, i+block_x_step)], outline=(0,255,0), width=5, shape=orig_img.size)
            del draw

            ext = os.path.splitext(file)[1]
            output_filename = file.replace(ext, "_highlighted_block_drawbuf_{}-{}{}".format(i / block_x_step,j / block_y_step, ext))
            output_filename = os.path.join(output_dir, os.path.basename(output_filename))

            orig_img.save(output_filename)

