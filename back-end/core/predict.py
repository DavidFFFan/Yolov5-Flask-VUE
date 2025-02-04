import cv2

def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    x = cv2.imread(x)
    img_y, image_info = model.generateone(x)
    print(f"fixed-img shape{img_y.shape}, fixed-img dtype {img_y.dtype}")    
    if not cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y):
        raise Exception('保存图片时出错.Error saving thepicture.')
    print(image_info)
    return image_info
