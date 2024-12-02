from ultralytics import YOLO
import cv2
import numpy as np
import torch
import models.RRDBNet_arch as arch
import easyocr

upscalerModel_path = 'models/RRDB_ESRGAN_x4.pth'
licensePlateDetector = YOLO('models/LP.pt')
imagePath = "./testPictures/img9.png"

#LP Extraction
results = licensePlateDetector([imagePath])  
result = results[0] 
cv2.imshow('Original Image', cv2.imread(imagePath, cv2.IMREAD_COLOR))
cv2.waitKey(0)
cv2.destroyAllWindows()
boxes = result.boxes  
result.show()   
x1, y1, x2, y2, score, class_id = boxes.data.tolist()[0][:]
lpCrop = result
croppedImg = lpCrop.orig_img[int(y1):int(y2), int(x1):int(x2), :]
cv2.imwrite('./resultingPictures/plate.png', croppedImg)

cv2.imshow('Extracted License Plate', croppedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
#Upscaling
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(upscalerModel_path), strict=True)
model.eval()
device = torch.device('cuda')
model = model.to(device)

toUpscale = cv2.imread('./resultingPictures/plate.png', cv2.IMREAD_COLOR)
toUpscale = toUpscale * 1.0 / 255
toUpscale = torch.from_numpy(np.transpose(toUpscale[:, :, [2, 1, 0]], (2, 0, 1))).float()
toUpscale_LR = toUpscale.unsqueeze(0)
toUpscale_LR = toUpscale_LR.to(device)
    
with torch.no_grad():
    output = model(toUpscale_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
cv2.imwrite('./resultingPictures/upscaledPlate.png', output)

cv2.imshow('Upscaled Plate', cv2.imread('./resultingPictures/upscaledPlate.png', cv2.IMREAD_COLOR))
cv2.waitKey(0)
cv2.destroyAllWindows()
 

#OCR
reader = easyocr.Reader(['en'])  
result = reader.readtext('./resultingPictures/upscaledPlate.png') 
for (bbox, text, prob) in result:
    print(f'Raw LP Text: {text}, Probability: {prob}')
result = reader.readtext('./resultingPictures/plate.png')  
for (bbox, text, prob) in result:
    print(f'Upscaled LP Text: {text}, Probability: {prob}')
  