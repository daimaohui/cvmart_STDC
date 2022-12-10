import json
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import json
sys.path.insert(1, '/project/train/src_repo/STDC-Seg/')
from models.model_stages import BiSeNet
import torchvision.transforms as transforms
import threading
def init():
    use_boundary_2=False
    use_boundary_4=False
    use_boundary_8=True
    use_boundary_16=False
    net = BiSeNet(backbone='STDCNet813', n_classes=2, pretrain_model="/project/train/src_repo/STDC-Seg/checkpoints/STDCNet813M_73.91.tar", 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8, 
    use_boundary_16=use_boundary_16, use_conv_last=False)
    net.load_state_dict(torch.load("/project/train/models/model_maxmIOU75.pth", map_location='cpu'))
    net.cuda()
    net.eval()
    return net
def run(mask_output_path,preds):
    cv2.imwrite(mask_output_path, preds)
def process_image(handle=None, input_image=None, args=None, **kwargs):
    args =json.loads(args)
    mask_output_path =args['mask_output_path']
    h, w, _ = input_image.shape
    model=handle
   
    scale=0.5
    input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    imgs = to_tensor(input_image)
    imgs=imgs.unsqueeze(0)
    imgs=imgs.cuda()
    # new_hw = [int(h*scale), int(w*scale)]
    # imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
    logits = model(imgs)[0]
    # logits = F.interpolate(logits, size=(h,w),
    #         mode='bilinear', align_corners=True)
    # probs = torch.softmax(logits, dim=1)
    # preds = torch.argmax(probs, dim=1).cpu().detach().numpy()[0]
    # # res=np.random.randn(h, w)
    # # cv2.imwrite(mask_output_path, preds)
    # t1 = threading.Thread(target=run, args=(mask_output_path,preds))
    # t1.start()
    # h, w, _ = input_image.shape
    # dummy_data = np.random.randint(2, size=(w, h), dtype=np.uint8)
    # pred_mask_per_frame = Image.fromarray(dummy_data)
    # pred_mask_per_frame.save(mask_output_path)
    mask_output_path="/project/ev_sdk/src/1.png"
    return json.dumps({
        "mask": mask_output_path,
        "algorithm_data": {},

        "model_data": {

            "mask": mask_output_path

        }

    }, indent=4)

if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/home/data/1856/CTWLroad_collapse20221207_V1_sample_online_12439.jpg')
    # img = np.random.randn(1024,512,3)
    model = init()
    process_image(model,img,json.dumps({"mask_output_path": "1.png"}, indent=4))