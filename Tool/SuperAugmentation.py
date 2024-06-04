import cv2
import torch
import numpy as np

def Super_Augmentation(video_tensor, mask_prob=0.1, noise_std=10):



    # print("Hey bro, here is Super Augmentation!")
    num_frames, num_channels, height, width = video_tensor.shape

 
    processed_video = torch.zeros_like(video_tensor)


    for frame_idx in range(num_frames):
    
        frame = video_tensor[frame_idx]

     
        mask = np.random.rand(height, width) < mask_prob
        frame = frame * (1 - mask)

      
        noise = np.random.normal(0, noise_std, size=(num_channels, height, width)).astype(np.float32)
        frame = frame + noise

        
        frame = np.clip(frame, 0, 255)

       
        processed_video[frame_idx] = torch.tensor(frame)

    return processed_video


# processed_video = process_video(video_tensor, mask_prob=0.1, noise_std=10)