# 3D Human Pose Estimtaion

## Epipolar Line Module               
* Introduction  

![image](https://user-images.githubusercontent.com/59610723/152629125-3335ad9f-20af-4807-a69d-f2baf0e168f8.png)  
=> Proposed FrameWork <Make fused Heatmap>  
  
2D Detection Model(Code) Baseline => HRNet  
Link: https://github.com/HRNet/HRNet-Human-Pose-Estimation
  
* Make Heatmap: Change Destination Pixel <Convolution>
    ```
    Source View Size: 256X256
    Heatmap Size: 64X64
    Reference View Size: 256X256
    Heatmap Size: 64X64
    Weight Matrix Size: 3X3
    ```
* Module Image: make Heatmap  
![image](https://user-images.githubusercontent.com/59610723/152629601-a03826fa-ddc3-4d2a-b58f-e1d78c09d4f3.png)
![image](https://user-images.githubusercontent.com/59610723/152629622-da9a73e5-9907-4ffa-ba9b-1fc35fcd7b14.png)  
=> This Framework works on Fused Heatmap
  
I thought of two cased  
  * 1: 2D Detection model is Pretrained Model  
    ```
      1) Find Fundamental Matrix (OpenCV)
      2) produced list (Soruce View Heatmap Coordinate 64X64)
      3) Used Padding (Soruce View Heatmap 66X66)
      4) Apply Convolution (Weight Matirx, Souce View Heatmap)
      5) Find the epiolar line of the pixel
      6) Change Destination Pixel of Convolution (threshold 1 or 3)
      7) plus reference heatmap
    ```
  * 2: 2D Detection model is Pretrained Model
    ```
      1) Same as above
      I Hoped this case would go well with the training
    ```
  * Visualization  
  ![image](https://user-images.githubusercontent.com/59610723/152630027-c5439028-c84d-4857-839d-29ba5b6ccec8.png)
  
  * Can the proposed model be trained?  
  Anser : yes
  Think about Chain Rule  
  ![image](https://user-images.githubusercontent.com/59610723/152630076-8c9f0494-0367-4f7e-a459-6255ab63ce2c.png)  
  Destination Pixel does not affect training  
  
  * Contribution  
  ![image](https://user-images.githubusercontent.com/59610723/152630120-8fbf932c-4c9f-48f2-b7e1-c7ceb078be90.png)  
  Number of training parameters is 9 However, constant A, B, C, D, ..., I is also trained (Change Fundamental Matrix)
  
  Approximately 18 trainable parameters are created (Only 9)
  
  * Main Problem
    ```
    It takes a long time to calculate (loss update time, make heatmap time)
    problem of gradient vanishing (use sigmoid)
    ```  
* Pretrained Model <Freeze>  
  
![image](https://user-images.githubusercontent.com/59610723/152630329-ec391d99-81d7-4484-8109-c8e6d27f4771.png)  
=> 0 Iteration  
![image](https://user-images.githubusercontent.com/59610723/152630337-216130c9-f45f-4f31-9798-f45d29d11dcf.png)  
=> 10 Iteration  
![image](https://user-images.githubusercontent.com/59610723/152630348-6bbc30a6-5a15-413a-9424-f632c3cdf6c3.png)  
=> 50 Iteration  
  
Training Time => 4 Hour 50 Iteration <Batch Size 8>  
  
* Pretrained Model <Not Freeze>  
![image](https://user-images.githubusercontent.com/59610723/152630374-6e679aba-9b2f-42b9-b686-e6459662d74b.png)  
=> 60 Iterantion 

Training Time => 1 day 190 Iteration <Batch Size 8>  
Big Problem  
  
## Make Another Epiploar Heatmap  
![image](https://user-images.githubusercontent.com/59610723/152630450-eb7ce7cd-e958-4fbc-a304-79784cc9100b.png)  
  * New Heatmap
    ```
    New Heatmap (Same as above)
    Convolution is Simple Code
    ```
    ```
    Experiment <human 3.6m dataset Train-set subject only 1, Test-set only 9> - feasibility
    Batch size - 16
    Benchmark - HRNet
    Training Time = 3 Day
    The Training time is long, but the test time is short
    ```
  Epoch 4 - Test <Proposed Module>
  ![image](https://user-images.githubusercontent.com/59610723/152630561-98e6503d-79af-4068-a89d-d9aef15edd35.png)  
  valiate 1600  
  ![image](https://user-images.githubusercontent.com/59610723/152630576-55244626-f8d6-48ac-9496-491b6d557180.png)  
  validate 2200  
  ![image](https://user-images.githubusercontent.com/59610723/152630587-2c82269c-46b3-4f8e-9df4-e02139cee045.png)  
  train 2200  
  ![image](https://user-images.githubusercontent.com/59610723/152630580-f9713af2-8a97-426c-bed7-dec21dc4d72d.png)  
  
  averge acc => Approximately 20%  
  
  Benchmark - Hrnet Approximately 46%  
  Failed Project...
## Key Code
  

  
