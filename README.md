# Self-Supervised-Label-Generator
This is a Python demo of the Self-Supervised Label Generator (SSLG), presented in ["Self-Supervised Drivable Area and Road Anomaly Segmentation using RGB-D Data for Robotic Wheelchairs"](https://arxiv.org/abs/2007.05950). Our SSLG can be used effectively for self-supervised drivable area and road anomaly segmentation based on RGB-D data.


![Self-supervised Label Generator](https://github.com/ntkhoa95/Self-Supervised-Label-Generator/blob/main/datasets/SSLG.png?raw=true)

The overview of SSLG method, which consists of
(a) Input of RGB-D images
(b) Processing pipeline of RGB-Dimages
(c) Output of self-supervised labels. 
(b) is composed of RGB Processing Pipeline shown in the orange box, Depth Processing Pipeline shown in the green box and (VII) Final Segmentation Label Generator shown in the blue lines. The RGB Processing Pipeline consists of (I) Original RGB Anomaly Map Generator and (II) Generation of final RGB anomaly maps. The Depth Processing Pipeline consists of (III) Computation of original v-disparity maps, (IV) Filtering of original v-disparity maps, (V) Extraction of the drivable area and original depth anomaly maps as well as (VI) Generation of final depth anomaly maps. The figure is best viewed in color.
