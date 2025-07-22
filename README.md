# Awesome-4D-Spatial-Intelligence

This repository collects summaries of over 350 recent studies on methods for reconstructing 4D spatial intelligence, and will be continuously updated.

If you have suggestions for new resources, improvements to methodologies, or corrections for broken links, please don't hesitate to open an [issue](https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence/issues) or submit a [pull request](hhttps://github.com/yukangcao/Awesome-4D-Spatial-Intelligence/pulls). Contributions of all kinds are welcome and greatly appreciated.

### Table of Contents

- [Level1 -- Low-level 3D cues](#level-1----low-level-3d-cues)
  - [Dynamic Video to Depth Estimation](#dynamic-video-to-depth-estimation)
  - [Static Video to Depth Estimation](#static-video-to-depth-estimation)
  - [Camera pose estimation](#camera-pose-estimation)
  - [3D tracking](#3d-tracking)
  - [Unifying depth and camera pose estimation](#unifying-depth-and-camera-pose-estimation)
  - [Unifying depth, camera pose, and 3D tracking](#unifying-depth-camera-pose-and-3d-tracking)

- [Level2 -- 3D scene components](#level-2----3d-scene-components)
  - [Small-scale 3D object/scene reconstruction](#small-scale-3d-objectscene-reconstruction)
  - [Large-scale 3D scene reconstruction](#large-scale-3d-scene-reconstruction)

- [Level3 -- 4D dynamic scenes](#level-3----4d-dynamic-scenes)
  - [General 4D scene reconstruction](#general-4d-scene-reconstruction)
  - [Human-centric dynamic modeling](#human-centric-dynamic-modeling---smpl)
    - [SMPL](#human-centric-dynamic-modeling---smpl)
    - [Egocentric](#human-centric-dynamic-modeling---egocentric)
    - [Appearance-rich](#human-centric-dynamic-modeling---appearance-rich)
    
- [Level 4 -- Interaction among scene components](#level-4----interaction-among-scene-components)
  - [SMPL-based human-centric interaction](#smpl-based-human-centric-interaction---hoi)
    - [Human-Object-Interaction (HOI)](#smpl-based-human-centric-interaction---hoi)
    - [Human-Scene-Interaction (HSI)](#smpl-based-human-centric-interaction---hsi)
    - [Human-Human-Interaction (HHI)](#smpl-based-human-centric-interaction---hhi)
  - [Appearance-rich human-centric interaction](#appearance-rich-human-centric-interaction)
  - [Egocentric human-centric interaction](#egocentric-human-centric-interaction)
  
- [Level 5 -- Incorporation of physical laws and constraints](#level-5----incorporation-of-physical-laws-and-constraints)
  - [Dynamic 4D human simulation with physics](#dynamic-4d-human-simulation-with-physics)
  - [3D scene reconstruction with physical plausibility](#3d-scene-reconstruction-with-physical-plausibility)

## Level 1 -- Low-level 3D cues
### Dynamic Video to Depth Estimation

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2019 | ICCV | GLNet | [Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Self-Supervised_Learning_With_Geometric_Constraints_in_Monocular_Video_Connecting_Flow_ICCV_2019_paper.pdf) | | |
| 2020 | ICLR | DeepV2D | [DeepV2D: Video to Depth with Differentiable Structure from Motion](https://arxiv.org/abs/1812.04605) |  | [GitHub](https://github.com/princeton-vl/DeepV2D) |
| 2020 | SIGGRAPH |  | [Consistent Video Depth Estimation](https://arxiv.org/abs/2004.15021) | [Project](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/) | [GitHub](https://github.com/facebookresearch/consistent_depth) |
| 2021 | TOG |  | [Consistent depth of moving objects in video](https://dl.acm.org/doi/pdf/10.1145/3450626.3459871) | [Project](https://dynamic-video-depth.github.io/) | [GitHub](https://github.com/google/dynamic-video-depth) |
| 2022 | ACMMM | FMNet | [Less is More: Skip Connections in Video Depth Estimation](https://arxiv.org/pdf/2208.00380) |  | [GitHub](https://github.com/RaymondWang987/FMNet) |
| 2023 | WACV | CODD | [Temporally consistent online depth estimation in dynamic scenes](https://arxiv.org/pdf/2111.09337) | [Project](https://mli0603.github.io/codd/) | [GitHub](https://github.com/facebookresearch/CODD) |
| 2023 | ICCV | MAMo | [Mamo: Leveraging memory and attention for monocular video depth estimation](https://arxiv.org/pdf/2307.14336) | | |
| 2023 | ICCV | NVDS | [Neural Video Depth Stabilizer](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Neural_Video_Depth_Stabilizer_ICCV_2023_paper.pdf) | [Project](https://raymondwang987.github.io/NVDS/) | [GitHub](https://github.com/RaymondWang987/NVDS) |
| 2024 | T-PAMI | NVDS+ | [NVDS+: Towards Efficient and Versatile Neural Stabilizer for Video Depth Estimation](https://arxiv.org/pdf/2307.08695) | [Project](https://raymondwang987.github.io/NVDS/) | [GitHub](https://github.com/RaymondWang987/NVDS) |
| 2025 | ICLR  | DepthAnyVideo | [Depth Any Video with Scalable Synthetic Data](https://arxiv.org/abs/2410.10815) | [Project](https://depthanyvideo.github.io/)  | [GitHub](https://github.com/Nightmare-n/DepthAnyVideo) |
| 2025 | CVPR | DepthCrafter | [DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](https://arxiv.org/abs/2409.02095) | [Project](https://depthcrafter.github.io/) | [GitHub](https://github.com/Tencent/DepthCrafter) |
| 2025 | CVPR | ChronoDepth | [Learning Temporally Consistent Video Depth from Video Diffusion Priors](https://arxiv.org/abs/2406.01493) | [Project](https://xdimlab.github.io/ChronoDepth/) | [GitHub](https://github.com/jiahao-shao1/ChronoDepth) |
| 2025 | CVPR | Video Depth Anything | [Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](https://arxiv.org/abs/2501.12375) | [Project](https://videodepthanything.github.io/)  | [GitHub](https://github.com/DepthAnything/Video-Depth-Anything) |


### Static Video to Depth Estimation

| Year | Venue      | Acronym | Paper                                                                                                                                               | Project | GitHub |
|------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------|--------|
| 2018 | CVPR       | GeoNet  | [GeoNet: Unsupervised learning of dense depth, optical flow and camera pose](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yin_GeoNet_Unsupervised_Learning_CVPR_2018_paper.pdf) | – | [GitHub](https://github.com/yzcjtr/GeoNet) |
| 2019 | NeurIPS    | SC-SfMLearner | [Unsupervised scale-consistent depth and ego-motion learning from monocular video](https://proceedings.neurips.cc/paper_files/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf) | – | [GitHub](https://github.com/JiawangBian/SC-SfMLearner-Release) |
| 2019 | ICCV       | –       | [Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gordon_Depth_From_Videos_in_the_Wild_Unsupervised_Monocular_Depth_Learning_ICCV_2019_paper.pdf) | – | [GitHub](https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild) |
| 2019 | ICCV       | –       | [Exploiting temporal consistency for real-time video depth estimation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Exploiting_Temporal_Consistency_for_Real-Time_Video_Depth_Estimation_ICCV_2019_paper.pdf) | – | [GitHub](https://github.com/hkzhang-git/ST-CLSTM) |
| 2019 | IEEE T‑ITS | FlowGRU | [Temporally consistent depth prediction with flow‑guided memory units](https://openaccess.thecvf.com/content/WACV2023/papers/Li_Temporally_Consistent_Online_Depth_Estimation_in_Dynamic_Scenes_WACV_2023_paper.pdf) | [Project](https://cvlab.yonsei.ac.kr/projects/FlowGRU/) | [GitHub](https://github.com/cvlab-yonsei/FlowGRU) |
| 2020 | IEEE RA‑L  | –       | [Don’t forget the past: Recurrent depth estimation from monocular video](https://arxiv.org/pdf/2001.02613) | [Project](https://www.trace.ethz.ch/publications/2020/rec_depth_estimation/index.html) | – |
| 2020 | IROS       | FDNet   | [Video depth estimation by fusing flow‑to‑depth proposals](https://ieeexplore.ieee.org/document/9341191) | [Project](https://jiaxinxie97.github.io/Jiaxin-Xie/FDNet/FDNet) | [GitHub](https://github.com/jiaxinxie97/Video-depth-estimation) |
| 2020 | SIGGRAPH   | –       | [Consistent video depth estimation](https://arxiv.org/abs/2004.15021) | [Project](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/) | [GitHub](https://github.com/facebookresearch/consistent_depth) |
| 2021 | CVPR       | CVD     | [Robust consistent video depth estimation](https://arxiv.org/pdf/2012.05901) | [Project](https://robust-cvd.github.io/) | [GitHub](https://github.com/facebookresearch/robust_cvd) |
| 2021 | CVPR       | ESTDepth | [Multi‑view depth estimation using epipolar spatio‑temporal networks](https://openaccess.thecvf.com/content/CVPR2021/papers/Long_Multi-View_Depth_Estimation_Using_Epipolar_Spatio-Temporal_Networks_CVPR_2021_paper.pdf) | [Project](https://www.xxlong.site/ESTDepth/) | [GitHub](https://github.com/xxlong0/ESTDepth) |
| 2021 | CVPR       | ManyDepth | [The temporal opportunist: Self‑supervised multi‑frame monocular depth](https://arxiv.org/abs/2104.14540) | – | [GitHub](https://github.com/nianticlabs/manydepth) |
| 2021 | ECCV       | SimpleRecon | [Simplerecon: 3D reconstruction without 3D convolutions](https://arxiv.org/abs/2208.14743) | [Project](https://nianticlabs.github.io/simplerecon/) | [GitHub](https://github.com/nianticlabs/simplerecon) |
| 2022 | CVPR       | DepthFormer | [Multi‑frame self‑supervised depth with transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Guizilini_Multi-Frame_Self-Supervised_Depth_With_Transformers_CVPR_2022_paper.pdf) | [Project](https://sites.google.com/tri.global/depthformer) | – |
| 2022 | 3DV        | MonoViT | [Monovit: Self‑supervised monocular depth estimation with a vision transformer](https://arxiv.org/abs/2208.03543) | – | [GitHub](https://github.com/zxcqlf/MonoViT) |
| 2022 | ACM MM     | FMNet   | [Less is more: Consistent video depth estimation with masked frames modeling](https://dl.acm.org/doi/10.1145/3503161.3547713) | – | [GitHub](https://github.com/RaymondWang987/FMNet) |
| 2023 | ICCV       | MAMo    | [MAMo: Leveraging memory and attention for monocular video depth estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Yasarla_MAMo_Leveraging_Memory_and_Attention_for_Monocular_Video_Depth_Estimation_ICCV_2023_paper.pdf) | – | – |
| 2024 | T-PAMI     | NVDS+   | [NVDS: Towards Efficient and Versatile Neural Stabilizer for Video Depth Estimation](https://arxiv.org/pdf/2307.08695) | [Project](https://raymondwang987.github.io/NVDS/) | [GitHub](https://github.com/RaymondWang987/NVDS) |
| 2024 | ECCV       | FutureDepth | [FutureDepth: Learning to predict the future improves video depth estimation](https://arxiv.org/pdf/2403.12953) | – | – |
| 2025 | CVPR       | ChronoDepth | [Learning Temporally Consistent Video Depth from Video Diffusion Priors](https://arxiv.org/abs/2406.01493) | [Project](https://xdimlab.github.io/ChronoDepth/) | [GitHub](https://github.com/jiahao-shao1/ChronoDepth) |
| 2024 | arXiv      | DepthAnyVideo | [Depth Any Video with Scalable Synthetic Data](https://arxiv.org/abs/2410.10815) | [Project](https://depthanyvideo.github.io/) | [GitHub](https://github.com/Nightmare-n/DepthAnyVideo) |
| 2025 | CVPR       | –       | [Video Depth Anything: Consistent Depth Estimation for Super‑Long Videos](https://arxiv.org/abs/2501.12375) | [Project](https://videodepthanything.github.io/) | [GitHub](https://github.com/DepthAnything/Video-Depth-Anything) |


---

### Camera pose estimation

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2014 | ECCV | LSD-SLAM | [LSD-SLAM: Large-scale direct monocular SLAM](https://jakobengel.github.io/pdf/engel14eccv.pdf) | [Project](https://cvg.cit.tum.de/research/vslam/lsdslam?redirect=1) | [GitHub](https://github.com/tum-vision/lsd_slam) |
| 2015 | TRO | ORB-SLAM | [ORB-SLAM: A Versatile and Accurate Monocular SLAM System](https://arxiv.org/abs/1502.00956) | [Project](https://webdiis.unizar.es/~raulmur/orbslam/) | [GitHub](https://github.com/raulmur/ORB_SLAM) |
| 2017 | T-PAMI | DSO | [Direct Sparse Odometry](https://arxiv.org/abs/1607.02565) | [Project](https://cvg.cit.tum.de/research/vslam/dso?redirect=1) | [GitHub](https://github.com/JakobEngel/dso) |
| 2017 | TRO | ORB-SLAM2 | [ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras](https://arxiv.org/abs/1610.06475) | | [GitHub](https://github.com/raulmur/ORB_SLAM2) |
| 2017 | ICRA | DeepVO | [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429) | [Project](https://senwang.gitlab.io/DeepVO/) | |
| 2020 | CVPR | D3VO | [D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](https://arxiv.org/abs/2003.01060) | | [GitHub](https://github.com/as821/D3VO) |
| 2021 | CoRL | TartanVO | [TartanVO: A Generalizable Learning-based VO](https://arxiv.org/abs/2011.00359) | | [GitHub](https://github.com/castacks/tartanvo) |
| 2021 | TITS | DDSO | [Deep Direct Visual Odometry](https://arxiv.org/abs/1912.05101) | | |
| 2021 | T-RO | LF-SLAM | [Line Flow Based Simultaneous Localization and Mapping](https://ieeexplore.ieee.org/abstract/document/9393474) | | |
| 2022 | ICRA | EDPLVO | [EDPLVO: Efficient Direct Point-Line Visual Odometry](https://www.cs.cmu.edu/~kaess/pub/Zhou22icra.pdf) | | |
| 2023 | ICCV | XVO | [XVO: Generalized Visual Odometry via Cross-Modal Self-Training](https://arxiv.org/abs/2309.16772) | [Project](https://genxvo.github.io/) | [GitHub](https://github.com/h2xlab/XVO) |
| 2023 | ICRA | DytanVO | [DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments](https://arxiv.org/abs/2209.08430) | | [GitHub](https://github.com/castacks/DytanVO) |
| 2023 | IROS | StereoVO | [Stereo VO with Point and Line Matching using Attention GNN](https://arxiv.org/abs/2308.01125) | | |
| 2023 | ICRA | Structure PLP-SLAM | [Structure PLP-SLAM: Efficient Sparse Mapping using Point, Line and Plane](https://arxiv.org/abs/2207.06058) | | [GitHub](https://github.com/PeterFWS/Structure-PLP-SLAM) |
| 2024 | ECCV | DPV-SLAM | [Deep Patch Visual SLAM](https://arxiv.org/abs/2408.01654) | | [GitHub](https://github.com/princeton-vl/DPVO) |
| 2024 | ECCV | RLVO | [Reinforcement Learning Meets Visual Odometry](https://arxiv.org/abs/2407.15626) | | [GitHub](https://github.com/uzh-rpg/rl_vo) |
| 2024 | RA-L | | [Efficient Camera Exposure Control for VO via Deep RL](https://arxiv.org/abs/2408.17005) | | [GitHub](https://github.com/ShuyangUni/drl_exposure_ctrl) |
| 2024 | T-ASE | UL-SLAM | [UL-SLAM: A Universal Monocular Line-Based SLAM via Unifying Structural and Non-Structural Constraints](https://ieeexplore.ieee.org/document/10488029) | | [GitHub](https://github.com/jhch1995/UL-SLAM-Mono) |
| 2023 | NeurIPS | DPVO | [Deep Patch Visual Odometry](https://arxiv.org/abs/2208.04726) | | [GitHub](https://github.com/princeton-vl/DPVO) |
| 2025 | CVPR | AnyCam | [AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos](https://arxiv.org/abs/2503.23282) | [Project](https://fwmb.github.io/anycam/) | [GitHub](https://github.com/Brummi/anycam) |
| 2025 | CVPR | DynPose | [Dynamic Camera Poses and Where to Find Them](https://arxiv.org/abs/2504.17788) | [Project](https://research.nvidia.com/labs/dir/dynpose-100k/) | |
| 2025 | T-RO | AirSLAM | [AirSLAM: An Efficient and Illumination-Robust Point-Line SLAM System](https://arxiv.org/abs/2408.03520) | [Project](https://xukuanhit.github.io/airslam/) | [GitHub](https://github.com/sair-lab/AirSLAM) |

---

### 3D tracking

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2023 | ICCV | OmniMotion | [Tracking Everything Everywhere All at Once](https://arxiv.org/abs/2306.05422) | [Project](https://omnimotion.github.io/) | [GitHub](https://github.com/qianqianwang68/omnimotion) |
| 2024 | ECCV | OmniTrackFast | [Track Everything Everywhere Fast and Robustly](https://arxiv.org/abs/2403.17931) | [Project](https://timsong412.github.io/FastOmniTrack/) | [GitHub](https://github.com/TimSong412/OmniTrackFast) |
| 2024 | CVPR | SpatialTracker | [SpatialTracker: Tracking Any 2D Pixels in 3D Space](https://arxiv.org/abs/2404.04319) | [Project](https://henry123-boy.github.io/SpaTracker/) | [GitHub](https://github.com/henry123-boy/SpaTracker) |
| 2025 |  T-PAMI | SceneTracker | [SceneTracker: Long-term Scene Flow Estimation Network](https://arxiv.org/abs/2403.19924) | | [GitHub](https://github.com/wwsource/SceneTracker) |
| 2025 | ICLR | DELTA | [DELTA: Dense Efficient Long-range 3D Tracking for any video](https://arxiv.org/abs/2410.24211) | [Project](https://snap-research.github.io/DELTA/) | [GitHub](https://github.com/snap-research/) |
| 2025 | CVPR | Seurat | [Seurat: From Moving Points to Depth](https://arxiv.org/abs/2504.14687) | [Project](https://seurat-cvpr.github.io/) | [GitHub](https://github.com/cvlab-kaist/seurat) |
| 2025 | arXiv | TAPIP3D | [TAPIP3D: Tracking Any Point in Persistent 3D Geometry](https://arxiv.org/abs/2504.14717) | [Project](https://tapip3d.github.io/) | [GitHub](https://github.com/zbw001/TAPIP3D) |


---

### Unifying depth and camera pose estimation

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2021 | CVPR  | Robust-CVD | [Robust consistent video depth estimation](https://arxiv.org/abs/2012.05901) | [Project](https://robust-cvd.github.io/) | [GitHub](https://github.com/facebookresearch/robust_cvd) |
| 2022 | ECCV  | CasualSAM | [Structure and motion from casual videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930020.pdf) |         |             |
| 2025 | CVPR    | MegaSam | [Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos](https://arxiv.org/abs/2412.04463) | [Project](https://mega-sam.github.io/) | [GitHub](https://github.com/mega-sam/mega-sam) |
| 2025 | 3DV    |  Spann3R | [3D reconstruction with spatial memory](https://arxiv.org/abs/2408.16061) | [Project](https://hengyiwang.github.io/projects/spanner) | [GitHub](https://github.com/HengyiWang/spann3r) |
| 2025 | ICLR    | MonST3R | [MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion](https://arxiv.org/abs/2410.03825) | [Project](https://monst3r-project.github.io/) | [GitHub](https://github.com/Junyi42/monst3r) |
| 2025 | CVPR    | Align3R | [Align3R: Aligned Monocular Depth Estimation for Dynamic Videos](https://arxiv.org/abs/2412.03079) | [Project](https://igl-hkust.github.io/Align3R.github.io/) | [GitHub](https://github.com/jiah-cloud/Align3R) |
| 2025 | CVPR    | CUT3R | [Continuous 3D Perception Model with Persistent State](https://arxiv.org/abs/2501.12387) | [Project](https://cut3r.github.io/) | [GitHub](https://github.com/CUT3R/CUT3R) |
| 2025 | ICCV    | Easi3R  | [Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](https://arxiv.org/abs/2503.24391) | [Project](https://easi3r.github.io/) | [GitHub](https://github.com/Inception3D/Easi3R) |
| 2025 | ICCV    | Geometrycrafter | [Geometrycrafter: Consistent geometry estimation for open-world videos with diffusion priors](https://arxiv.org/abs/2504.01016) | [Project](https://geometrycrafter.github.io/) | [GitHub](https://github.com/TencentARC/GeometryCrafter) |
| 2025 | ICCV     | Aether  | [Aether: Geometric-aware unified world modeling](https://arxiv.org/abs/2503.18945) | [Project](https://aether-world.github.io/) | [GitHub](https://github.com/OpenRobotLab/Aether) |
| 2025 | ICCV    | Geo4D   | [Geo4d: Leveraging video generators for geometric 4D scene reconstruction](https://arxiv.org/abs/2504.07961) | [Project](https://geo4d.github.io/) | [GitHub](https://github.com/jzr99/Geo4D) |
| 2025 | arXiv    | UniGeo  | [UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation](https://arxiv.org/abs/2505.24521) | [Project](https://sunyangtian.github.io/UniGeo-web/) | [GitHub](https://github.com/SunYangtian/UniGeo) |
| 2025 | arXiv    | Point3R  | [Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory](https://arxiv.org/abs/2507.02863) | [Project](https://ykiwu.github.io/Point3R/) | [GitHub](https://github.com/YkiWu/Point3R) |
| 2025 | arXiv    | StreamVGGT  | [Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539) | [Project](https://wzzheng.net/StreamVGGT/) | [GitHub](https://github.com/wzzheng/StreamVGGT) |
| 2025 | arXiv    | π$`^3`$  | [π$`^3`$: Scalable Permutation-Equivariant Visual Geometry Learning](https://arxiv.org/abs/2507.13347) | [Project](https://yyfz.github.io/pi3/) | [GitHub](https://github.com/yyfz/Pi3) |

### Unifying depth, camera pose, and 3D tracking

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2024 | NeurIPS   | TracksTo4D | [Fast Encoder-Based 3D from Casual Videos via Point Track Processing](https://arxiv.org/abs/2404.07097)     | [Project](https://tracks-to-4d.github.io/) | [GitHub](https://github.com/NVlabs/tracks-to-4d) |
| 2025 | CVPR      | Uni4D       | [Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](https://arxiv.org/abs/2503.21761)  | [Project](https://davidyao99.github.io/uni4d/) | [GitHub](https://github.com/Davidyao99/uni4d/tree/main) |
| 2025 | arXiv     | BA-Track | [Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](https://arxiv.org/abs/2504.14516)           | [Project](https://wrchen530.github.io/projects/batrack/) | |  
| 2025 | arXiv     | TrackingWorld | [TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels](https://github.com/IGL-HKUST/TrackingWorld)           | [Project](https://github.com/IGL-HKUST/TrackingWorld) | [GitHub](https://github.com/IGL-HKUST/TrackingWorld) | 
| 2025 | CVPR     | Stereo4D    | [Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](https://arxiv.org/abs/2412.09621)       | [Project](https://stereo4d.github.io/) | [GitHub](https://github.com/Stereo4d/stereo4d-code) | 
| 2025 | arXiv     | DPM | [Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](https://arxiv.org/abs/2503.16318) | [Project](https://www.robots.ox.ac.uk/~vgg/research/dynamic-point-maps/) | |
| 2025 | ICCV     | St4RTrack   | [St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World](https://arxiv.org/abs/2504.13152)         | [Project](https://st4rtrack.github.io/) | |
| 2025 | arXiv     | POMATO      | [POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction](https://arxiv.org/abs/2504.05692) | | [GitHub](https://github.com/wyddmw/POMATO) |
| 2025 | arXiv     | D$`^2`$USt3R    | [D$`^2`$USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes](https://arxiv.org/abs/2504.06264)    | [Project](https://cvlab-kaist.github.io/DDUSt3R/) | |
| 2025 | CVPR      | VGGT        | [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)  | [Project](https://vgg-t.github.io/) | [GitHub](https://github.com/facebookresearch/vggt) |
| 2025 | CVPR      | Zero-MSF | [Zero-Shot Monocular Scene Flow Estimation in the Wild](https://arxiv.org/abs/2501.10357)                        | [Project](https://research.nvidia.com/labs/lpr/zero_msf//) | [GitHub](https://github.com/SunYangtian/UniGeo) |
| 2025 | ICCV | SpatialTrackerV2 | [SpatialTrackerV2: 3D Point Tracking Made Easy](https://arxiv.org/abs/2507.12462) | [Project](https://spatialtracker.github.io/) | [GitHub](https://github.com/henry123-boy/SpaTrackerV2) |

## Level 2 -- 3D scene components

### Small-scale 3D object/scene reconstruction

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2006 | Photogrammetric Computer Vision |  | [Bundle Adjustment Rules](https://www.isprs.org/proceedings/xxxvi/part3/singlepapers/O_24.pdf)     | |  |
| 2010 | ECCV |  | [Bundle adjustment in the large](https://homes.cs.washington.edu/~sagarwal/bal.pdf)     | [Project](https://grail.cs.washington.edu/projects/bal/) |  |
| 2016 | CVPR | COLMAP | [Structure-from-motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)     |  |  [GitHub](https://github.com/colmap/colmap) |
| 2016 | ECCV |  | [Pixelwise View Selection for Unstructured Multi-View Stereo](https://demuc.de/papers/schoenberger2016mvs.pdf)     |  |  [GitHub](https://github.com/mitjap/pwmvs) |
| 2019 | ECCV | MVSNet | [MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://demuc.de/papers/schoenberger2016mvs.pdf)     |  |  [GitHub](https://github.com/YoYo000/MVSNet) |
| 2020 | BMVC |  | [Visibility-aware Multi-view Stereo Network](https://arxiv.org/abs/2008.07928)     |  |  [GitHub](https://github.com/jzhangbs/Vis-MVSNet) |
| 2020 | CVPR |  | [Cost Volume Pyramid Based Depth Inference for Multi-View Stereo](https://arxiv.org/abs/1912.08329)     |  |  [GitHub](https://github.com/JiayuYANG/CVP-MVSNet) |
| 2020 | CVPR |  | [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/abs/1912.06378)     |  |  [GitHub](https://github.com/alibaba/cascade-stereo) |
| 2021 | ICCV | | [Pixel-Perfect Structure-from-Motion with Featuremetric Refinement](https://arxiv.org/abs/2108.08291)     |  |  [GitHub](https://github.com/cvg/pixel-perfect-sfm) |
| 2021 | NeurIPS | NeuS | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/abs/2106.10689)     | [Project](https://lingjie0206.github.io/papers/NeuS/) |  [GitHub](https://github.com/Totoro97/NeuS?tab=readme-ov-file) |
| 2021 | NeurIPS | VolSDF | [Volume Rendering of Neural Implicit Surfaces](https://arxiv.org/abs/2106.12052)     | [Project](https://lioryariv.github.io/volsdf/) |  [GitHub](https://github.com/lioryariv/volsdf) |
| 2021 | CVPR | PatchMatchNet | [PatchmatchNet: Learned Multi-View Patchmatch Stereo](https://arxiv.org/abs/2108.08291)     |  |  [GitHub](https://github.com/FangjinhuaWang/PatchmatchNet) |
| 2022 | ECCV | SparseNeuS | [SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views](https://arxiv.org/abs/2206.05737)     | [Project](https://www.xxlong.site/SparseNeuS/) |  [GitHub](https://github.com/xxlong0/SparseNeuS) |
| 2023 | ICCV | C2F2NeUS | [C2F2NeUS: Cascade Cost Frustum Fusion for High Fidelity and Generalizable Neural Surface Reconstruction](https://arxiv.org/abs/2306.10003)     | | |
| 2023 | NeurIPS | GenS | [GenS: Generalizable Neural Surface Reconstruction from Multi-View Images](https://arxiv.org/abs/2406.02495)     | |  [GitHub](https://github.com/prstrive/GenS) |
| 2023 | CVPR | NeAT | [NeAT: Learning Neural Implicit Surfaces with Arbitrary Topologies from Multi-view Images](https://arxiv.org/abs/2303.12012)     | [Project](https://xmeng525.github.io/xiaoxumeng.github.io/projects/cvpr23_neat) |  [GitHub](https://github.com/xmeng525/NeAT) |
| 2023 | CVPR | Neuralangelo | [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://arxiv.org/abs/2306.03092)     | [Project](https://research.nvidia.com/labs/dir/neuralangelo/) |  [GitHub](https://github.com/NVlabs/neuralangelo) |
| 2024 | Siggraph | 2DGS | [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888)     | [Project](https://surfsplatting.github.io/) |  [GitHub](https://github.com/hbb1/2d-gaussian-splatting) |
| 2024 | TVCG | PGSR | [PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction](https://arxiv.org/abs/2406.06521)     | [Project](https://zju3dv.github.io/pgsr/) |  [GitHub](https://github.com/zju3dv/PGSR) |
| 2025 |  | SolidGS | [SolidGS: Consolidating Gaussian Surfel Splatting for Sparse-View Surface Reconstruction](https://arxiv.org/abs/2412.15400)     | [Project](https://mickshen7558.github.io/projects/SolidGS/) | |
| 2024 | CVPR | UFORecon | [UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and UnFavOrable Sets](https://arxiv.org/abs/2403.05086)     | [Project](https://youngju-na.github.io/uforecon.github.io/) |  [GitHub](https://github.com/Youngju-Na/UFORecon/) |
| 2024 | CVPR | SuGaR | [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://arxiv.org/abs/2311.12775)     | [Project](https://anttwo.github.io/sugar/) |  [GitHub](https://github.com/Anttwo/SuGaR) |
| 2024 | ECCV | SuRF | [Surface-Centric Modeling for High-Fidelity Generalizable Neural Surface Reconstruction](https://arxiv.org/abs/2409.03634)     | |  [GitHub](https://github.com/prstrive/SuRF) |
| 2024 | ECCV | LoRa | [LaRa: Efficient Large-Baseline Radiance Fields](https://arxiv.org/abs/2407.04699)     | [Project](https://apchenstu.github.io/LaRa/) |  [GitHub](https://github.com/autonomousvision/LaRa) |
| 2024 | ECCV | SceneScript | [SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model](https://arxiv.org/abs/2403.13064)     | [Project](https://www.projectaria.com/scenescript/) |  [GitHub](https://github.com/facebookresearch/scenescript) |
| 2024 | ECCV | EgoLifter | [EgoLifter: Open-world 3D Segmentation for Egocentric Perception](https://arxiv.org/abs/2403.18118)     | [Project](https://egolifter.github.io/) |  [GitHub](https://github.com/facebookresearch/egolifter) |
| 2025 | TMM | Tri2Plane | [Tri2Plane: Advancing Neural Implicit Surface Reconstruction for Indoor Scenes](https://ieeexplore.ieee.org/document/10982030)     |  |  |
| 2025 | CVPR | GaussianUDF | [GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting](https://arxiv.org/abs/2503.19458)     | [Project](https://lisj575.github.io/GaussianUDF/) | [GitHub](https://github.com/lisj575/GaussianUDF) |
| 2025 |  | SOF | [SOF: Sorted Opacity Fields for Fast Unbounded Surface Reconstruction](https://www.arxiv.org/abs/2506.19139)     |  | |
| 2025 |  | SOF | [RNb-NeuS2: Multi-View Surface Reconstruction Using Normal and Reflectance Cues](https://arxiv.org/abs/2506.04115)     | | [GitHub](https://github.com/RobinBruneau/RNb-NeuS2) |
| 2025 |  | QuickSplat | [QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization](https://arxiv.org/abs/2505.05591)     | [Project](https://liu115.github.io/quicksplat) | |
| 2025 | CVPR | | [Geometry Field Splatting with Gaussian Surfels](https://arxiv.org/abs/2411.17067)     | | |
| 2025 | NeurIPS | ReTR | [ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction](https://arxiv.org/abs/2305.18832)     | [Project](https://yixunliang.github.io/ReTR/) | [GitHub](https://github.com/YixunLiang/ReTR) |


---

### Large-scale 3D scene reconstruction

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2021 | CVPR | NeRF++ | [NeRF++: Analyzing and Improving Neural Radiance Fields](https://arxiv.org/abs/2404.10772)     | |  [GitHub](https://github.com/Kai-46/nerfplusplus) |
| 2021 | CVPR | NeuralRecon | [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://arxiv.org/abs/2104.00681)     | [Project](https://zju3dv.github.io/neuralrecon/) |  [GitHub](https://github.com/zju3dv/NeuralRecon) |
| 2021 | NeurIPS | TransformerFusion | [TransformerFusion: Monocular RGB Scene Reconstruction using Transformers](https://arxiv.org/abs/2107.02191)     | |  [GitHub](https://github.com/AljazBozic/TransformerFusion) |
| 2022 | CVPR | Block-NeRF | [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://arxiv.org/abs/2202.05263)     | [Project](https://waymo.com/research/block-nerf/) | |
| 2022 | CVPR | Mega-NeRF | [Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs](https://arxiv.org/abs/2112.10703)     | [Project](https://meganerf.cmusatyalab.org/) | [GitHub](https://github.com/cmusatyalab/mega-nerf) |
| 2022 | ECCV | BungeeNeRF | [BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering](https://arxiv.org/abs/2112.05504)     | [Project](https://city-super.github.io/citynerf/) | [GitHub](https://github.com/city-super/BungeeNeRF) |
| 2022 | NeurIPS | MonoSDF | [MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction](https://arxiv.org/abs/2206.00665)     | [Project](https://niujinshuchong.github.io/monosdf/) | [GitHub](https://github.com/autonomousvision/monosdf) |
| 2023 | ICCV | FineRecon | [FineRecon: Depth-aware Feed-forward Network for Detailed 3D Reconstruction](https://arxiv.org/abs/2304.01480)     |  |  [GitHub](https://github.com/apple/ml-finerecon) |
| 2023 | ICCV | CVRecon | [CVRecon: Rethinking 3D Geometric Feature Learning For Neural Reconstruction](https://arxiv.org/abs/2304.14633)     | [Project](https://cvrecon.ziyue.cool/) |  [GitHub](https://github.com/fengziyue/CVRecon) |
| 2023 | ICCV | Zip-NeRF | [Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields](https://arxiv.org/abs/2404.10772)     | [Project](https://jonbarron.info/zipnerf/) |  [GitHub](https://github.com/SuLvXiangXin/zipnerf-pytorch) |
| 2023 | CVPR | F2-NeRF | [F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories](https://arxiv.org/abs/2303.15951)     | [Project](https://totoro97.github.io/projects/f2-nerf/) |  [GitHub](https://github.com/Totoro97/f2-nerf) |
| 2023 | CVPR | DG-Recon | [DG-Recon: Depth-Guided Neural 3D Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/papers/Ju_DG-Recon_Depth-Guided_Neural_3D_Scene_Reconstruction_ICCV_2023_paper.pdf)     | | |
| 2023 | CVPR | VisFusion | [VisFusion: Visibility-aware Online 3D Scene Reconstruction from Videos](https://arxiv.org/abs/2304.10687)     | [Project](https://huiyu-gao.github.io/visfusion/) |  [GitHub](https://github.com/huiyu-gao/VisFusion) |
| 2023 | AAAI | Flora | [Flora: Dual-Frequency LOss-Compensated ReAl-Time Monocular 3D Video Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/25358)     | |  |
| 2023 | ICLR | Switch-NeRF | [Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields](https://openreview.net/forum?id=PQ2zoIZqvm)     | [Project](https://mizhenxing.github.io/switchnerf/) |  [GitHub](https://github.com/MiZhenxing/Switch-NeRF) |
| 2024 | ECCV | CityGaussian | [CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians](https://arxiv.org/abs/2404.01133)     | [Project](https://dekuliutesla.github.io/citygs/) |  [GitHub](https://github.com/Linketic/CityGaussian) |
| 2024 |  | Octree-GS | [Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians](https://arxiv.org/abs/2403.17898)     | [Project](https://city-super.github.io/octree-gs/) |  [GitHub](https://github.com/city-super/Octree-GS) |
| 2024 |  | SCALAR-NeRF | [SCALAR-NeRF: SCAlable LARge-scale Neural Radiance Fields for Scene Reconstruction](https://arxiv.org/abs/2311.16657)     |  |  |
| 2024 | Siggraph Asia | | [Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes](https://arxiv.org/abs/2404.10772)     | [Project](https://niujinshuchong.github.io/gaussian-opacity-fields/) |  [GitHub](https://github.com/autonomousvision/gaussian-opacity-fields) |
| 2024 | CVPR | Scaffold-GS | [Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering](https://arxiv.org/abs/2312.00109)     | [Project](https://city-super.github.io/scaffold-gs/) |  [GitHub](https://github.com/city-super/Scaffold-GS) |
| 2024 | CVPR | MonoSelfRecon | [MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views](https://arxiv.org/abs/2404.06753)     |  |  [GitHub](https://github.com/BlarkLee/MonoSelfRecon) |
| 2025 |  | CityGS-X | [CityGS-X: A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction](https://arxiv.org/abs/2503.23044)     | [Project](https://lifuguan.github.io/CityGS-X/) |  [GitHub](https://github.com/gyy456/CityGS-X) |
| 2025 |  | LODGE | [LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](https://arxiv.org/abs/2505.23158)     | [Project](https://lodge-gs.github.io/) |  |
| 2025 | ICLR | CityGaussianV2 | [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](https://arxiv.org/abs/2411.00771)     | [Project](https://dekuliutesla.github.io/CityGaussianV2/) | [GitHub](https://github.com/Linketic/CityGaussian) |
| 2025 | TMM | DetailRecon | [Focusing on Detailed Regions for Online Monocular 3D Reconstruction](https://ieeexplore.ieee.org/document/10855550)     |  |  |
| 2025 | IROS |  | [3D Gaussian Splatting for Fine-Detailed Surface Reconstruction in Large-Scale Scene](https://arxiv.org/abs/2506.17636)     |  |  |
| 2025 | EGSR |  | [Multiview Geometric Regularization of Gaussian Splatting for Accurate Radiance Fields](https://arxiv.org/abs/2506.13508)     |  |  |
| 2025 | Siggraph | | [Photoreal Scene Reconstruction from an Egocentric Device](https://arxiv.org/abs/2506.04444)     | [Project](https://www.projectaria.com/photoreal-reconstruction/) | [GitHub](https://github.com/facebookresearch/egocentric_splats) |


## Level 3 -- 4D dynamic scenes

### General 4D scene reconstruction

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2005 | Siggraph | | [Video-based rendering](https://dl.acm.org/doi/10.1145/1128923.1128969)     |  |  |
| 2017 | CVPR | | [3D Menagerie: Modeling the 3D shape and pose of animals](https://arxiv.org/abs/1611.07700)     |  |  |
| 2020 | NeurIPS | | [Online adaptation for consistent mesh reconstruction in the wild](https://arxiv.org/abs/2012.03196)     |  |  |
| 2020 | CVPR | | [Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera](https://arxiv.org/abs/2004.01294)     | [Project](https://research.nvidia.com/publication/2020-06_novel-view-synthesis-dynamic-scenes-globally-coherent-depths) | [GitHub](https://github.com/visonpon/New-View-Synthesis) |
| 2021 | | | [Neural Trajectory Fields for Dynamic Novel View Synthesis](https://arxiv.org/abs/2105.05994)     | | |
| 2021 | IJCV | | [The Isowarp: The Template-Based Visual Geometry of Isometric Surfaces](https://link.springer.com/article/10.1007/s11263-021-01472-w)     | | |
| 2021 | CVPR | | [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://arxiv.org/abs/2105.02976)     | [Project](https://video-nerf.github.io/) | |
| 2021 | CVPR | LASR | [LASR: Learning Articulated Shape Reconstruction from a Monocular Video](https://arxiv.org/abs/2105.02976)     | [Project](https://lasr-google.github.io/) | [GitHub](https://github.com/google/lasr) |
| 2021 | CVPR | | [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://arxiv.org/abs/2011.13084)     | [Project](https://www.cs.cornell.edu/~zl548/NSFF/) | [GitHub](https://github.com/zhengqili/Neural-Scene-Flow-Fields) |
| 2021 | ICCV | | [Neural radiance flow for 4d view synthesis and video processing](https://arxiv.org/abs/2012.09790)     | [Project](https://yilundu.github.io/nerflow/) | [GitHub](https://github.com/yilundu/nerflow) |
| 2021 | ICCV | Nerfies | [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/abs/2011.12948)     | [Project](https://nerfies.github.io/) | [GitHub](https://github.com/google/nerfies) |
| 2021 | ICCV | | [Dynamic View Synthesis from Dynamic Monocular Video](https://arxiv.org/abs/2105.06468)     | [Project](https://free-view-video.github.io/) | [GitHub](https://github.com/gaochen315/DynamicNeRF) |
| 2021 | ICCV | | [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video](https://arxiv.org/abs/2012.12247)     | [Project](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/) | [GitHub](https://github.com/facebookresearch/nonrigid_nerf) |
| 2021 | Siggraph Asia | HyperNeRF | [HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields](https://arxiv.org/abs/2106.13228)     | [Project](https://hypernerf.github.io/) | [GitHub](https://github.com/google/hypernerf) |
| 2021 | CVPR | D-NeRF | [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)     | [Project](https://www.albertpumarola.com/research/D-NeRF/index.html) | [GitHub](https://github.com/albertpumarola/D-NeRF) |
| 2022 | CVPR | | [ϕ-SfT: Shape-from-Template with a Physics-Based Deformation Model](https://arxiv.org/abs/2203.11938)     | [Project](https://4dqv.mpi-inf.mpg.de/phi-SfT/) | [GitHub](https://github.com/navamikairanda/phi_sft) |
| 2022 | CVPR | BANMo | [BANMo: Building Animatable 3D Neural Models from Many Casual Videos](https://arxiv.org/abs/2112.12761)     | [Project](https://banmo-www.github.io/) | [GitHub](https://github.com/facebookresearch/banmo) |
| 2022 | CVPR | | [Revealing Occlusions with 4D Neural Fields](https://arxiv.org/abs/2204.10916)     | [Project](https://occlusions.cs.columbia.edu/) | [GitHub](https://github.com/basilevh/occlusions-4d) |
| 2023 | | EmerNeRF | [EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision](https://arxiv.org/abs/2311.02077)     | [Project](https://emernerf.github.io/) | [GitHub](https://github.com/NVlabs/EmerNeRF) |
| 2023 | | DeformGS | [DeformGS: Scene Flow in Highly Deformable Scenes for Deformable Object Manipulation](https://arxiv.org/abs/2312.00583)     | [Project](https://md-splatting.github.io/) | [GitHub](https://github.com/momentum-robotics-lab/deformgs) |
| 2023 | | | [Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis](https://arxiv.org/abs/2308.09713)     | [Project](https://dynamic3dgaussians.github.io/) | [GitHub](https://github.com/JonathonLuiten/Dynamic3DGaussians?tab=readme-ov-file) |
| 2023 | | | [An Efficient 3D Gaussian Representation for Monocular/Multi-view Dynamic Scenes](https://openreview.net/forum?id=EcGS75uziT)     | | [GitHub](https://github.com/raven38/EfficientDynamic3DGaussian) |
| 2023 | CVPR | Tensor4D | [Tensor4D : Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering](https://arxiv.org/abs/2211.11610)     | [Project](https://liuyebin.com/tensor4d/tensor4d.html) | [GitHub](https://github.com/DSaurus/Tensor4D) |
| 2023 | CVPR | | [Unbiased 4D: Monocular 4D Reconstruction with a Neural Deformation Model](https://arxiv.org/abs/2206.08368)     | [Project](https://4dqv.mpi-inf.mpg.de/Ub4D/) | [GitHub](https://github.com/ecmjohnson/ub4d) |
| 2023 | CVPR | DyLiN | [DyLiN: Making Light Field Networks Dynamic](https://arxiv.org/abs/2303.14243)     | [Project](https://dylin2023.github.io/) | [GitHub](https://github.com/Heng14/DyLiN) |
| 2023 | CVPR | HexPlane | [HexPlane: A Fast Representation for Dynamic Scenes](https://arxiv.org/abs/2301.09632)     | | [GitHub](https://github.com/Caoang327/HexPlane) |
| 2023 | CVPR | K-Planes | [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://arxiv.org/abs/2301.10241)     | [Project](https://sarafridov.github.io/K-Planes/) | [GitHub](https://github.com/sarafridov/K-Planes) |
| 2023 | CVPR | | [Flow supervision for Deformable NeRF](https://arxiv.org/abs/2303.16333)     | [Project](https://mightychaos.github.io/projects/fsdnerf/) | [GitHub](https://github.com/MightyChaos/fsdnerf) |
| 2023 | CVPR | | [Neural Scene Chronology](https://arxiv.org/abs/2306.07970)     | [Project](https://zju3dv.github.io/neusc/) | [GitHub](https://github.com/zju3dv/NeuSC) |
| 2023 | CVPR | | [Spacetime Surface Regularization for Neural Dynamic Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/papers/Choe_Spacetime_Surface_Regularization_for_Neural_Dynamic_Scene_Reconstruction_ICCV_2023_paper.pdf)     | | |
| 2023 | CVPR | | [Robust Dynamic Radiance Fields](https://arxiv.org/abs/2301.02239)     | [Project](https://robust-dynrf.github.io/) | [GitHub](https://github.com/facebookresearch/robust-dynrf) |
| 2023 | ICCV | PPR | [PPR: Physically Plausible Reconstruction from Monocular Videos](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_PPR_Physically_Plausible_Reconstruction_from_Monocular_Videos_ICCV_2023_paper.html)     | [Project](https://gengshan-y.github.io/ppr/) | [GitHub](https://github.com/gengshan-y/ppr?tab=readme-ov-file) |
| 2023 | ICCV | MonoNeRF | [MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos](https://arxiv.org/abs/2212.13056)     | | [GitHub](https://github.com/tianfr/MonoNeRF) |
| 2024 | CVPR | REACTO | [REACTO: Reconstructing Articulated Objects from a Single Video](https://arxiv.org/abs/2404.11151)     | [Project](https://chaoyuesong.github.io/REACTO/) | [GitHub](https://github.com/ChaoyueSong/REACTO) |
| 2024 | CVPR | | [Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis](https://arxiv.org/abs/2312.16812)     | [Project](https://oppo-us-research.github.io/SpacetimeGaussians-website/) | [GitHub](https://github.com/oppo-us-research/SpacetimeGaussians) |
| 2024 | CVPR | 3DGStream | [3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos](https://arxiv.org/abs/2403.01444)     | [Project](https://sjojok.top/3dgstream/) | [GitHub](https://github.com/SJoJoK/3DGStream) |
| 2024 | CVPR | | [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://arxiv.org/abs/2310.08528)     | [Project](https://guanjunwu.github.io/4dgs/index.html) | [GitHub](https://github.com/hustvl/4DGaussians) |
| 2024 | CVPR | | [Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://arxiv.org/abs/2309.13101)     | [Project](https://ingra14m.github.io/Deformable-Gaussians/) | [GitHub](https://github.com/ingra14m/Deformable-3D-Gaussians) |
| 2024 | CVPR | DaReNeRF | [DaReNeRF: Direction-aware Representation for Dynamic Scenes](https://arxiv.org/abs/2309.13101)     | | |
| 2024 | CVPR | | [Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction](https://arxiv.org/abs/2312.01196)     | [Project](https://geometric-rl.mpi-inf.mpg.de/npg/) | [GitHub](https://github.com/DevikalyanDas/npgs) |
| 2024 | CVPR | | [3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis](https://arxiv.org/abs/2404.06270)     | [Project](https://npucvr.github.io/GaGS/) | [GitHub](https://github.com/zhichengLuxx/GaGS) |
| 2024 | CVPR | SC-GS | [SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes](https://arxiv.org/abs/2312.14937)     | [Project](https://yihua7.github.io/SC-GS-web/) | [GitHub](https://github.com/CVMI-Lab/SC-GS) |
| 2024 | CVPRW | FlowIBR | [FlowIBR: Leveraging Pre-Training for Efficient Neural Image-Based Rendering of Dynamic Scenes](https://arxiv.org/abs/2309.05418)     | [Project](https://flowibr.github.io/) | |
| 2024 | ICLR | | [Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting](https://arxiv.org/abs/2310.10642)     | [Project](https://fudan-zvg.github.io/4d-gaussian-splatting/) | [GitHub](https://github.com/fudan-zvg/4d-gaussian-splatting) |
| 2024 | Siggraph | | [4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes](https://arxiv.org/abs/2402.03307)     | [Project](https://weify627.github.io/4drotorgs/) | [GitHub](https://github.com/weify627/4D-Rotor-Gaussians) |
| 2024 | Siggraph Asia | | [Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos](https://arxiv.org/abs/2406.18717)     | [Project](https://geometry.stanford.edu/projects/dynamic-gaussian-marbles.github.io/) | [GitHub](https://github.com/coltonstearns/dynamic-gaussian-marbles) |
| 2024 |  | MoSca | [MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](https://arxiv.org/abs/2405.17421)     | [Project](https://www.cis.upenn.edu/~leijh/projects/mosca/) | [GitHub](https://github.com/JiahuiLei/MoSca) |
| 2024 |  | Das3R | [DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction](https://arxiv.org/abs/2412.19584)     | | [GitHub](https://github.com/kai422/DAS3R) |
| 2024 |  | | [Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos](https://arxiv.org/abs/2412.03526)     | [Project](https://research.nvidia.com/labs/toronto-ai/bullet-timer/) |  |
| 2024 |  | | [Shape of Motion: 4D Reconstruction from a Single Video](https://arxiv.org/abs/2407.13764)     | [Project](https://shape-of-motion.github.io/) | [GitHub](https://github.com/vye16/shape-of-motion) |
| 2024 | TVCG |  | [Decoupling Dynamic Monocular Videos for Dynamic View Synthesis](https://arxiv.org/abs/2304.01716)     | | |
| 2024 | TMLR  | GaussianFlow | [GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation](https://arxiv.org/abs/2403.12365)     | [Project](https://zerg-overmind.github.io/GaussianFlow.github.io/) | [GitHub](https://github.com/Zerg-Overmind/GaussianFlow) |
| 2024 | ECCV | | [Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](https://arxiv.org/abs/2404.03613)     | [Project](https://jeongminb.github.io/e-d3dgs/) | [GitHub](https://github.com/JeongminB/E-D3DGS) |
| 2024 | NeurIPS | DN-4DGS | [DN-4DGS: Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering](https://arxiv.org/abs/2410.13607)     | | [GitHub](https://github.com/peoplelu/DN-4DGS) |
| 2024 | NeurIPS | MotionGS | [MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting](https://arxiv.org/abs/2410.07707)     | [Project](https://ruijiezhu94.github.io/MotionGS_page/) | [GitHub](https://github.com/RuijieZhu94/MotionGS) |
| 2024 | ICLR |  | [Neural SDF Flow for 3D Reconstruction of Dynamic Scenes](https://openreview.net/forum?id=rzF0R6GOd4)     | | [GitHub](https://github.com/wei-mao-2019/SDFFlow) |
| 2024 | ICLR |  | [Pseudo-Generalized Dynamic View Synthesis from a Video](https://arxiv.org/abs/2310.08587)     | [Project](https://xiaoming-zhao.github.io/projects/pgdvs/) | [GitHub](https://github.com/apple/ml-pgdvs) |
| 2024 | | DynaSurfGS | [DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting](https://arxiv.org/abs/2408.13972)     | [Project](https://open3dvlab.github.io/DynaSurfGS/) | [GitHub](https://github.com/Open3DVLab/DynaSurfGS) |
| 2024 | | st-2dgs | [Space-time 2D Gaussian Splatting for Accurate Surface Reconstruction under Complex Dynamic Scenes](https://arxiv.org/abs/2409.18852)     | [Project](https://tb2-sy.github.io/st-2dgs/) | [GitHub](https://github.com/tb2-sy/st-2dgs) |
| 2024 | | DGNS | [DGNS: Deformable Gaussian Splatting and Dynamic Neural Surface for Monocular Dynamic 3D Reconstruction](https://arxiv.org/abs/2412.03910)     |  | |
| 2025 | ICLR | DG-Mesh  | [Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Dynamic Scenes](https://arxiv.org/abs/2404.12379)     | [Project](https://www.liuisabella.com/DG-Mesh/) | [GitHub](https://github.com/Isabella98Liu/DG-Mesh) |
| 2025 | ICLR | MoDGS  | [MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors](https://arxiv.org/abs/2406.00434)     | [Project](https://modgs.github.io/) | [GitHub](https://github.com/MobiusLqm/MoDGS) |
| 2025 | WACV | AT-GS  | [Adaptive and Temporally Consistent Gaussian Surfels for Multi-view Dynamic Reconstruction](https://arxiv.org/abs/2411.06602)     | [Project](https://fraunhoferhhi.github.io/AT-GS/) | [GitHub](https://github.com/fraunhoferhhi/AT-GS) |
| 2025 | CVPR | SpectroMotion  | [SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](https://arxiv.org/abs/2410.17249)     | [Project](https://cdfan0627.github.io/spectromotion/) | [GitHub](https://github.com/cdfan0627/SpectroMotion) |
| 2025 |  |  | [Light of Normals: Unified Feature Representation for Universal Photometric Stereo](https://arxiv.org/abs/2506.18882)     | [Project](https://houyuanchen111.github.io/lino.github.io/) | [GitHub](https://github.com/houyuanchen111/LINO_UniPS) |

---

### Human-centric dynamic modeling - SMPL


| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2015 | TOG | SMPL | [Smpl: A skinned multi-person linear model](https://dl.acm.org/doi/10.1145/2816795.2818013)     | [Project](https://smpl.is.tue.mpg.de/)  |  |
| 2016 | ECCV | SMPLify | [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://arxiv.org/abs/1607.08128)     | [Project](https://smplify.is.tue.mpg.de/)  | [GitHub](https://github.com/vchoutas/smplify-x) |
| 2018 | CVPR | HMR | [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/abs/1712.06584)     | [Project](https://akanazawa.github.io/hmr/)  | [GitHub](https://github.com/akanazawa/hmr)  |
| 2018 | CVPR | | [Learning to Estimate 3D Human Pose and Shape from a Single Color Image](https://arxiv.org/abs/1805.04092)     |   | |
| 2019 | CVPR | GraphCMR | [Convolutional Mesh Regression for Single-Image Human Shape Reconstruction](https://arxiv.org/abs/1905.03244)     | [Project](https://www.nikoskolot.com/projects/cmr/)  | [GitHub](https://github.com/nkolot/GraphCMR)  |
| 2019 | CVPR | SMPL-X | [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://arxiv.org/abs/1904.05866)     | [Project](https://smpl-x.is.tue.mpg.de/)  | [GitHub](https://github.com/vchoutas/smplify-x)  |
| 2019 | CVPR | HoloPose | [HoloPose: Holistic 3D Human Reconstruction In-The-Wild](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guler_HoloPose_Holistic_3D_Human_Reconstruction_In-The-Wild_CVPR_2019_paper.pdf)     |   |  |
| 2019 | CVPR | | [Learning 3D Human Dynamics from Video](https://arxiv.org/abs/1812.01601)     | [Project](https://akanazawa.github.io/human_dynamics/)  | [GitHub](https://github.com/akanazawa/human_dynamics)  |
| 2019 | ICCV | SPIN | [Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://arxiv.org/abs/1909.12828)     | [Project](https://www.nikoskolot.com/projects/spin/)  | [GitHub](https://github.com/nkolot/SPIN)  |
| 2019 | ICCV | DenseRaC | [DenseRaC: Joint 3D Pose and Shape Estimation by Dense Render-and-Compare](https://arxiv.org/abs/1910.00116)     |   |   |
| 2019 | MM | DaNet | [DaNet: Decompose-and-aggregate Network for 3D Human Shape and Pose Estimation](https://zhanghongwen.cn/pdf/acmmm19DaNet.pdf)     |   |   |
| 2019 | NeurIPS | | [Sim2real transfer learning for 3D human pose estimation: motion to the rescue](https://arxiv.org/abs/1907.02499)     |   |   |
| 2020 | CVPR | DecoMR | [3D Human Mesh Regression with Dense Correspondence](https://arxiv.org/pdf/2006.05734)     |  | [GitHub](https://github.com/zengwang430521/DecoMR?tab=readme-ov-file)  |
| 2020 | CVPR | VIBE | [VIBE: Video Inference for Human Body Pose and Shape Estimation](https://arxiv.org/abs/1912.05656)     |  | [GitHub](https://github.com/mkocabas/VIBE)  |
| 2020 | TOG | PhysCap | [PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time](https://arxiv.org/abs/2008.08880)     | [Project](https://vcai.mpi-inf.mpg.de/projects/PhysCap/)  | [GitHub](https://github.com/soshishimada/PhysCap_demo_release)  |
| 2020 | ECCV | | [Human Body Model Fitting by Learned Gradient Descent](https://arxiv.org/abs/2008.08474)     |  | [GitHub](https://github.com/InpatientJam/Learned-Gradient-Descent)  |
| 2020 | ECCV | HKMR | [Hierarchical Kinematic Human Mesh Recovery](https://arxiv.org/abs/2003.04232)     |  |  |
| 2020 | ECCV | I2L-MeshNet | [I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image](https://arxiv.org/abs/2008.03713)     |  | [GitHub](https://github.com/mks0601/I2L-MeshNet_RELEASE)  |
| 2021 | CVPR | METRO | [End-to-End Human Pose and Mesh Reconstruction with Transformers](https://arxiv.org/abs/2012.09760)     |  | [GitHub](https://github.com/microsoft/MeshTransformer)  |
| 2021 | CVPR | HybrIK | [HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2011.14672)     | [Project](https://jeffli.site/HybrIK/) | [GitHub](https://github.com/jeffffffli/HybrIK)  |
| 2021 | ICCV | MAED | [Encoder-decoder with Multi-level Attention for 3D Human Shape and Pose Estimation](https://arxiv.org/abs/2109.02303)     |  | [GitHub](https://github.com/ziniuwan/maed)  |
| 2021 | ICCV | SPEC | [SPEC: Seeing People in the Wild with an Estimated Camera](https://arxiv.org/abs/2110.00620)     | [Project](https://spec.is.tue.mpg.de/)  | [GitHub](https://github.com/mkocabas/SPEC)  |
| 2021 | ICCV | PyMAF | [PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop](https://arxiv.org/abs/2103.16507)     | [Project](https://zhanghongwen.cn/pymaf/)  | [GitHub](https://github.com/HongwenZhang/PyMAF)  |
| 2021 | ICCV | HuMoR | [HuMoR: 3D Human Motion Model for Robust Pose Estimation](https://arxiv.org/abs/2105.04668)     | [Project](https://geometry.stanford.edu/projects/humor/) | [GitHub](https://github.com/davrempe/humor)  |
| 2021 | ICCV | PARE | [PARE: Part Attention Regressor for 3D Human Body Estimation](https://arxiv.org/abs/2104.08527)     | [Project](https://pare.is.tue.mpg.de/)  | [GitHub](https://github.com/mkocabas/PARE)  |
| 2022 | CVPR | GLAMR | [GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras](https://arxiv.org/abs/2112.01524)     | [Project](https://nvlabs.github.io/GLAMR/)  | [GitHub](https://github.com/NVlabs/GLAMR)  |
| 2022 | ECCV | CLIFF | [CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation](https://arxiv.org/abs/2208.00571)     |  | [GitHub](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)  |
| 2022 | ECCV | D&D | [D&D: Learning Human Dynamics from Dynamic Camera](https://arxiv.org/abs/2209.08790)     | | [GitHub](https://github.com/jeffffffli/DnD)  |
| 2023 | TPAMI | PyMAF-X | [PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images](https://arxiv.org/abs/2207.06400)     | [Project](https://www.liuyebin.com/pymaf-x/)  | [GitHub](https://github.com/HongwenZhang/PyMAF-X)  |
| 2023 | CVPR | TRACE | [TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments](https://arxiv.org/abs/2306.02850)     | [Project](https://www.yusun.work/TRACE/TRACE.html)  | [GitHub](https://github.com/Arthur151/ROMP)  |
| 2023 | CVPR | SLAHMR | [Decoupling Human and Camera Motion from Videos in the Wild](https://arxiv.org/abs/2302.12827)     | [Project](https://vye16.github.io/slahmr/)  | [GitHub](https://github.com/vye16/slahmr)  |
| 2023 | CVPR | IPMAN | [3D Human Pose Estimation via Intuitive Physics](https://arxiv.org/abs/2303.18246)     | [Project](https://ipman.is.tue.mpg.de/)  | [GitHub](https://github.com/sha2nkt/ipman-r)  |
| 2023 | CVPR | NIKI | [NIKI: Neural Inverse Kinematics with Invertible Neural Networks for 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2305.08590)     | | [GitHub](https://github.com/jeffffffli/NIKI)
| 2023 | ICCV | HMR2.0 | [Humans in 4D: Reconstructing and Tracking Humans with Transformers](https://arxiv.org/abs/2305.20091)     | [Project](https://shubham-goel.github.io/4dhumans/)  | [GitHub](https://github.com/shubham-goel/4D-Humans)  |
| 2023 | NeurIPS | SMPLer-X | [SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation](https://arxiv.org/abs/2309.17448)     | [Project](https://caizhongang.com/projects/SMPLer-X/)  | [GitHub](https://github.com/SMPLCap/SMPLer-X)  |
| 2024 | CVPR | TokenHMR | [TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation](https://arxiv.org/abs/2404.16752)     | [Project](https://tokenhmr.is.tue.mpg.de/)  | [GitHub](https://github.com/saidwivedi/TokenHMR)  
| 2024 | CVPR | WHAM | [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531)     | [Project](https://wham.is.tue.mpg.de/)  | [GitHub](https://github.com/yohanshin/WHAM)  
| 2024 | ECCV | TRAM | [TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](https://arxiv.org/abs/2403.17346)     | [Project](https://yufu-wang.github.io/tram4d/)  | [GitHub](https://github.com/yufu-wang/tram)
| 2024 | ECCV | COIN | [COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimatio](https://arxiv.org/abs/2408.16426)     | [Project](https://nvlabs.github.io/COIN/)  |
| 2024 | NeurIPS | NLF | [Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2407.07532)     | [Project](https://istvansarandi.com/nlf/)  | [GitHub](https://github.com/isarandi/nlf)  |
| 2025 | AAAI | GenHMR | [GenHMR: Generative Human Mesh Recovery](https://arxiv.org/abs/2412.14444)     | [Project](https://m-usamasaleem.github.io/publication/GenHMR/GenHMR.html)  |  |
| 2025 | ICLR | CoMotion | [CoMotion: Concurrent Multi-person 3D Motion](https://arxiv.org/abs/2504.12186)     | | [GitHub](https://github.com/apple/ml-comotion) |
| 2025 | CVPR | BLADE | [BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](https://arxiv.org/abs/2412.08640)     | [Project](https://research.nvidia.com/labs/amri/projects/blade/)  |  |
| 2025 | ICCV | GENMO | [GENMO: A GENarlist Model for Human MOtion](https://arxiv.org/abs/2505.01425)     | [Project](https://research.nvidia.com/labs/dair/genmo/)  |  |

---

### Human-centric dynamic modeling - Egocentric


| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2022 | ECCV | AvatarPoser | [Avatarposer: Articulated full-body pose tracking from sparse motion sensing](https://arxiv.org/abs/2207.13784)     | | [GitHub](https://github.com/eth-siplab/AvatarPoser) |
| 2023 | CVPR | EgoEgo | [Ego-Body Pose Estimation via Ego-Head Pose Estimation](https://arxiv.org/abs/2212.04636)     | [Project](https://lijiaman.github.io/projects/egoego/)  | [GitHub](https://github.com/lijiaman/egoego_release) |
| 2023 | CVPR | BoDiffusion | [BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis](https://arxiv.org/abs/2304.11118)     |  | [GitHub](https://github.com/BCV-Uniandes/BoDiffusion) |
| 2023 | CVPR | SeceneEgo | [Scene-aware egocentric 3d human pose estimation](https://arxiv.org/abs/2212.11684)     |  | [GitHub](https://github.com/jianwang-mpi/SceneEgo) |
| 2023 | CVPR | AGRoL | [Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model](https://arxiv.org/abs/2304.08577)     | [Project](https://dulucas.github.io/agrol/)  | [GitHub](https://github.com/facebookresearch/AGRoL) |
| 2024 | CVPR | EgoWholeBody | [Egocentric Whole-Body Motion Capture with FisheyeViT and Diffusion-Based Motion Refinement](https://arxiv.org/abs/2311.16495)     |  | [GitHub](https://github.com/jianwang-mpi/egowholemocap) |
| 2024 | CVPR | EventEgo3D | [EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams](https://arxiv.org/abs/2404.08640)     | [Project](https://4dqv.mpi-inf.mpg.de/EventEgo3D/) | [GitHub](https://github.com/Chris10M/EventEgo3D) |
| 2025 | 3DV | HMD$`^2`$ | [HMD$`^2`$: Environment-aware Motion Generation from Single Egocentric Head-Mounted Device](https://arxiv.org/abs/2409.13426)     | [Project](https://hmdsquared.github.io/) | |
| 2025 | IJCV | EventEgo3D++ | [EventEgo3D++: 3D Human Motion Capture from a Head Mounted Event Camera](https://arxiv.org/abs/2502.07869)     | [Project](https://eventego3d.mpi-inf.mpg.de/)  | [GitHub](https://github.com/Chris10M/EventEgo3D_plus_plus) |
| 2025 | CVPR | EgoLM | [Egolm: Multi-modal language model of egocentric motions](https://arxiv.org/pdf/2409.18127)     | [Project](https://hongfz16.github.io/projects/EgoLM)  |  |
| 2025 | CVPR | EgoAllo | [Estimating Body and Hand Motion in an Ego-sensed World](https://arxiv.org/abs/2410.03665)     | [Project](https://egoallo.github.io/)  | [GitHub](https://github.com/brentyi/egoallo)  |
| 2025 | CVPR | Ego4o | [Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](https://arxiv.org/abs/2504.08449)     | [Project](https://jianwang-mpi.github.io/ego4o/)  | |
| 2025 | CVPR | FRAME | [FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video](https://arxiv.org/abs/2503.23094)     | [Project](https://vcai.mpi-inf.mpg.de/projects/FRAME/)  | [GitHub](https://github.com/abcamiletto/frame) |

---

### Human-centric dynamic modeling - Appearance-rich


| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2018 | TOG | MonoPerfCap | [MonoPerfCap: Human Performance Capture from Monocular Video](https://arxiv.org/abs/1708.02136)     | | |
| 2018 | CVPR | VideoAvatars | [Video Based Reconstruction of 3D People Models](https://arxiv.org/abs/1803.04758)     | | [GitHub](https://github.com/thmoa/videoavatars) |
| 2018 | TOG | LiveCap | [LiveCap: Real-time Human Performance Capture from Monocular Video](https://arxiv.org/abs/1810.02648)     | |  |
| 2021 | 3DV |  | [Human Performance Capture from Monocular Video in the Wild](https://arxiv.org/abs/2111.14672)     | [Project](https://ait.ethz.ch/human-performance-capture) | [GitHub](https://github.com/MoyGcc/hpcwild) |
| 2021 | NeurIPS | A-NeRF | [A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose](https://arxiv.org/abs/2102.06199)     | [Project](https://lemonatsu.github.io/anerf/) | [GitHub](https://github.com/LemonATsu/A-NeRF) |
| 2021 | NeurIPS | | [ViSER: Video-Specific Surface Embeddings for Articulated 3D Shape Reconstruction](https://openreview.net/forum?id=-JJy-Hw8TFB)     | | [GitHub](https://github.com/gengshan-y/viser) |
| 2022 | CVPR | SelfRecon | [SelfRecon: Self Reconstruction Your Digital Avatar from Monocular Video](https://arxiv.org/abs/2201.12792)     | [Project](https://jby1993.github.io/SelfRecon/) | [GitHub](https://github.com/jby1993/SelfReconCode) |
| 2022 | ECCV | DANBO | [DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks](https://arxiv.org/abs/2205.01666)     | [Project](https://lemonatsu.github.io/danbo/) | [GitHub](https://github.com/LemonATsu/DANBO-pytorch) |
| 2022 | ECCV | AvatarPoser | [Avatarposer: Articulated full-body pose tracking from sparse motion sensing](https://arxiv.org/abs/2207.13784)     | | [GitHub](https://github.com/eth-siplab/AvatarPoser) |
| 2022 | NeurIPS | FOF | [FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction](https://arxiv.org/abs/2206.02194)     | [Project](https://cic.tju.edu.cn/faculty/likun/projects/FOFX/index.html) | [GitHub](https://github.com/fengq1a0/FOF) |
| 2023 | CVPR | Vid2Avatar | [Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition](https://arxiv.org/abs/2302.11566)     | [Project](https://moygcc.github.io/vid2avatar/) | [GitHub](https://github.com/MoyGcc/vid2avatar) |
| 2024 | CVPR | HUGS  | [HUGS: Human Gaussian Splats](https://arxiv.org/abs/2311.17910)     | [Project](https://machinelearning.apple.com/research/hugs)  | [GitHub](https://github.com/apple/ml-hugs)  |
| 2024 | CVPR | GaussianAvatar  | [GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians](https://arxiv.org/abs/2312.02134)     | [Project](https://huliangxiao.github.io/GaussianAvatar)  | [GitHub](https://github.com/aipixel/GaussianAvatar)  |
| 2024 | CVPR | Animatable Gaussians  | [Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling](https://arxiv.org/pdf/2311.16096)     | [Project](https://animatable-gaussians.github.io/)  | [GitHub](https://github.com/lizhe00/AnimatableGaussians?tab=readme-ov-file)  |
| 2024 | CVPR | GPS-Gaussian  | [GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis](https://arxiv.org/abs/2312.02155)     | [Project](https://shunyuanzheng.github.io/GPS-Gaussian)  | [GitHub](https://github.com/aipixel/GPS-Gaussian)  |
| 2024 | CVPR | 3DGS-Avatar  | [3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting](https://arxiv.org/abs/2312.09228)     | [Project](https://neuralbodies.github.io/3DGS-Avatar/)  | [GitHub](https://github.com/mikeqzy/3dgs-avatar-release)  |
| 2025 | | GauSTAR  | [GauSTAR: Gaussian Surface Tracking and Reconstruction](https://arxiv.org/abs/2501.10283)     |  |  |



## Level 4 -- Interaction among scene components

### SMPL-based human-centric interaction - HOI

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2016 | TOG | PiGraphs | [PiGraphs: learning interaction snapshots from observations](https://dl.acm.org/doi/10.1145/2897824.2925867)     | | |
| 2020 | ECCV | PHOSA | [Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild](https://arxiv.org/abs/2007.15649)     | [Project](https://jasonyzhang.com/phosa/) | [GitHub](https://github.com/facebookresearch/phosa) |
| 2021 | | D3D-HOI | [3d-hoi: Dynamic 3d human-object interactions from videos](https://arxiv.org/abs/2108.08420)     | | [GitHub](https://github.com/facebookresearch/d3d-hoi) |
| 2021 | CVPR | GraviCap | [Gravity-Aware Monocular 3D Human-Object Reconstruction](https://arxiv.org/abs/2108.08844)     | [Project](https://4dqv.mpi-inf.mpg.de/GraviCap/) | [GitHub](https://github.com/rishabhdabral/gravicap) |
| 2022 | CVPR | BEHAVE | [BEHAVE: Dataset and Method for Tracking Human Object Interactions](https://arxiv.org/abs/2204.06950)     | [Project](https://virtualhumans.mpi-inf.mpg.de/behave/) | [GitHub](https://github.com/xiexh20/behave-dataset) |
| 2022 | ECCV | CHORE | [CHORE: Contact, Human and Object REconstruction from a single RGB image](https://arxiv.org/abs/2204.02445)     | [Project](https://virtualhumans.mpi-inf.mpg.de/chore/) | [GitHub](https://github.com/xiexh20/CHORE) |
| 2023 | ICCV | CHAIRS | [Full-Body Articulated Human-Object Interaction](https://arxiv.org/abs/2212.10621)     | [Project](https://yzhu.io/publication/hoi2023iccv/) | [GitHub](https://github.com/jnnan/chairs) |
| 2023 | ICCV | Humans in 4D | [Humans in 4D: Reconstructing and Tracking Humans with Transformers](https://arxiv.org/abs/2305.20091)     | [Project](https://shubham-goel.github.io/4dhumans/) | [GitHub](https://github.com/shubham-goel/4D-Humans) |
| 2023 | IJCAI | StackFLOW | [StackFLOW: Monocular Human-Object Reconstruction by Stacked Normalizing Flow with Offset](https://arxiv.org/abs/2407.20545)     | [Project](https://huochf.github.io/StackFLOW/) | [GitHub](https://github.com/huochf/StackFLOW) |
| 2023 | CVPR | VisTracker | [Visibility Aware Human-Object Interaction Tracking from Single RGB Camera](https://arxiv.org/abs/2303.16479)     | [Project](https://virtualhumans.mpi-inf.mpg.de/VisTracker/) | [GitHub](https://github.com/xiexh20/VisTracker) |
| 2024 | IJCV | InterCap | [InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction](https://arxiv.org/abs/2209.12354)     | [Project](https://intercap.is.tue.mpg.de/) | [GitHub](https://github.com/YinghaoHuang91/InterCap/tree/master) |
| 2024 | MM | WildHOI | [Monocular Human-Object Reconstruction in the Wild](https://arxiv.org/abs/2407.20566)     | [Project](https://huochf.github.io/WildHOI/) | [GitHub](https://github.com/huochf/WildHOI) |
| 2024 | CVPR | I'M HOI | [I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions](https://arxiv.org/abs/2312.08869)     | [Project](https://afterjourney00.github.io/IM-HOI.github.io/) | [GitHub](https://github.com/AfterJourney00/IMHD-Dataset) |
| 2024 | CVPR | HDM | [Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation](https://arxiv.org/abs/2312.07063)     | [Project](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/) | [GitHub](https://github.com/xiexh20/HDM) |
| 2024 |  | InterTrack | [InterTrack: Tracking Human Object Interaction without Object Templates](https://arxiv.org/abs/2408.13953)     | [Project](https://virtualhumans.mpi-inf.mpg.de/InterTrack/) | |
| 2024 | NeurIPS | InterDreamer | [InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction](https://arxiv.org/abs/2403.19652)     | [Project](https://sirui-xu.github.io/InterDreamer/) | |


### SMPL-based human-centric interaction - HSI

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2019 | ICCV | PROX | [Resolving 3D Human Pose Ambiguities with 3D Scene Constraints](https://arxiv.org/abs/1908.06963)     | [Project](https://prox.is.tue.mpg.de/) | [GitHub](https://github.com/mohamedhassanmus/prox) |
| 2020 | ECCV | HMP | [Long-term Human Motion Prediction with Scene Context](https://arxiv.org/abs/2007.03672)     | [Project](https://zhec.github.io/hmp/) | [GitHub](https://github.com/ZheC/GTA-IM-Dataset) |
| 2022 | CVPR | RICH | [Capturing and Inferring Dense Full-Body Human-Scene Contact](https://arxiv.org/abs/2206.09553)     | [Project](https://rich.is.tue.mpg.de/) | [GitHub](https://github.com/paulchhuang/bstro) |
| 2022 | ECCV | SitComs3D | [The One Where They Reconstructed 3D Humans and Environments in TV Shows](https://arxiv.org/abs/2204.06950)     | [Project](https://ethanweber.me/sitcoms3D/) | [GitHub](https://github.com/ethanweber/sitcoms3D) |
| 2023 | CVPR | CIRCLE | [CIRCLE: Capture In Rich Contextual Environments](https://arxiv.org/abs/2303.17912)     | | [GitHub](https://github.com/Stanford-TML/circle_dataset) |
| 2024 | CVPR | TRUMANS | [Scaling Up Dynamic Human-Scene Interaction Modeling](https://arxiv.org/pdf/2403.08629)     | [Project](https://jnnan.github.io/trumans/) | [GitHub](https://github.com/jnnan/trumans_utils) |
| 2025 | | JOSH | [InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction](https://arxiv.org/abs/2501.02158)     | [Project](https://genforce.github.io/JOSH/) | [GitHub](https://github.com/genforce/JOSH) |
| 2025 | CVPR | ODHSR | [ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](https://arxiv.org/abs/2504.13167)     | [Project](https://eth-ait.github.io/ODHSR/) | |


### SMPL-based human-centric interaction - HHI

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2021 | ICCV | ROMP | [Monocular, One-stage, Regression of Multiple 3D People](https://arxiv.org/abs/2008.12272)     | | [GitHub](https://github.com/Arthur151/ROMP) |
| 2022 | CVPR | BEV | [Putting People in their Place: Monocular Regression of 3D People in Depth](https://arxiv.org/abs/2112.08274)     | [Project](https://www.yusun.work/BEV/BEV.html) | [GitHub](https://github.com/Arthur151/ROMP) |
| 2023 | Siggraph Asia | CloseMoCap | [Reconstructing Close Human Interactions from Multiple Views](https://arxiv.org/abs/2401.16173)     | | [GitHub](https://github.com/zju3dv/CloseMoCap) |
| 2023 | CVPR| Hi4D | [Hi4D: 4D Instance Segmentation of Close Human Interaction](https://arxiv.org/abs/2303.15380)     | [Project](https://yifeiyin04.github.io/Hi4D/) | [GitHub](https://github.com/yifeiyin04/Hi4D) |
| 2024 | CVPR | BUDDI | [Generative Proxemics: A Prior for 3D Social Interaction from Images](https://arxiv.org/abs/2306.09337)     | [Project](https://muelea.github.io/buddi/) | [GitHub](https://github.com/muelea/buddi) |
| 2024 | CVPR | CloseInt | [Closely Interactive Human Reconstruction with Proxemics and Physics-Guided Adaption](https://arxiv.org/abs/2404.11291)     | | [GitHub](https://github.com/boycehbz/HumanInteraction) |
| 2024 | CVPR | MultiPhys | [MultiPhys: Multi-Person Physics-aware 3D Motion Estimation](https://arxiv.org/abs/2404.11987)     | [Project](https://www.iri.upc.edu/people/nugrinovic/multiphys/) | [GitHub](https://github.com/nicolasugrinovic/multiphys) |
| 2024 | ECCV | AvatarPose | [AvatarPose: Avatar-guided 3D Pose Estimation of Close Human Interaction from Sparse Multi-view Videos](https://arxiv.org/abs/2408.02110)     | [Project](https://eth-ait.github.io/AvatarPose/) | [GitHub](https://github.com/eth-ait/AvatarPose) |
| 2024 | NeurIPS | Harmony4D | [Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions](https://arxiv.org/abs/2410.20294)     | [Project](https://jyuntins.github.io/harmony4d/) | [GitHub](https://github.com/jyuntins/harmony4d) |

---

### Appearance-rich human-centric interaction

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2022 | ECCV | NeuMan | [NeuMan: Neural Human Radiance Field from a Single Video](https://arxiv.org/abs/2203.12575)     | [Project](https://machinelearning.apple.com/research/neural-human-radiance-field) | [GitHub](https://github.com/apple/ml-neuman) |
| 2023 | ICCV | HOSNeRF | [HOSNeRF: Dynamic Human-Object-Scene Neural Radiance Fields from a Single Video](https://arxiv.org/abs/2304.12281)     | [Project](https://showlab.github.io/HOSNeRF/) | [GitHub](https://github.com/TencentARC/HOSNeRF) |

---

### Egocentric human-centric interaction

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2021 | ICCV | H2O | [H2O: Two Hands Manipulating Objects for First Person Interaction Recognition](https://arxiv.org/abs/2104.11181)     |  | |
| 2022 | CVPR | HOI4D | [HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction](https://arxiv.org/abs/2203.01577)     |  | |
| 2023 |  | Aria | [Project Aria: A New Tool for Egocentric Multi-Modal AI Research](https://arxiv.org/abs/2308.13561)     |  | |
| 2023 | MICCAI  | POV-Surgery | [POV-Surgery: A Dataset for Egocentric Hand and Tool Pose Estimation During Surgical Activities](https://arxiv.org/abs/2307.10387)     | [Project](https://batfacewayne.github.io/POV_Surgery_io/)  | [GitHub](https://github.com/BatFaceWayne/POV_Surgery) |
| 2024 | CVPR  | Ego-Exo4D | [Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives](https://arxiv.org/abs/2311.18259)     | [Project](https://ego-exo4d-data.org/)  | |
| 2024 |   | Nymeria | [Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](https://arxiv.org/abs/2406.09905)     | [Project](https://www.projectaria.com/datasets/nymeria/)  | |
| 2025 | CVPR | HOT3D | [Introducing HOT3D: An Egocentric Dataset for 3D Hand and Object Tracking](https://arxiv.org/abs/2406.09598)     | [Project](https://facebookresearch.github.io/hot3d/)  | |



## Level 5 -- Incorporation of physical laws and constraints


### Dynamic 4D human simulation with physics

| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 1995 | Siggraph  | | [Animating human athletics](https://arxiv.org/abs/2302.06108)     | | |
| 2002 | TOG   | | [Interactive control of avatars animated with human motion data](https://graphics.cs.cmu.edu/projects/Avatar/avatar.pdf)     | | |
| 2007 | TOG  | SIMBICON | [Simbicon: Simple biped locomotion control](https://dl.acm.org/doi/10.1145/1276377.1276509)     | | |
| 2007 | Siggraph  | | [Construction and optimal search of interpolated motion graphs](https://dl.acm.org/doi/abs/10.1145/1276377.1276510)     | | |
| 2007 | Siggraph  | | [Near-optimal Character Animation with Continuous Control](https://dl.acm.org/doi/10.1145/1276377.1276386)     | | |
| 2010 | Siggraph Asia   | | [Motion fields for interactive character locomotion](https://dl.acm.org/doi/10.1145/1882261.1866160)     | | |
| 2010 | TOG  | | [Generalized biped walking control](https://dl.acm.org/doi/10.1145/1778765.1781156)     | | |
| 2010 | TOG  | | [Spatial relationship preserving character motion adaptation](https://dl.acm.org/doi/10.1145/1778765.1778770)     | | |
| 2012 | TOG   | | [Continuous character control with low-dimensional embeddings](https://dl.acm.org/doi/abs/10.1145/2185520.2185524)     | | |
| 2014 | TOG   | | [Learning bicycle stunts](https://dl.acm.org/doi/10.1145/2601097.2601121)     | | |
| 2016 | NeurIPS   | GAIL | [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)     | | |
| 2017 | TOG   | PFNN | [Phase-functioned neural networks for character control](https://dl.acm.org/doi/10.1145/3072959.3073663)     | | [GitHub](http://github.com/sreyafrancis/PFNN) |
| 2017 | TOG   | | [Learning to schedule control fragments for physics-based characters using deep q-learning](https://dl.acm.org/doi/10.1145/3083723)     | | |
| 2018 | TOG   | MANN | [Mode-adaptive neural networks for quadruped motion control](https://dl.acm.org/doi/10.1145/3197517.3201366)     | | [GitHub](https://github.com/cghezhang/MANN) |
| 2018 | TOG   | | [Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning](https://dl.acm.org/doi/10.1145/3197517.3201315)     | | |
| 2018 | TOG   | DeepMimic | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717)     | | [GitHub](https://github.com/xbpeng/DeepMimic) |
| 2020 | | UniCon | [UniCon: Universal Neural Controller For Physics-based Character Motion](https://arxiv.org/abs/2011.15119)     | [Project](https://research.nvidia.com/labs/toronto-ai/unicon/) | |
| 2020 | TOG | ScaDiver | [A scalable approach to control diverse behaviors for physically simulated characters](https://dl.acm.org/doi/10.1145/3386569.3392381)     | [Project](https://research.facebook.com/publications/a-scalable-approach-to-control-diverse-behaviors-for-physically-simulated-characters/) | [GitHub](https://github.com/facebookresearch/ScaDiver) |
| 2020 | Siggraph   | MotionVAEs | [Character Controllers using Motion VAEs](https://arxiv.org/abs/2103.14274)     | | [GitHub](https://github.com/electronicarts/character-motion-vaes) |
| 2021 | TOG | AMP | [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180)     |  |  |
| 2022 | NeurIPS | MoCapAct | [MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control](https://arxiv.org/abs/2208.07363)     | [Project](https://microsoft.github.io/MoCapAct/) | [GitHub](https://github.com/microsoft/MoCapAct) |
| 2022 | Siggraph Asia | PADL | [PADL: Language-Directed Physics-Based Character Control](https://arxiv.org/abs/2301.13868)     | | [GitHub](https://github.com/nv-tlabs/PADL) |
| 2022 | TOG | ASE | [ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters](https://arxiv.org/abs/2205.01906)     | [Project](https://xbpeng.github.io/projects/ASE/index.html) | [GitHub](https://github.com/nv-tlabs/ASE) |
| 2022 | TOG | ControlVAE | [ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters](https://arxiv.org/abs/2210.06063)     | [Project](https://heyuanyao-pku.github.io/Control-VAE/) | [GitHub](https://github.com/heyuanYao-pku/Control-VAE) |
| 2023 | Siggraph | CALM | [CALM: Conditional Adversarial Latent Models for Directable Virtual Characters](https://arxiv.org/abs/2305.02195)     | [Project](https://research.nvidia.com/labs/par/calm/) | [GitHub](https://github.com/NVlabs/CALM) |
| 2023 | Siggraph | | [Simulation and Retargeting of Complex Multi-Character Interactions](https://arxiv.org/abs/2305.20041)     | | [GitHub](https://github.com/JackZhouSz/InteractionGraph) |
| 2023 | Siggraph | PMP | [PMP: Learning to Physically Interact with Environments using Part-wise Motion Priors](https://arxiv.org/abs/2305.03249)     | | [GitHub](https://github.com/jinseokbae/pmp) |
| 2023 | CVPR   | | [Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion](https://arxiv.org/abs/2304.01893)     | [Project](https://research.nvidia.com/labs/toronto-ai/trace-pace/) | [GitHub](https://github.com/nv-tlabs/trace) |
| 2023 | TOG   | | [Learning Physically Simulated Tennis Skills from Broadcast Videos](https://dl.acm.org/doi/10.1145/3592408)     | [Project](https://research.nvidia.com/labs/toronto-ai/vid2player3d/) | [GitHub](https://github.com/nv-tlabs/vid2player3d) |
| 2024 | 3DV   | | [Physically Plausible Full-Body Hand-Object Interaction Synthesis](https://arxiv.org/abs/2309.07907)     | [Project](https://eth-ait.github.io/phys-fullbody-grasp/) | |
| 2024 | NeurIPS   | Omnigrasp | [Omnigrasp: Grasping Diverse Objects with Simulated Humanoids](https://arxiv.org/abs/2407.11385)     | | [GitHub](https://github.com/ZhengyiLuo/Omnigrasp) |
| 2024 | Siggraph   | SuperPADL | [SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation](https://arxiv.org/abs/2407.10481)     | [Project](https://xbpeng.github.io/projects/SuperPADL/) | |
| 2024 | Siggraph Asia  | PDP | [PDP: Physics-Based Character Animation via Diffusion Policy](https://arxiv.org/abs/2406.00960)     |  | |
| 2024 | TOG   | MaskedMimic | [MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting](https://arxiv.org/abs/2409.14393)     | [Project](https://research.nvidia.com/labs/par/maskedmimic/) | [GitHub](https://github.com/NVlabs/ProtoMotions) |
| 2024 | CVPR   | PACER++ | [PACER+: On-Demand Pedestrian Animation Controller in Driving Scenarios](https://arxiv.org/abs/2404.19722)     | | [GitHub](https://github.com/IDC-Flash/PacerPlus) |
| 2025 | ICLR   | CLoSD | [CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control](https://arxiv.org/abs/2410.03441)     | [Project](https://guytevet.github.io/CLoSD-page/) | [GitHub](https://github.com/GuyTevet/CLoSD) |
| 2025 | ICLR   | | [Hierarchical World Models as Visual Whole-Body Humanoid Controllers](https://arxiv.org/abs/2405.18418)     | [Project](https://www.nicklashansen.com/rlpuppeteer/) | [GitHub](https://github.com/nicklashansen/puppeteer) |
| 2025 | CVPR   | SkillMimic | [SkillMimic: Learning Basketball Interaction Skills from Demonstrations](https://arxiv.org/abs/2408.15270)     | [Project](https://ingrid789.github.io/SkillMimic/) | [GitHub](https://github.com/wyhuai/SkillMimic) |
| 2025 | ICRA   | HOVER | [HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots](https://arxiv.org/abs/2410.21229)     | [Project](https://hover-versatile-humanoid.github.io/) | [GitHub](https://github.com/NVlabs/HOVER/) |
| 2025 | RSS   | ASAP | [ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills](https://arxiv.org/abs/2502.01143)     | [Project](https://agile.human2humanoid.com/) | [GitHub](https://github.com/LeCAR-Lab/ASAP) |
| 2025 |   | UniPhys | [UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control](https://arxiv.org/abs/2504.12540)     | [Project](https://wuyan01.github.io/uniphys-project/) | |

---

### 3D scene reconstruction with physical plausibility
| Year | Venue | Acronym | Paper | Project | GitHub |
|------|-------|---------|-------|---------|-------------|
| 2022 | CVPR   | AugNeRF | [Aug-NeRF: Training Stronger Neural Radiance Fields with Triple-Level Physically-Grounded Augmentations](https://arxiv.org/abs/2207.01164)     | | [GitHub](https://github.com/VITA-Group/Aug-NeRF) |
| 2024 | ECCV   | PhysDreamer | [PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](https://arxiv.org/abs/2404.13026)     | [Project](https://physdreamer.github.io/) | [GitHub](https://github.com/a1600012888/PhysDreamer) |
| 2024 | Siggraph Asia   | | [Planar Reflection-Aware Neural Radiance Fields](https://arxiv.org/abs/2411.04984)     | | |
| 2024 | NeurIPS   | PhyRecon | [Phyrecon: Physically plausible neural scene reconstruction](https://arxiv.org/abs/2404.16666)     | [Project](https://phyrecon.github.io/) | [GitHub](https://github.com/PhyRecon/PhyRecon) |
| 2025 | ICML   | PhysicsNeRF | [PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views](https://arxiv.org/abs/2505.23481)     | | [GitHub](https://github.com/bmrayan/PhysicsNeRF) |
| 2025 | CVPR   | PBR-NeRF | [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](https://arxiv.org/abs/2505.23481)     | | [GitHub](https://github.com/s3anwu/pbrnerf) |
| 2025 |   | CAST | [CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image](https://arxiv.org/abs/2502.12894)     | [Project](https://sites.google.com/view/cast4) | |
