# MobReID
Created for NSF's 2025 REU program for CS3

Abstract: A common challenge in pedestrian tracking and detection is the inability to consistently identify the same individual over different images or video frames, typically due to occlusions and differing camera angles. In response, a computer vision technique titled person re-identification (Re-ID) has been extensively researched in order to alleviate the issue. Re-ID is a computer vision process that extracts the feature representation of a query image to compare to a gallery of other images with the goal of identifying the same individual across multiple cameras. The topic has become more prominent in research due to innovations in performance and its high utility, specifically in areas where data extraction from video footage is necessary, such as the creation of digital twins. 
Even regarding recent advancements in Re-ID model performance, there is no perfect solution to matching IDs across cameras, especially across videos. To address the difficulty of consistent identification tracking across multiple cameras, specifically in videos rather than images, we present MobReID, a video processing module that leverages FastReID to assist with inter-camera identity matching. Additionally, we explore the impact of camera position and camera overlap on MobReID's performance, questioning whether overlapping camera regions can affect model performance. Altogether, we integrate FastReID’s identity matching with YOLO11’s object tracking to consistently identify and track pedestrians across multiple cameras oriented at different angles. 

Note: Must be ran in its own folder within the FastReID repo

@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
