# 3DeepVOG (Code will be released after paper is accepted) 

**Deep learning-based 3D monocular eye tracking (horizontal, vertical, torsional)**  
Accurate, real-time open-source eye movement analysis for clinical and research applications.

---

## Paper Abstract

**Objective**: Eye movements are key biomarkers for diagnosing and monitoring neuro-otological, neuro-ophthalmological and neurodegenerative disorders. Apparative video-oculography (VOG) systems afford the detection and quantification of small-amplitude and rapid eye movements, as well as subtle oculomotor pathologies that may not be evident during clinical examination. However, these systems typically require high-quality input data for accurate pupil tracking and often show limited reliability in capturing torsional movements. High system costs further constrain their use in broader clinical and research contexts.

**Methods**: To overcome these limitations, we developed **3DeepVOG**, a deep learning-based framework for **three-dimensional monocular eye tracking** (horizontal, vertical, and torsional rotation) designed to operate robustly across varied imaging conditions, including low-light and noisy environments. The method includes automated framewise segmentation of the pupil and iris from video frames, followed by geometrically interpretable gaze estimation based on a two-sphere anatomical eyeball model incorporating corneal refraction correction. Torsion is tracked in real time using a mini-iris-patch template matching approach. The system was trained on over **24,000 annotated samples** obtained across multiple devices and clinical scenarios. Application was tested against a gold-standard VOG system in healthy controls.

**Results**: 3DeepVOG operates in real time (>300 fps) and achieves mean gaze errors of approximately **0.1°** in all three motion dimensions. Derived oculomotor metrics – such as **saccadic peak velocity**, **smooth pursuit gain**, and **optokinetic nystagmus slow-phase velocity** – show good-to-excellent agreement with results from a clinical gold-standard system.

**Conclusions**: 3DeepVOG enables accurate, quantitative eye movement tracking across three dimensions under diverse conditions. As an open-source framework, it provides an accessible and scalable tool for advancing research and clinical assessment in neurological oculomotor disorders.

---

## Features

✅ 3D gaze estimation: horizontal, vertical, torsional  
✅ Deep learning-based segmentation of pupil & iris  
✅ Real-time processing (>300 fps)  
✅ Robust to low-light & noisy video  
✅ Validated against the clinical gold-standard VOG system  
✅ Open-source & extensible  

---

## System Requirements

- **Python 3.11**
- **PyTorch**
- **MONAI**
- **OpenCV**
- **scikit-image**
- **Kornia**
- **pye3d**
- Tested on:  
  - GPU: **NVIDIA GeForce RTX 4090**  
  - CPU: **AMD Ryzen 9 7950X3D**  
  - OS: Windows / Linux  

---

## Installation

```bash
git clone https://github.com/DSGZ-MotionLab/3DeepVOG.git

cd 3DeepVOG
# create environment (example using conda)
conda create -n 3deepvog python=3.11
conda activate 3deepvog
# install dependencies
pip install -r requirements.txt
```

## Example usage
```bash
python run_demo.py --input_video myvideo.mp4 --output_dir results/
```

## License
This project is licensed under the Apache License Version 2.0.

## Acknowledgements
Developed at LMU Klinikum
Clinical Open Research Engine (CORE)
Supported by the German Space Agency (DLR) on behalf of the Federal Ministry of Economics and Technology/Energy (50WB2236) and by the German Federal Ministry of Education and Research (13GW0490B).


## Contact
For questions or collaborations:
E-mail: Jingkang.Zhao@med.uni-muenchen.de
