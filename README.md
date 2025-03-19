## Argo

<img src="[drawing.jpg](https://github.com/user-attachments/assets/8eb6a4c1-bbcc-486d-92a5-6c24e6e2f0be)" alt="argo" width="200"/>

> Argo, the ship that carried Jason and the Argonauts on their quest for the Golden Fleece

This is a playground to re-implement model architectures from industry/academic papers in Pytorch. The primary goal is educational and the target audiance is for those who would like to start journey in machine learning & machine learning infra. The code implementation is optimized for readablility and explainability, while not for the best of performance.

### Repo structure
- data: functions for dataset management, such as downloading public dataset, cache management, etc
- model: model code implementation
- trainer: simple wrapper around train/val/eval loop

### Papers
- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### Road Map

- [ ] Deep Interest Network E2E ~training~ & inference example, MovieLen Small
- [ ] Deep Interest Network Scaling, MovieLen 32M dataset
